// Copyright — see /LICENSE for details.

#include "MotionReceiverSubsystem.h"

#include "Common/UdpSocketBuilder.h"
#include "HAL/IConsoleManager.h"
#include "IPAddress.h"
#include "Interfaces/IPv4/IPv4Address.h"
#include "Interfaces/IPv4/IPv4Endpoint.h"
#include "SocketSubsystem.h"
#include "Sockets.h"

DEFINE_LOG_CATEGORY_STATIC(LogMotionRetarget, Log, All);

namespace
{
	// Wire format v2 (variable joint count per person):
	//   uint32 frame_id
	//   uint8  person_count
	//   per person:
	//     uint16 pid, uint8 joint_count, float32 x3 root, float32 x(joint_count*4) quats
	//   joint_count = 24 for SMPL, 55 for SMPL-X.
	constexpr int32 HeaderSize = 5;                         // 4 + 1
	constexpr int32 PersonHeadSize = 2 + 1;                 // pid + joint_count
	constexpr int32 MaxUdpPayload = 65507;
	constexpr int32 RecvBufferSize = 1 << 20;
	constexpr int32 MaxJoints = 128;                        // sanity cap

	template <typename T>
	FORCEINLINE T ReadLE(const uint8*& Cursor)
	{
		T Value;
		FMemory::Memcpy(&Value, Cursor, sizeof(T));
		Cursor += sizeof(T);
		return Value;
	}
}

// -----------------------------------------------------------------------------
// FMotionUdpReceiver
// -----------------------------------------------------------------------------

FMotionUdpReceiver::FMotionUdpReceiver(UMotionReceiverSubsystem* InOwner, int32 InPort, const FString& InBindAddress)
	: Owner(InOwner)
	, Port(InPort)
	, BindAddress(InBindAddress)
{
}

FMotionUdpReceiver::~FMotionUdpReceiver()
{
	if (Socket)
	{
		ISocketSubsystem* SS = ISocketSubsystem::Get(PLATFORM_SOCKETSUBSYSTEM);
		if (SS)
		{
			SS->DestroySocket(Socket);
		}
		Socket = nullptr;
	}
}

bool FMotionUdpReceiver::Init()
{
	UE_LOG(LogMotionRetarget, Display, TEXT("[Receiver] Init entering, target=%s:%d"),
	       *BindAddress, Port);

	FIPv4Address Addr;
	if (!FIPv4Address::Parse(BindAddress, Addr))
	{
		UE_LOG(LogMotionRetarget, Error, TEXT("[Receiver] Invalid BindAddress '%s'"), *BindAddress);
		return false;
	}
	FIPv4Endpoint Endpoint(Addr, static_cast<uint16>(Port));

	Socket = FUdpSocketBuilder(TEXT("MotionRetargetUDP"))
		.AsReusable()
		.BoundToEndpoint(Endpoint)
		.WithReceiveBufferSize(RecvBufferSize)
		.Build();

	if (!Socket)
	{
		UE_LOG(LogMotionRetarget, Error, TEXT("[Receiver] Failed to bind UDP socket on %s:%d"),
		       *BindAddress, Port);
		return false;
	}

	UE_LOG(LogMotionRetarget, Display, TEXT("[Receiver] UDP listener bound on %s:%d — Run() should follow"),
	       *BindAddress, Port);
	return true;
}

uint32 FMotionUdpReceiver::Run()
{
	UE_LOG(LogMotionRetarget, Display, TEXT("[Receiver] Run entering, socket=%s"),
	       Socket ? TEXT("valid") : TEXT("NULL"));
	if (!Socket)
	{
		return 1;
	}

	ISocketSubsystem* SS = ISocketSubsystem::Get(PLATFORM_SOCKETSUBSYSTEM);
	TSharedRef<FInternetAddr> Sender = SS->CreateInternetAddr();

	TArray<uint8> Buffer;
	Buffer.SetNumUninitialized(MaxUdpPayload);

	// Cache per-pid last frame ids to reject stale packets without touching the
	// mutex-guarded map on every UDP arrival. Race window is fine: worst case
	// we push a slightly older frame; the map-level check confirms.
	TMap<int32, uint32> LocalLatestFrame;

	while (!bStopRequested.Load())
	{
		// ``Socket->Wait()`` is unreliable for UDP on some Windows configs — the
		// underlying select() never signals ready even when packets are queued.
		// Poll ``HasPendingData()`` at ~200 Hz instead; RecvFrom itself is a
		// syscall so per-iteration overhead is negligible.
		uint32 PendingBytes = 0;
		if (!Socket->HasPendingData(PendingBytes) || PendingBytes == 0)
		{
			FPlatformProcess::Sleep(0.005f);
			continue;
		}

		int32 BytesRead = 0;
		if (!Socket->RecvFrom(Buffer.GetData(), Buffer.Num(), BytesRead, *Sender))
		{
			continue;
		}
		if (BytesRead < HeaderSize)
		{
			++PacketsDropped;
			continue;
		}

		const uint8* Cursor = Buffer.GetData();
		const uint8* const CursorEnd = Cursor + BytesRead;
		const uint32 FrameId = ReadLE<uint32>(Cursor);
		const uint8 PersonCount = ReadLE<uint8>(Cursor);

		UMotionReceiverSubsystem* OwnerPtr = Owner.Get();
		if (!OwnerPtr)
		{
			continue;   // subsystem gone
		}

		bool bMalformed = false;
		for (uint8 P = 0; P < PersonCount; ++P)
		{
			if (Cursor + PersonHeadSize > CursorEnd) { bMalformed = true; break; }
			FPersonMotionData Data;
			Data.LastFrameId = static_cast<int32>(FrameId);
			Data.PersonId = static_cast<int32>(ReadLE<uint16>(Cursor));
			const uint8 JointCount = ReadLE<uint8>(Cursor);
			if (JointCount == 0 || JointCount > MaxJoints) { bMalformed = true; break; }

			const int32 RemainingNeeded = 12 + JointCount * 16;
			if (Cursor + RemainingNeeded > CursorEnd) { bMalformed = true; break; }

			const float Rx = ReadLE<float>(Cursor);
			const float Ry = ReadLE<float>(Cursor);
			const float Rz = ReadLE<float>(Cursor);
			Data.RootTranslation = FVector(Rx, Ry, Rz);

			Data.BoneRotations.SetNumUninitialized(JointCount);
			for (int32 J = 0; J < JointCount; ++J)
			{
				const float Qx = ReadLE<float>(Cursor);
				const float Qy = ReadLE<float>(Cursor);
				const float Qz = ReadLE<float>(Cursor);
				const float Qw = ReadLE<float>(Cursor);
				Data.BoneRotations[J] = FQuat(Qx, Qy, Qz, Qw);
			}

			// Local stale-frame short-circuit (best-effort; final check inside PushMotionFrame).
			const uint32* Prev = LocalLatestFrame.Find(Data.PersonId);
			if (Prev && *Prev >= FrameId)
			{
				++PacketsDropped;
				continue;
			}
			LocalLatestFrame.Add(Data.PersonId, FrameId);

			OwnerPtr->PushMotionFrame(MoveTemp(Data));
		}
		if (bMalformed) { ++PacketsDropped; continue; }
		++PacketsReceived;
	}

	return 0;
}

void FMotionUdpReceiver::Stop()
{
	bStopRequested.Store(true);
}

void FMotionUdpReceiver::Exit()
{
	if (Socket)
	{
		Socket->Close();
	}
}


// -----------------------------------------------------------------------------
// UMotionReceiverSubsystem
// -----------------------------------------------------------------------------

void UMotionReceiverSubsystem::Initialize(FSubsystemCollectionBase& Collection)
{
	Super::Initialize(Collection);

	UE_LOG(LogMotionRetarget, Display, TEXT("[Subsystem] Initialize: target %s:%d"),
	       *BindAddress, ListenPort);

	Receiver = MakeUnique<FMotionUdpReceiver>(this, ListenPort, BindAddress);
	ReceiverThread = FRunnableThread::Create(
		Receiver.Get(),
		TEXT("MotionRetargetUDP"),
		0,
		TPri_AboveNormal);

	if (!ReceiverThread)
	{
		UE_LOG(LogMotionRetarget, Error, TEXT("[Subsystem] Failed to spawn UDP listener thread."));
		Receiver.Reset();
	}
	else
	{
		UE_LOG(LogMotionRetarget, Display, TEXT("[Subsystem] Receiver thread spawned"));
	}

	// Register `motion.stats` console command so users can eyeball throughput
	// without wiring up a Blueprint. Prints packet counters + per-pid latest.
	StatsCommand = IConsoleManager::Get().RegisterConsoleCommand(
		TEXT("motion.stats"),
		TEXT("Print MotionRetarget UDP listener counters and per-person latest frame."),
		FConsoleCommandDelegate::CreateWeakLambda(this, [this]() { DumpStatsToLog(); }),
		ECVF_Default);
}

void UMotionReceiverSubsystem::Deinitialize()
{
	if (StatsCommand)
	{
		IConsoleManager::Get().UnregisterConsoleObject(StatsCommand);
		StatsCommand = nullptr;
	}

	if (ReceiverThread)
	{
		if (Receiver.IsValid())
		{
			Receiver->Stop();
		}
		ReceiverThread->WaitForCompletion();
		delete ReceiverThread;
		ReceiverThread = nullptr;
	}
	Receiver.Reset();

	{
		FScopeLock Lock(&PerPersonMutex);
		PerPersonLatest.Empty();
	}

	Super::Deinitialize();
}

void UMotionReceiverSubsystem::DumpStatsToLog() const
{
	const int32 Recv = GetPacketsReceived();
	const int32 Drop = GetPacketsDropped();
	UE_LOG(LogMotionRetarget, Display,
	       TEXT("motion.stats: recv=%d drop=%d bind=%s:%d"),
	       Recv, Drop, *BindAddress, ListenPort);

	FScopeLock Lock(&PerPersonMutex);
	if (PerPersonLatest.Num() == 0)
	{
		UE_LOG(LogMotionRetarget, Display, TEXT("  (no persons yet)"));
		return;
	}
	for (const TPair<int32, FPersonMotionData>& Pair : PerPersonLatest)
	{
		UE_LOG(LogMotionRetarget, Display,
		       TEXT("  pid=%d frame=%d root=(%.1f, %.1f, %.1f) quats=%d"),
		       Pair.Key,
		       Pair.Value.LastFrameId,
		       Pair.Value.RootTranslation.X,
		       Pair.Value.RootTranslation.Y,
		       Pair.Value.RootTranslation.Z,
		       Pair.Value.BoneRotations.Num());
	}
}

bool UMotionReceiverSubsystem::GetLatestMotion(int32 PersonId, FPersonMotionData& OutData) const
{
	FScopeLock Lock(&PerPersonMutex);
	if (const FPersonMotionData* Data = PerPersonLatest.Find(PersonId))
	{
		OutData = *Data;
		return true;
	}
	return false;
}

void UMotionReceiverSubsystem::PushMotionFrame(FPersonMotionData&& NewData)
{
	FScopeLock Lock(&PerPersonMutex);
	FPersonMotionData& Slot = PerPersonLatest.FindOrAdd(NewData.PersonId);
	if (Slot.LastFrameId < NewData.LastFrameId)
	{
		Slot = MoveTemp(NewData);
	}
}

int32 UMotionReceiverSubsystem::GetPacketsReceived() const
{
	return Receiver.IsValid() ? Receiver->GetPacketsReceived() : 0;
}

int32 UMotionReceiverSubsystem::GetPacketsDropped() const
{
	return Receiver.IsValid() ? Receiver->GetPacketsDropped() : 0;
}
