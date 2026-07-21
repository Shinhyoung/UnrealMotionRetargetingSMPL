// Copyright — see /LICENSE for details.

#pragma once

#include "CoreMinimal.h"
#include "Subsystems/GameInstanceSubsystem.h"
#include "HAL/CriticalSection.h"
#include "HAL/Runnable.h"
#include "HAL/RunnableThread.h"
#include "Templates/Atomic.h"
#include "MotionReceiverSubsystem.generated.h"

class FSocket;

/**
 * One person's latest motion frame. Populated by the UDP listener thread,
 * consumed by the game thread via UMotionReceiverSubsystem::GetLatestMotion.
 *
 * All fields are already in UE coordinate space (Z-up, X-forward, left-handed)
 * because the python side (docs/protocol.md) does the swizzling before send.
 */
USTRUCT(BlueprintType)
struct MOTIONRETARGET_API FPersonMotionData
{
	GENERATED_BODY()

	/** Stable ID from the python tracker. */
	UPROPERTY(BlueprintReadOnly, Category = "MotionRetarget")
	int32 PersonId = -1;

	/** Frame id from the sender; used to reject stale packets. */
	UPROPERTY(BlueprintReadOnly, Category = "MotionRetarget")
	int32 LastFrameId = -1;

	/** Root translation in UE cm. */
	UPROPERTY(BlueprintReadOnly, Category = "MotionRetarget")
	FVector RootTranslation = FVector::ZeroVector;

	/**
	 * 24 SMPL joint local (parent-relative) rotations, xyzw quaternion order
	 * (matches ``FQuat`` component ordering). Index 0 is pelvis / global orient.
	 * Kept as ``FQuat`` end-to-end per CLAUDE.md §4.2 (no Rotator conversion).
	 */
	UPROPERTY(BlueprintReadOnly, Category = "MotionRetarget")
	TArray<FQuat> BoneRotations;
};


/**
 * FRunnable listener that owns the UDP socket. Blocks on RecvFrom in ``Run()``
 * and forwards parsed frames to the owning subsystem's per-person cache.
 * Failure to bind or a recv error just tears the thread down cleanly — the
 * subsystem stays alive so hot-restart of the listener is trivial.
 */
class FMotionUdpReceiver : public FRunnable
{
public:
	FMotionUdpReceiver(class UMotionReceiverSubsystem* InOwner, int32 InPort, const FString& InBindAddress);
	virtual ~FMotionUdpReceiver();

	//~ Begin FRunnable
	virtual bool Init() override;
	virtual uint32 Run() override;
	virtual void Stop() override;
	virtual void Exit() override;
	//~ End FRunnable

	int32 GetPacketsReceived() const { return PacketsReceived.Load(); }
	int32 GetPacketsDropped() const { return PacketsDropped.Load(); }

private:
	TWeakObjectPtr<UMotionReceiverSubsystem> Owner;
	int32 Port;
	FString BindAddress;
	FSocket* Socket = nullptr;

	TAtomic<bool> bStopRequested{ false };
	TAtomic<int32> PacketsReceived{ 0 };
	TAtomic<int32> PacketsDropped{ 0 };
};


/**
 * Owns the UDP listener and the per-person latest-motion cache.
 *
 * Design (CLAUDE.md §5, §7):
 *   - Listener runs on a worker thread — game thread never blocks on socket I/O.
 *   - Writes are atomic replacements per person under ``PerPersonMutex``.
 *   - Old ``LastFrameId`` values are dropped (UDP loss/reorder tolerance).
 */
UCLASS()
class MOTIONRETARGET_API UMotionReceiverSubsystem : public UGameInstanceSubsystem
{
	GENERATED_BODY()

public:
	virtual void Initialize(FSubsystemCollectionBase& Collection) override;
	virtual void Deinitialize() override;

	/**
	 * Non-blocking snapshot. Safe from any thread. Returns false when nothing
	 * has been received yet for that ``PersonId``.
	 */
	bool GetLatestMotion(int32 PersonId, FPersonMotionData& OutData) const;

	/**
	 * Called by the listener thread. Replaces the cache entry for this pid
	 * iff ``FrameId`` is strictly newer than the current entry.
	 */
	void PushMotionFrame(FPersonMotionData&& NewData);

	/** UDP port to listen on. Matches ``docs/protocol.md`` default. */
	UPROPERTY(EditAnywhere, BlueprintReadWrite, Category = "Network")
	int32 ListenPort = 9527;

	/** Bind address. Default any-interface. */
	UPROPERTY(EditAnywhere, BlueprintReadWrite, Category = "Network")
	FString BindAddress = TEXT("0.0.0.0");

	/** Total packets received since listener started (game-thread readable). */
	UFUNCTION(BlueprintPure, Category = "MotionRetarget")
	int32 GetPacketsReceived() const;

	/** Packets rejected as stale or malformed. */
	UFUNCTION(BlueprintPure, Category = "MotionRetarget")
	int32 GetPacketsDropped() const;

private:
	void DumpStatsToLog() const;

	mutable FCriticalSection PerPersonMutex;
	TMap<int32, FPersonMotionData> PerPersonLatest;

	TUniquePtr<FMotionUdpReceiver> Receiver;
	FRunnableThread* ReceiverThread = nullptr;

	struct IConsoleCommand* StatsCommand = nullptr;
};
