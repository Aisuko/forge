//! Creating and destroying devices from several threads at once must not hang.
//!
//! Concurrent `request_adapter` + `request_device` wedges inside the driver —
//! parallel `Device::wgpu()` calls, doing no compute at all, are what hung a
//! full `cargo test` roughly one run in four before creation was serialized.
//! Teardown is the same lock from the other side, and it hung the surprisal
//! suite the same way after creation alone was fixed. These tests are both
//! reproducers, kept where a regression will trip over them.
//!
//! Each one joins through `recv_timeout`, so a regression *fails* after two
//! minutes rather than hanging the suite with no message. The wedged threads
//! are left detached on timeout: they are parked in the driver and are not
//! coming back.

use std::sync::mpsc;
use std::time::Duration;

use forge::{Device, Tensor, ops};

/// Generous: this is a deadlock detector, not a performance budget. Cold
/// device creation on a loaded machine is seconds, not minutes.
const PATIENCE: Duration = Duration::from_secs(120);

/// Threads racing, and rounds of create-and-drop each. Rounds matter: a single
/// round of eight does not lose the coin flip often enough to catch anything.
const THREADS: usize = 8;
const ROUNDS: usize = 4;

/// Race `f` on [`THREADS`] threads for [`ROUNDS`] rounds, returning one flag
/// per creation: did it produce a device. Panics rather than waits forever.
///
/// Each device is dropped inside its thread, before the next round — the
/// reproducer is repeated creation, not thirty-two live devices.
fn race<F>(what: &str, f: F) -> Vec<bool>
where
    F: Fn(usize) -> Option<Device> + Send + Sync + Copy + 'static,
{
    let (tx, rx) = mpsc::channel();
    let threads: Vec<_> = (0..THREADS)
        .map(|i| {
            let tx = tx.clone();
            std::thread::spawn(move || {
                for _ in 0..ROUNDS {
                    // The device drops here, before the send.
                    if tx.send(f(i).is_some()).is_err() {
                        return; // the test already gave up
                    }
                }
            })
        })
        .collect();
    drop(tx);

    let total = THREADS * ROUNDS;
    let made: Vec<bool> = (0..total)
        .map(|done| {
            rx.recv_timeout(PATIENCE).unwrap_or_else(|_| {
                panic!(
                    "{what}: {done}/{total} device creations finished in {PATIENCE:?} — \
                     concurrent creation is wedged again"
                )
            })
        })
        .collect();

    // Joined, not detached: a thread still unwinding its wgpu state when the
    // process exits segfaults in the driver, and the last result arrives
    // before the last thread is gone. On the panic above they stay detached —
    // they are parked in the driver and are not coming back.
    for t in threads {
        t.join().expect("a creating thread panicked");
    }
    made
}

/// All of them succeeded, or none did (no adapter here — CPU-only CI).
fn assert_unanimous(made: &[bool], what: &str) {
    let ok = made.iter().filter(|m| **m).count();
    if ok == 0 {
        eprintln!("no WebGPU adapter; {what} proved only that nothing hung");
        return;
    }
    assert_eq!(
        ok,
        made.len(),
        "{what}: {ok} of {} creations succeeded — serializing creation must not \
         make it fail",
        made.len()
    );
}

/// The reproducer on the async path — the one a lock in the sync facade alone
/// leaves open.
#[test]
fn async_devices_created_in_parallel() {
    let made = race("async", |_| pollster::block_on(Device::wgpu_async()).ok());
    assert_unanimous(&made, "async");
}

/// Sync and async creators racing *each other*, which two separate locks would
/// not have covered.
#[test]
fn sync_and_async_creators_interleaved() {
    let made = race("mixed", |i| {
        if i % 2 == 0 {
            Device::wgpu().ok()
        } else {
            pollster::block_on(Device::wgpu_async()).ok()
        }
    });
    assert_unanimous(&made, "mixed");
}

/// Creation racing *teardown*, which the two tests above cannot reach: they
/// create and drop empty contexts, and an empty context has nothing to tear
/// down but the device itself.
///
/// Destroying a context that has done real work is a different animal —
/// compiled pipelines, pooled buffers, a submitted encoder, and only then
/// `vkDestroyDevice`. That teardown holds a driver lock the creating thread
/// needs, and until `WgpuContext`'s `Drop` took the same gate as creation, a
/// thread doing it while a neighbour created a device hung the process outright.
/// The compute here is what staggers the threads so the two phases overlap; the
/// numbers it produces do not matter, `op_parity` owns those.
#[test]
fn devices_are_created_used_and_dropped_in_parallel() {
    let made = race("busy", |_| {
        let device = Device::wgpu().ok()?;
        let n = 4096;
        let x: Vec<f32> = (0..n).map(|i| (i as f32 * 0.01).sin()).collect();
        let t = Tensor::from_f32(&x, [n], &device).unwrap();
        // A scope, so this is one submit of several kernels — the encoder and
        // the pipelines it used are live state the drop below has to unwind.
        let y = {
            let _scope = device.dispatch_scope();
            let mut y = t.clone();
            for _ in 0..4 {
                y = ops::gelu(&ops::add(&y, &t).unwrap()).unwrap();
            }
            y.to_vec_f32().unwrap()
        };
        assert_eq!(y.len(), n, "the dispatch did not produce a full result");
        Some(device)
    });
    assert_unanimous(&made, "busy");
}

/// The gate must not cost `wgpu_async` its `Send`, or nobody can `tokio::spawn`
/// it. Compile-time; the future here is never polled, so it touches no adapter.
#[test]
fn the_async_creation_future_is_send() {
    fn assert_send<T: Send>(_: &T) {}
    assert_send(&Device::wgpu_async());
}
