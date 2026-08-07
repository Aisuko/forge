//! A minimal async mutex, holding one thing: the right to create a device.
//!
//! Concurrent `request_adapter` + `request_device` wedges inside the driver
//! (see the 0.4.0 changelog), so creation has to be serialized. The obvious
//! `std::sync::Mutex` cannot do it here: its guard is not `Send`, and a future
//! holding one across an `.await` is `!Send`, which is exactly what
//! `tokio::spawn` refuses. So the wait list goes under a `std::sync::Mutex`
//! that is never held across an await, and the guard handed to the caller is
//! nothing but a reference.
//!
//! Deliberately not `tokio::sync::Mutex`: that is a runtime dependency, on the
//! wasm build too, to hold a bool.

use std::future::Future;
use std::pin::Pin;
use std::sync::Mutex;
use std::task::{Context, Poll, Waker};

/// A gate one holder passes through at a time.
pub struct Gate {
    state: Mutex<State>,
}

struct State {
    held: bool,
    waiters: Vec<Waker>,
}

impl Gate {
    pub const fn new() -> Self {
        Gate {
            state: Mutex::new(State {
                held: false,
                waiters: Vec::new(),
            }),
        }
    }

    /// Wait for the gate. The returned future is `Send`, and so is its output.
    pub fn lock(&self) -> Lock<'_> {
        Lock { gate: self }
    }

    /// Wait for the gate, blocking the calling thread.
    ///
    /// For `Drop`, which cannot await: destroying a device has to be serialized
    /// against creating one (see `WgpuContext`'s `Drop`), and a destructor has
    /// no executor to yield to. Parking here is safe because the inner mutex is
    /// never held across an `.await` — the holder is always making progress in
    /// the driver, not waiting on this thread.
    ///
    /// Native only: the browser has one thread, which must not block, and no
    /// second thread to race with.
    #[cfg(not(target_arch = "wasm32"))]
    pub fn lock_blocking(&self) -> Guard<'_> {
        pollster::block_on(self.lock())
    }

    /// The inner mutex guards no user data, so a panicking holder must not
    /// poison every later creator.
    fn state(&self) -> std::sync::MutexGuard<'_, State> {
        self.state.lock().unwrap_or_else(|e| e.into_inner())
    }
}

/// The future returned by [`Gate::lock`].
pub struct Lock<'a> {
    gate: &'a Gate,
}

impl<'a> Future for Lock<'a> {
    type Output = Guard<'a>;

    fn poll(self: Pin<&mut Self>, cx: &mut Context<'_>) -> Poll<Self::Output> {
        let mut s = self.gate.state();
        if !s.held {
            s.held = true;
            return Poll::Ready(Guard { gate: self.gate });
        }
        // One entry per waiting future, however many times it is polled.
        if !s.waiters.iter().any(|w| w.will_wake(cx.waker())) {
            s.waiters.push(cx.waker().clone());
        }
        Poll::Pending
    }
}

/// Holds the gate until dropped.
pub struct Guard<'a> {
    gate: &'a Gate,
}

impl Drop for Guard<'_> {
    fn drop(&mut self) {
        let waiters = {
            let mut s = self.gate.state();
            s.held = false;
            std::mem::take(&mut s.waiters)
        };
        // Every waiter, not the first one. Ownership is never handed to a
        // sleeping future — each one re-polls and races for the gate — so a
        // caller that drops its `lock()` future mid-wait cannot strand the
        // gate closed, which waking exactly one would allow. The herd is the
        // number of threads creating a device at once: single digits, once.
        for w in waiters {
            w.wake();
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::sync::Arc;
    use std::sync::atomic::{AtomicUsize, Ordering};

    /// The invariant, without a GPU: two tasks contending, and the second one
    /// only ever runs the critical section after the first has left it.
    #[test]
    fn one_holder_at_a_time() {
        static GATE: Gate = Gate::new();
        let inside = Arc::new(AtomicUsize::new(0));
        let peak = Arc::new(AtomicUsize::new(0));

        let threads: Vec<_> = (0..8)
            .map(|_| {
                let (inside, peak) = (inside.clone(), peak.clone());
                std::thread::spawn(move || {
                    pollster::block_on(async {
                        let _g = GATE.lock().await;
                        let n = inside.fetch_add(1, Ordering::SeqCst) + 1;
                        peak.fetch_max(n, Ordering::SeqCst);
                        std::thread::sleep(std::time::Duration::from_millis(5));
                        inside.fetch_sub(1, Ordering::SeqCst);
                    })
                })
            })
            .collect();
        for t in threads {
            t.join().unwrap();
        }
        assert_eq!(peak.load(Ordering::SeqCst), 1, "two holders at once");
    }

    /// A waiter that gives up must not take the gate with it.
    #[test]
    fn a_dropped_waiter_does_not_strand_the_gate() {
        static GATE: Gate = Gate::new();
        let held = pollster::block_on(GATE.lock());

        let mut waiting = Box::pin(GATE.lock());
        let waker = std::task::Waker::noop();
        let mut cx = Context::from_waker(waker);
        assert!(waiting.as_mut().poll(&mut cx).is_pending());
        drop(waiting);

        drop(held);
        assert!(
            Box::pin(GATE.lock()).as_mut().poll(&mut cx).is_ready(),
            "the gate stayed closed after its waiter went away"
        );
    }
}
