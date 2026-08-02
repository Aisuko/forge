//! The buffer pool's two invariants, held honest.
//!
//! `alloc` hands out recycled buffers, so a kernel's output buffer arrives
//! holding whatever the last op left in it — not zeros, which is what a freshly
//! created `wgpu::Buffer` gives you. Every kernel is written to fill its whole
//! output except one: `unsplit_heads` writes one third of a `[t, 3c]` gradient
//! and needs the rest zero.
//!
//! That is not a hypothetical. Before `alloc_zeroed` existed, the pool silently
//! corrupted `wte` gradients, and the only thing that caught it was a CPU/GPU
//! parity assertion two layers away. These tests catch it at the op.

use forge::{Device, Tensor, ops};

/// A device per test, deliberately.
///
/// The pool and the scope depth are per-context, so sharing one context across
/// libtest's parallel threads would let another test's allocations take the
/// dirtied buffer, and another test's open scope batch the arm this file calls
/// "unscoped" — both tests would still pass, while checking less. Concurrent
/// creation is what used to hang the run for minutes; `WgpuContext::new_async`
/// serializes it now, on every path, which is the fix and leaves each test its
/// own pool. `tests/device_concurrency.rs` holds that fix honest.
fn device() -> Option<Device> {
    Device::wgpu().ok()
}

/// Fill the free list with buffers holding a recognisable non-zero pattern, so
/// the next allocation of the same size gets one back dirty.
fn dirty_pool(device: &Device, numel: usize, marker: f32) {
    for _ in 0..4 {
        // Dropped at the end of each iteration, which is what returns it.
        let t = Tensor::from_f32(&vec![marker; numel], [numel], device).unwrap();
        // `add` allocates an output of exactly `numel` and writes all of it —
        // that output is the buffer the pool then hands to the next caller.
        let _ = ops::add(&t, &t).unwrap();
    }
}

// `unsplit_heads` is a backward kernel, so it only exists under `train` — but
// the invariant it guards belongs to the pool, which every build uses. The
// other tests here stay in the default suite for that reason.
#[cfg(feature = "train")]
#[test]
fn unsplit_head_zeroes_the_thirds_it_does_not_write() {
    let Some(device) = device() else {
        eprintln!("no WebGPU adapter; skipping");
        return;
    };
    let (h, t, hd) = (3usize, 4usize, 8usize);
    let c = h * hd;

    dirty_pool(&device, t * 3 * c, 7.5);

    let d = Tensor::from_f32(&vec![1.0f32; h * t * hd], [h, t, hd], &device).unwrap();
    for which in 0..3 {
        let out = ops::unsplit_head(&d, which).unwrap().to_vec_f32().unwrap();
        assert_eq!(out.len(), t * 3 * c);
        for row in 0..t {
            for third in 0..3 {
                for i in 0..c {
                    let v = out[row * 3 * c + third * c + i];
                    let expected = if third == which { 1.0 } else { 0.0 };
                    assert_eq!(
                        v, expected,
                        "which={which} row={row} third={third} col={i}: \
                         a recycled buffer leaked into the untouched thirds"
                    );
                }
            }
        }
    }
}

/// The general invariant behind `alloc`: an op's result must not depend on what
/// was in the buffer it was handed. Runs each op twice with the pool seeded
/// differently and requires bit-identical output.
#[test]
fn results_do_not_depend_on_recycled_contents() {
    let Some(device) = device() else {
        eprintln!("no WebGPU adapter; skipping");
        return;
    };
    let n = 512;
    let x: Vec<f32> = (0..n).map(|i| (i as f32 * 0.01).sin()).collect();

    let run = |marker: f32| -> Vec<f32> {
        dirty_pool(&device, n, marker);
        let t = Tensor::from_f32(&x, [n], &device).unwrap();
        let g = ops::gelu(&t).unwrap();
        let s = ops::softmax(&g.reshape([1, n]).unwrap(), false, 0).unwrap();
        ops::add(&s.reshape([n]).unwrap(), &g)
            .unwrap()
            .to_vec_f32()
            .unwrap()
    };

    let a = run(0.0);
    let b = run(-1234.5);
    assert_eq!(a, b, "op results changed with the pool's prior contents");
}

/// A dispatch scope must not change results, only the number of submits.
#[test]
fn scope_does_not_change_results() {
    let Some(device) = device() else {
        eprintln!("no WebGPU adapter; skipping");
        return;
    };
    let n = 256;
    let x: Vec<f32> = (0..n).map(|i| (i as f32 * 0.03).cos()).collect();
    let t = Tensor::from_f32(&x, [n], &device).unwrap();

    let unscoped = {
        let mut y = t.clone();
        for _ in 0..8 {
            y = ops::gelu(&ops::add(&y, &t).unwrap()).unwrap();
        }
        y.to_vec_f32().unwrap()
    };

    let scoped = {
        let _scope = device.dispatch_scope();
        let mut y = t.clone();
        for _ in 0..8 {
            y = ops::gelu(&ops::add(&y, &t).unwrap()).unwrap();
        }
        // Reading inside the scope must flush what is recorded, not read stale
        // bytes — the whole reason readbacks flush rather than trusting callers.
        y.to_vec_f32().unwrap()
    };

    assert_eq!(unscoped, scoped, "dispatch scope changed the result");
}

/// Reading a tensor mid-scope, continuing to compute, and reading again must
/// both be correct: `flush` takes the encoder, so the scope has to reopen one.
#[test]
fn scope_survives_a_readback_in_the_middle() {
    let Some(device) = device() else {
        eprintln!("no WebGPU adapter; skipping");
        return;
    };
    let t = Tensor::from_f32(&[1.0f32, 2.0, 3.0, 4.0], [4], &device).unwrap();
    let _scope = device.dispatch_scope();

    let a = ops::add(&t, &t).unwrap();
    let mid = a.to_vec_f32().unwrap();
    assert_eq!(mid, vec![2.0, 4.0, 6.0, 8.0]);

    let b = ops::add(&a, &t).unwrap();
    assert_eq!(b.to_vec_f32().unwrap(), vec![3.0, 6.0, 9.0, 12.0]);
}
