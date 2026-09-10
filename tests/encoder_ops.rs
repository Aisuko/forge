use forge::ops;
use forge::{Device, Tensor};
use rand::rngs::StdRng;
use rand::{Rng, SeedableRng};

const TOL: f32 = 1e-4;

fn rand_vec(rng: &mut StdRng, n: usize) -> Vec<f32> {
    (0..n).map(|_| rng.random_range(-2.0..2.0)).collect()
}

fn assert_close(a: &[f32], b: &[f32], tol: f32, what: &str) {
    assert_eq!(a.len(), b.len(), "{what}: length mismatch");
    for (i, (x, y)) in a.iter().zip(b).enumerate() {
        assert!(
            (x - y).abs() <= tol,
            "{what}: element {i} differs: {x} vs {y}"
        );
    }
}

fn wgpu_device() -> Device {
    static DEV: std::sync::OnceLock<Device> = std::sync::OnceLock::new();
    DEV.get_or_init(|| Device::wgpu().expect("wgpu device"))
        .clone()
}

#[test]
fn softmax_masked_zeroes_padding_and_normalizes() {
    let (batch, heads, t) = (2usize, 2usize, 4usize);
    let rows = batch * heads * t;
    let mut rng = StdRng::seed_from_u64(7);
    let x = rand_vec(&mut rng, rows * t);
    let key_len = [3u32, 1];
    let y = ops::softmax_masked(
        &Tensor::from_f32(&x, [rows, t], &Device::Cpu).unwrap(),
        &Tensor::from_u32(&key_len, [batch], &Device::Cpu).unwrap(),
        batch,
    )
    .unwrap()
    .to_vec_f32()
    .unwrap();
    for r in 0..rows {
        let b = r / (heads * t);
        let n = key_len[b] as usize;
        let row = &y[r * t..(r + 1) * t];
        assert!(
            (row[..n].iter().sum::<f32>() - 1.0).abs() < 1e-5,
            "row {r} does not sum to 1 over {n} visible keys"
        );
        for (j, &v) in row.iter().enumerate().skip(n) {
            assert_eq!(v, 0.0, "row {r} key {j} should be masked");
        }
    }
}

#[test]
fn mean_pool_averages_only_valid_rows() {
    let (batch, seq, c) = (2usize, 4usize, 3usize);
    let x: Vec<f32> = (0..batch * seq * c).map(|i| i as f32).collect();
    let key_len = [2u32, 4];
    let y = ops::mean_pool(
        &Tensor::from_f32(&x, [batch * seq, c], &Device::Cpu).unwrap(),
        &Tensor::from_u32(&key_len, [batch], &Device::Cpu).unwrap(),
        batch,
    )
    .unwrap()
    .to_vec_f32()
    .unwrap();
    assert_eq!(&y[..3], &[1.5, 2.5, 3.5]);
    assert_eq!(&y[3..], &[16.5, 17.5, 18.5]);
}

#[test]
fn l2_norm_produces_unit_rows() {
    let mut rng = StdRng::seed_from_u64(11);
    let (rows, c) = (5usize, 17usize);
    let x = rand_vec(&mut rng, rows * c);
    let y = ops::l2_norm(
        &Tensor::from_f32(&x, [rows, c], &Device::Cpu).unwrap(),
        1e-12,
    )
    .unwrap()
    .to_vec_f32()
    .unwrap();
    for r in 0..rows {
        let n: f32 = y[r * c..(r + 1) * c].iter().map(|v| v * v).sum::<f32>();
        assert!((n - 1.0).abs() < 1e-5, "row {r} norm {n}");
    }
}

#[test]
fn softmax_masked_parity() {
    let gpu = wgpu_device();
    let mut rng = StdRng::seed_from_u64(21);
    for (batch, group, cols) in [(1usize, 1usize, 8usize), (4, 6, 33), (3, 2, 300)] {
        let rows = batch * group;
        let x = rand_vec(&mut rng, rows * cols);
        let key_len: Vec<u32> = (0..batch)
            .map(|_| rng.random_range(1..=cols) as u32)
            .collect();
        let run = |d: &Device| {
            ops::softmax_masked(
                &Tensor::from_f32(&x, [rows, cols], d).unwrap(),
                &Tensor::from_u32(&key_len, [batch], d).unwrap(),
                batch,
            )
            .unwrap()
            .to_vec_f32()
            .unwrap()
        };
        assert_close(
            &run(&Device::Cpu),
            &run(&gpu),
            TOL,
            &format!("softmax_masked b={batch} g={group} c={cols}"),
        );
    }
}

#[test]
fn mean_pool_parity() {
    let gpu = wgpu_device();
    let mut rng = StdRng::seed_from_u64(22);
    for (batch, seq, c) in [(1usize, 1usize, 4usize), (8, 16, 64), (3, 257, 33)] {
        let x = rand_vec(&mut rng, batch * seq * c);
        let key_len: Vec<u32> = (0..batch)
            .map(|_| rng.random_range(1..=seq) as u32)
            .collect();
        let fwd = |d: &Device| {
            ops::mean_pool(
                &Tensor::from_f32(&x, [batch * seq, c], d).unwrap(),
                &Tensor::from_u32(&key_len, [batch], d).unwrap(),
                batch,
            )
            .unwrap()
        };
        assert_close(
            &fwd(&Device::Cpu).to_vec_f32().unwrap(),
            &fwd(&gpu).to_vec_f32().unwrap(),
            TOL,
            &format!("mean_pool b={batch} t={seq} c={c}"),
        );

        let dy = rand_vec(&mut rng, batch * c);
        let bwd = |d: &Device| {
            ops::mean_pool_bwd(
                &Tensor::from_f32(&dy, [batch, c], d).unwrap(),
                &Tensor::from_u32(&key_len, [batch], d).unwrap(),
                batch,
                seq,
            )
            .unwrap()
            .to_vec_f32()
            .unwrap()
        };
        assert_close(
            &bwd(&Device::Cpu),
            &bwd(&gpu),
            TOL,
            &format!("mean_pool_bwd b={batch} t={seq} c={c}"),
        );
    }
}

#[test]
fn l2_norm_parity() {
    let gpu = wgpu_device();
    let mut rng = StdRng::seed_from_u64(23);
    for (rows, c) in [(1usize, 3usize), (64, 320), (7, 513)] {
        let x = rand_vec(&mut rng, rows * c);
        let dy = rand_vec(&mut rng, rows * c);
        let fwd = |d: &Device| {
            ops::l2_norm(&Tensor::from_f32(&x, [rows, c], d).unwrap(), 1e-12)
                .unwrap()
                .to_vec_f32()
                .unwrap()
        };
        assert_close(
            &fwd(&Device::Cpu),
            &fwd(&gpu),
            TOL,
            &format!("l2_norm {rows}x{c}"),
        );
        let bwd = |d: &Device| {
            ops::l2_norm_bwd(
                &Tensor::from_f32(&x, [rows, c], d).unwrap(),
                &Tensor::from_f32(&dy, [rows, c], d).unwrap(),
                1e-12,
            )
            .unwrap()
            .to_vec_f32()
            .unwrap()
        };
        assert_close(
            &bwd(&Device::Cpu),
            &bwd(&gpu),
            TOL,
            &format!("l2_norm_bwd {rows}x{c}"),
        );
    }
}

fn gradcheck(
    x: &[f32],
    w: &[f32],
    forward: impl Fn(&[f32]) -> Vec<f32>,
    analytic: impl Fn(&[f32], &[f32]) -> Vec<f32>,
) {
    let dx = analytic(x, w);
    let h = 1e-3f32;
    for i in 0..x.len() {
        let mut plus = x.to_vec();
        let mut minus = x.to_vec();
        plus[i] += h;
        minus[i] -= h;
        let fp: f32 = forward(&plus).iter().zip(w).map(|(a, b)| a * b).sum();
        let fm: f32 = forward(&minus).iter().zip(w).map(|(a, b)| a * b).sum();
        let num = (fp - fm) / (2.0 * h);
        assert!(
            (num - dx[i]).abs() <= 2e-2 * (1.0 + num.abs()),
            "coord {i}: analytic {} vs numeric {num}",
            dx[i]
        );
    }
}

#[test]
fn mean_pool_gradcheck() {
    let (batch, seq, c) = (3usize, 5usize, 4usize);
    let mut rng = StdRng::seed_from_u64(31);
    let x = rand_vec(&mut rng, batch * seq * c);
    let w = rand_vec(&mut rng, batch * c);
    let key_len = [2u32, 5, 1];
    let klen = |d: &Device| Tensor::from_u32(&key_len, [batch], d).unwrap();
    gradcheck(
        &x,
        &w,
        |xv| {
            ops::mean_pool(
                &Tensor::from_f32(xv, [batch * seq, c], &Device::Cpu).unwrap(),
                &klen(&Device::Cpu),
                batch,
            )
            .unwrap()
            .to_vec_f32()
            .unwrap()
        },
        |_, wv| {
            ops::mean_pool_bwd(
                &Tensor::from_f32(wv, [batch, c], &Device::Cpu).unwrap(),
                &klen(&Device::Cpu),
                batch,
                seq,
            )
            .unwrap()
            .to_vec_f32()
            .unwrap()
        },
    );
}

#[test]
fn l2_norm_gradcheck() {
    let (rows, c) = (4usize, 6usize);
    let mut rng = StdRng::seed_from_u64(32);
    let x = rand_vec(&mut rng, rows * c);
    let w = rand_vec(&mut rng, rows * c);
    gradcheck(
        &x,
        &w,
        |xv| {
            ops::l2_norm(
                &Tensor::from_f32(xv, [rows, c], &Device::Cpu).unwrap(),
                1e-12,
            )
            .unwrap()
            .to_vec_f32()
            .unwrap()
        },
        |xv, wv| {
            ops::l2_norm_bwd(
                &Tensor::from_f32(xv, [rows, c], &Device::Cpu).unwrap(),
                &Tensor::from_f32(wv, [rows, c], &Device::Cpu).unwrap(),
                1e-12,
            )
            .unwrap()
            .to_vec_f32()
            .unwrap()
        },
    );
}

#[test]
fn softmax_masked_gradcheck() {
    let (batch, group, cols) = (2usize, 3usize, 5usize);
    let rows = batch * group;
    let mut rng = StdRng::seed_from_u64(33);
    let x = rand_vec(&mut rng, rows * cols);
    let w = rand_vec(&mut rng, rows * cols);
    let key_len = [4u32, 2];
    let klen = Tensor::from_u32(&key_len, [batch], &Device::Cpu).unwrap();
    let fwd = |xv: &[f32]| {
        ops::softmax_masked(
            &Tensor::from_f32(xv, [rows, cols], &Device::Cpu).unwrap(),
            &klen,
            batch,
        )
        .unwrap()
    };
    gradcheck(
        &x,
        &w,
        |xv| fwd(xv).to_vec_f32().unwrap(),
        |xv, wv| {
            ops::softmax_bwd(
                &fwd(xv),
                &Tensor::from_f32(wv, [rows, cols], &Device::Cpu).unwrap(),
            )
            .unwrap()
            .to_vec_f32()
            .unwrap()
        },
    );
}
