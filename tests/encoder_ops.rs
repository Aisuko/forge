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
