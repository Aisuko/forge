//! Stage 8 gates (roadmap v4): analytic gradients match central-difference
//! numerical gradients on a small random-init model (CPU reference), and
//! CPU vs WGPU gradients agree per parameter.

use forge::{Device, Gpt2, Gpt2Config};

fn tiny_config() -> Gpt2Config {
    Gpt2Config {
        n_layer: 2,
        n_head: 2,
        n_embd: 16,
        n_ctx: 16,
        vocab_size: 31,
        layer_norm_epsilon: 1e-5,
        eos_token_id: None,
    }
}

const INPUT: [u32; 8] = [1, 7, 30, 4, 4, 19, 2, 11];
const TARGET: [u32; 8] = [7, 30, 4, 4, 19, 2, 11, 5];

#[test]
fn gradcheck_tiny_model_cpu() {
    let mut model = Gpt2::init_random(tiny_config(), &Device::Cpu, 42).unwrap();
    let (loss0, grads) = model.loss_grads(&INPUT, &TARGET, 0.0, 0).unwrap();
    assert!(
        (loss0 - (31f32).ln()).abs() < 1.5,
        "random-init loss {loss0} far from ln(vocab)"
    );

    let specs = model.param_specs();
    let h = 1e-2f32;
    let mut checked = 0usize;
    for pi in 0..specs.len() {
        let g = grads[pi].to_vec_f32().unwrap();
        // Check the two largest-magnitude entries of each parameter's grad
        // (best signal-to-noise for f32 central differences).
        let mut order: Vec<usize> = (0..g.len()).collect();
        order.sort_by(|&a, &b| g[b].abs().total_cmp(&g[a].abs()));
        for &ei in order.iter().take(2) {
            let analytic = g[ei];
            if analytic.abs() < 5e-3 {
                continue; // below numeric noise floor at f32
            }
            let eval = |model: &mut Gpt2, delta: f32| -> f32 {
                let shape = model.params().unwrap()[pi].shape().clone();
                let mut vals = model.params().unwrap()[pi].to_vec_f32().unwrap();
                vals[ei] += delta;
                let nt = forge::Tensor::from_f32(&vals, shape, &Device::Cpu).unwrap();
                *model.params_mut().unwrap()[pi] = nt;
                let (l, _) = model.loss_grads(&INPUT, &TARGET, 0.0, 0).unwrap();
                l
            };
            let lp = eval(&mut model, h);
            let lm = eval(&mut model, -2.0 * h); // from +h to -h
            eval(&mut model, h); // restore
            let numeric = (lp - lm) / (2.0 * h);
            let denom = 1.0f32.max(numeric.abs()).max(analytic.abs());
            let rel = (numeric - analytic).abs() / denom;
            assert!(
                rel <= 1e-2,
                "gradcheck failed for {} [{ei}]: analytic {analytic:.5} vs numeric {numeric:.5} (rel {rel:.3})",
                specs[pi].0
            );
            checked += 1;
        }
    }
    assert!(checked >= 40, "too few gradcheck samples ran: {checked}");
    println!("gradcheck passed on {checked} sampled entries, loss {loss0:.4}");
}

#[test]
fn grads_parity_cpu_wgpu() {
    let gpu = Device::wgpu().expect("wgpu adapter required");
    let cpu_model = Gpt2::init_random(tiny_config(), &Device::Cpu, 7).unwrap();
    let gpu_model = Gpt2::init_random(tiny_config(), &gpu, 7).unwrap();

    let (cl, cg) = cpu_model.loss_grads(&INPUT, &TARGET, 0.0, 0).unwrap();
    let (gl, gg) = gpu_model.loss_grads(&INPUT, &TARGET, 0.0, 0).unwrap();
    assert!(
        (cl - gl).abs() <= 1e-4,
        "loss diverged: cpu {cl} vs wgpu {gl}"
    );

    let specs = cpu_model.param_specs();
    for ((name, _), (c, g)) in specs.iter().zip(cg.iter().zip(&gg)) {
        let cv = c.to_vec_f32().unwrap();
        let gv = g.to_vec_f32().unwrap();
        let mut max = 0.0f32;
        for (a, b) in cv.iter().zip(&gv) {
            max = max.max((a - b).abs());
        }
        assert!(max <= 1e-3, "grad parity {name}: max abs diff {max:.2e}");
    }

    // With dropout active, deterministic masks keep the backends identical.
    let (cl2, _) = cpu_model.loss_grads(&INPUT, &TARGET, 0.2, 99).unwrap();
    let (gl2, _) = gpu_model.loss_grads(&INPUT, &TARGET, 0.2, 99).unwrap();
    assert!(
        (cl2 - gl2).abs() <= 1e-4,
        "dropout loss diverged: cpu {cl2} vs wgpu {gl2}"
    );
    println!("grad parity OK; loss {cl:.4}, dropout loss {cl2:.4}");
}
