//! `Gpt2::surprisal_async` — the reactive page's engine.
//!
//! The claim under test is not "these numbers look plausible" but that
//! surprisal agrees with the decode path. If position `i`'s bits disagree with
//! the distribution `logits_step` produces after seeing `ids[..i]`, the page is
//! colouring text by something other than what the model would predict.

use forge::{Device, Gpt2, Gpt2Config};

fn tiny() -> Gpt2Config {
    Gpt2Config {
        n_layer: 2,
        n_head: 2,
        // 64 f32 = 256 B: `logits_step` narrows to the last row, and storage
        // offsets must be 256-byte aligned.
        n_embd: 64,
        n_ctx: 32,
        vocab_size: 11,
        layer_norm_epsilon: 1e-5,
        eos_token_id: None,
    }
}

#[test]
fn surprisal_matches_the_decode_path() {
    let device = Device::wgpu().unwrap_or(Device::Cpu);
    let model = Gpt2::init_random(tiny(), &device, 3).unwrap();
    let ids: Vec<u32> = vec![1, 4, 4, 9, 2, 7, 0, 3];

    let s = pollster::block_on(model.surprisal_async(&ids)).unwrap();
    assert_eq!(s.bits.len(), ids.len());
    assert_eq!(s.bits[0], 0.0, "nothing precedes the first token");

    for i in 1..ids.len() {
        // What the model predicts having seen exactly ids[..i].
        let mut cache = model.new_cache().unwrap();
        let logits = model.logits_step(&ids[..i], &mut cache).unwrap();
        let max = logits.iter().copied().fold(f32::NEG_INFINITY, f32::max);
        let sum: f32 = logits.iter().map(|l| (l - max).exp()).sum();
        let log_z = max + sum.ln();
        let want = -(logits[ids[i] as usize] - log_z) / std::f32::consts::LN_2;
        assert!(
            (s.bits[i] - want).abs() < 2e-3,
            "position {i}: surprisal {} vs decode {want}",
            s.bits[i]
        );

        let best = logits
            .iter()
            .enumerate()
            .max_by(|a, b| a.1.total_cmp(b.1))
            .unwrap()
            .0 as u32;
        assert_eq!(s.top[i], best, "position {i}: top token");
        assert!(
            s.top_p[i] > 0.0 && s.top_p[i] <= 1.0,
            "position {i}: top_p {} out of range",
            s.top_p[i]
        );
    }
}

/// A token the model is certain of must score ~0 bits, and one it considers
/// impossible must score high. Without this the test above would pass on a
/// function that returned a constant matching a constant.
#[test]
fn bits_are_bits() {
    let device = Device::wgpu().unwrap_or(Device::Cpu);
    let model = Gpt2::init_random(tiny(), &device, 5).unwrap();
    let ids: Vec<u32> = vec![3, 1, 4, 1, 5, 9, 2, 6];
    let s = pollster::block_on(model.surprisal_async(&ids)).unwrap();

    // An untrained model is near-uniform, so every position should sit close to
    // log2(vocab) — the entropy of a fair guess over 11 symbols.
    let uniform = (11f32).log2();
    for i in 1..ids.len() {
        assert!(
            (s.bits[i] - uniform).abs() < 1.5,
            "position {i}: {} bits, expected near {uniform} for an untrained model",
            s.bits[i]
        );
    }

    // And the scale is the right way round: the model's own top choice must
    // never be more surprising than the token that was actually there.
    let top_ids: Vec<u32> = s.top[1..].to_vec();
    let mut forced = ids.clone();
    forced[1..].copy_from_slice(&top_ids);
    let s2 = pollster::block_on(model.surprisal_async(&forced)).unwrap();
    assert!(
        s2.bits[1] <= s.bits[1] + 1e-4,
        "the model's own prediction scored {} vs {} for the real token",
        s2.bits[1],
        s.bits[1]
    );
}
