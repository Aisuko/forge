//! `forge_surprise::surprisal` — the reactive page's engine.
//!
//! The claim under test is not "these numbers look plausible" but that
//! surprisal agrees with the decode path. If position `i`'s bits disagree with
//! the distribution `logits_step` produces after seeing `ids[..i]`, the page is
//! colouring text by something other than what the model would predict.

use forge::{Device, Gpt2, Gpt2Config};
use forge_surprise::surprisal;

/// What the page asks for: enough alternatives to make a spin look considered,
/// and small enough that `top_probs` partitions rather than sorts.
const K: usize = 8;

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

    let s = pollster::block_on(surprisal(&model, &ids, K)).unwrap();
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
        assert_eq!(s.top(i), best, "position {i}: top token");
        assert!(
            s.top_p(i) > 0.0 && s.top_p(i) <= 1.0,
            "position {i}: top_p {} out of range",
            s.top_p(i)
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
    let s = pollster::block_on(surprisal(&model, &ids, K)).unwrap();

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
    let top_ids: Vec<u32> = (1..ids.len()).map(|i| s.top(i)).collect();
    let mut forced = ids.clone();
    forced[1..].copy_from_slice(&top_ids);
    let s2 = pollster::block_on(surprisal(&model, &forced, K)).unwrap();
    assert!(
        s2.bits[1] <= s.bits[1] + 1e-4,
        "the model's own prediction scored {} vs {} for the real token",
        s2.bits[1],
        s.bits[1]
    );
}

/// Every row is a genuine ranking. The page samples the flicker in proportion
/// to `alt_p` and reads column 0 as "what it expected", so a row that is not
/// sorted would put the model's second choice in the panel's top bar.
#[test]
fn candidates_are_ranked_and_sum_to_at_most_one() {
    let device = Device::wgpu().unwrap_or(Device::Cpu);
    let model = Gpt2::init_random(tiny(), &device, 7).unwrap();
    let ids: Vec<u32> = vec![2, 8, 5, 5, 1, 10, 0, 6];
    let s = pollster::block_on(surprisal(&model, &ids, K)).unwrap();

    assert_eq!(s.k, K);
    assert_eq!(s.alt_ids.len(), ids.len() * K);
    assert_eq!(s.alt_p.len(), ids.len() * K);

    for i in 1..ids.len() {
        let row = &s.alt_p[i * K..(i + 1) * K];
        for j in 1..K {
            assert!(
                row[j] <= row[j - 1] + 1e-6,
                "position {i}: p rose from {} to {} at column {j}",
                row[j - 1],
                row[j]
            );
        }
        // A partial ranking of a full softmax: k of the vocabulary's mass, so
        // never more than all of it and — for k = 8 of 11 — never much less.
        let mass: f32 = row.iter().sum();
        assert!(
            mass > 0.0 && mass <= 1.0 + 1e-4,
            "position {i}: top-{K} mass {mass}"
        );

        // No duplicates: a repeated id would make the flicker cycle one
        // character twice and understate how torn the model was.
        let mut ids_row = s.alt_ids[i * K..(i + 1) * K].to_vec();
        ids_row.sort_unstable();
        let before = ids_row.len();
        ids_row.dedup();
        assert_eq!(ids_row.len(), before, "position {i}: repeated candidate");
    }
}

/// `k = 1` is the shape this function had before it learned about
/// alternatives, and it still has to produce the same numbers: column 0 is the
/// old `top`/`top_p`, and `bits` never depended on `k` at all.
#[test]
fn k_of_one_reproduces_the_winner() {
    let device = Device::wgpu().unwrap_or(Device::Cpu);
    let model = Gpt2::init_random(tiny(), &device, 11).unwrap();
    let ids: Vec<u32> = vec![4, 0, 7, 3, 3, 9, 1, 2];

    let one = pollster::block_on(surprisal(&model, &ids, 1)).unwrap();
    let many = pollster::block_on(surprisal(&model, &ids, K)).unwrap();

    assert_eq!(one.k, 1);
    assert_eq!(one.alt_ids.len(), ids.len());
    for i in 0..ids.len() {
        assert_eq!(one.bits[i], many.bits[i], "position {i}: bits moved with k");
        assert_eq!(one.top(i), many.top(i), "position {i}: winner moved with k");
        assert!(
            (one.top_p(i) - many.top_p(i)).abs() < 1e-6,
            "position {i}: top_p moved with k"
        );
    }
}

/// `k` above the vocabulary is clamped rather than rejected: the page asks for
/// 8 without knowing how many characters the model knows, and a 4-symbol
/// vocabulary is a legitimate model, not a caller error.
#[test]
fn k_is_clamped_to_the_vocabulary() {
    let device = Device::wgpu().unwrap_or(Device::Cpu);
    let model = Gpt2::init_random(tiny(), &device, 13).unwrap();
    let ids: Vec<u32> = vec![1, 2, 3, 4];
    let s = pollster::block_on(surprisal(&model, &ids, 4096)).unwrap();
    assert_eq!(s.k, 11, "clamped to vocab_size");
    assert_eq!(s.alt_ids.len(), ids.len() * 11);

    // And k = 0 is 1, not an empty row nothing downstream could index.
    let z = pollster::block_on(surprisal(&model, &ids, 0)).unwrap();
    assert_eq!(z.k, 1);
    assert_eq!(z.top(1), s.top(1));
}
