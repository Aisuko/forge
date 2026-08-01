//! The council's two load-bearing invariants.
//!
//! 1. The merge is the identity when there is one expert — i.e. running a model
//!    through `hidden_step` + `logits_from_hidden` is the same arithmetic as
//!    running it through `logits_step`. If that ever drifts, every vector the
//!    council page draws is a picture of something other than the model.
//! 2. The router is a proper distribution, and `beta = 0` really does weight
//!    every expert equally.

use forge::{Device, Gpt2, Gpt2Config, Sampling};
use forge_council::{Council, DEFAULT_BETA};

fn tiny_config() -> Gpt2Config {
    Gpt2Config {
        n_layer: 2,
        n_head: 2,
        n_embd: 16,
        n_ctx: 32,
        vocab_size: 31,
        layer_norm_epsilon: 1e-5,
        eos_token_id: None,
    }
}

const PROMPT: [u32; 6] = [1, 7, 30, 4, 4, 19];

/// Splitting the head off the body and putting it back must change nothing.
#[test]
fn hidden_then_head_equals_logits() {
    let device = Device::Cpu;
    let model = Gpt2::init_random(tiny_config(), &device, 7).unwrap();

    let mut cache = model.new_cache().unwrap();
    let direct = model.logits_step(&PROMPT, &mut cache).unwrap();

    let mut cache = model.new_cache().unwrap();
    let hidden = model.hidden_step(&PROMPT, &mut cache).unwrap();
    assert_eq!(hidden.len(), tiny_config().n_embd);
    let via_hidden = model.logits_from_hidden(&hidden).unwrap();

    assert_eq!(direct.len(), via_hidden.len());
    for (a, b) in direct.iter().zip(&via_hidden) {
        assert!(
            (a - b).abs() < 1e-4,
            "logits diverged after routing through the hidden state: {a} vs {b}"
        );
    }
}

/// A council of one is just that model — the merge adds nothing to remove.
#[test]
fn single_expert_council_matches_the_model() {
    let device = Device::Cpu;
    let config = tiny_config();
    let solo = Gpt2::init_random(config.clone(), &device, 7).unwrap();
    let mut cache = solo.new_cache().unwrap();
    let expected = solo.logits_step(&PROMPT, &mut cache).unwrap();
    let expected_top = argmax(&expected);

    let member = Gpt2::init_random(config, &device, 7).unwrap();
    let mut council = Council::new(vec![member], vec!["solo".into()], 42).unwrap();
    let step = council.step(&PROMPT, Sampling::Greedy, 5).unwrap();

    assert_eq!(step.experts.len(), 1);
    assert_eq!(
        step.experts[0].weight, 1.0,
        "one expert must carry all the weight"
    );
    assert_eq!(step.chosen, expected_top);
    assert_eq!(step.consensus, 1.0);
}

/// Differently shaped models have hidden states in different spaces; adding
/// them is meaningless and must be refused rather than silently averaged.
#[test]
fn mismatched_experts_are_rejected() {
    let device = Device::Cpu;
    let a = Gpt2::init_random(tiny_config(), &device, 1).unwrap();
    let mut wider = tiny_config();
    wider.n_embd = 32;
    let b = Gpt2::init_random(wider, &device, 2).unwrap();
    let err = Council::new(vec![a, b], vec!["a".into(), "b".into()], 42);
    assert!(err.is_err(), "a council must not accept mismatched experts");
}

/// The router is a distribution, and at beta 0 it is the uniform one.
#[test]
fn router_weights_are_a_distribution() {
    let device = Device::Cpu;
    let config = tiny_config();
    let models: Vec<Gpt2> = (0..3)
        .map(|s| Gpt2::init_random(config.clone(), &device, s).unwrap())
        .collect();
    let names: Vec<String> = (0..3).map(|i| format!("e{i}")).collect();

    let mut council = Council::new(models, names, 42).unwrap();
    assert_eq!(council.beta, DEFAULT_BETA);
    let sharp = council.step(&PROMPT, Sampling::Greedy, 5).unwrap();
    let total: f32 = sharp.experts.iter().map(|e| e.weight).sum();
    assert!((total - 1.0).abs() < 1e-5, "weights sum to {total}, not 1");
    assert!(sharp.experts.iter().all(|e| e.entropy > 0.0));

    council.reset(42).unwrap();
    council.beta = 0.0;
    let flat = council.step(&PROMPT, Sampling::Greedy, 5).unwrap();
    for e in &flat.experts {
        assert!(
            (e.weight - 1.0 / 3.0).abs() < 1e-5,
            "beta 0 must weight every expert equally, got {}",
            e.weight
        );
    }
}

fn argmax(v: &[f32]) -> u32 {
    let mut best = 0usize;
    for (i, x) in v.iter().enumerate() {
        if *x > v[best] {
            best = i;
        }
    }
    best as u32
}
