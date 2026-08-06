//! `.fzm` — a flat, hand-rolled q4 quantized checkpoint format. No external
//! decoder (no GGUF/candle-core): forge implements its own dequant, matching
//! the rest of the runtime's "no bindings" rule.
//!
//! Layout, little-endian throughout:
//!
//! ```text
//! magic:        b"FZM1"
//! tensor_count: u32
//! per tensor header:
//!   name_len: u16, name: [u8; name_len] (utf8)
//!   ndim: u8, shape: [u32; ndim]
//!   group_size: u32, n_groups: u32
//! per tensor body:
//!   [scale: f32, min: f32] * n_groups
//!   packed nibbles: ceil(numel/2) bytes, 2 values/byte (low nibble first)
//! ```
//!
//! Quantization is per-group affine: `scale = (max-min)/15`,
//! `q = round((x-min)/scale)` clamped to `0..=15`; dequant is
//! `x = q as f32 * scale + min`. Groups are `group_size` consecutive flat
//! elements in row-major order; the last group in a tensor may be short if
//! `numel % group_size != 0`.
//!
//! This is read/write at the host (f32) boundary only — same place
//! `save_safetensors`/`host()` in [`crate::models::gpt2::Gpt2::from_safetensors_bytes`]
//! operate. No WGSL or GPU-side packed matmul yet: values are dequantized to
//! f32 on load, then uploaded exactly like a safetensors checkpoint.

use std::path::Path;

use crate::error::{ForgeError, Result};

const MAGIC: &[u8; 4] = b"FZM1";

/// Flat elements per quantization group. 64 balances the 8-bytes/group
/// scale+min overhead against accuracy: finer groups cost more header bytes,
/// coarser ones average over noisier value ranges.
pub const GROUP_SIZE: usize = 64;

/// `(name, shape, host f32 data)` — same shape `save_safetensors` takes and
/// the safetensors `host()` closure produces.
pub type TensorEntries = Vec<(String, Vec<usize>, Vec<f32>)>;

/// Save named f32 tensors as `.fzm` q4. Takes the same
/// `(name, shape, host data)` entries `save_safetensors` does.
pub fn save_fzm_q4(
    path: impl AsRef<Path>,
    entries: &[(String, Vec<usize>, Vec<f32>)],
) -> Result<()> {
    let mut out = Vec::new();
    out.extend_from_slice(MAGIC);
    out.extend_from_slice(&(entries.len() as u32).to_le_bytes());

    for (name, shape, data) in entries {
        if name.len() > u16::MAX as usize {
            return Err(ForgeError::Fzm(format!("{name}: name too long")));
        }
        if shape.len() > u8::MAX as usize {
            return Err(ForgeError::Fzm(format!("{name}: too many dims")));
        }
        let numel: usize = shape.iter().product();
        if numel != data.len() {
            return Err(ForgeError::Fzm(format!(
                "{name}: data length {} does not match shape {shape:?}",
                data.len()
            )));
        }
        out.extend_from_slice(&(name.len() as u16).to_le_bytes());
        out.extend_from_slice(name.as_bytes());
        out.push(shape.len() as u8);
        for &d in shape {
            out.extend_from_slice(&(d as u32).to_le_bytes());
        }
        let n_groups = numel.div_ceil(GROUP_SIZE).max(1);
        out.extend_from_slice(&(GROUP_SIZE as u32).to_le_bytes());
        out.extend_from_slice(&(n_groups as u32).to_le_bytes());

        let mut codes = vec![0u8; numel];
        for g in 0..n_groups {
            let start = g * GROUP_SIZE;
            let end = (start + GROUP_SIZE).min(numel);
            let group = &data[start..end];
            let (min, max) = group
                .iter()
                .fold((f32::MAX, f32::MIN), |(lo, hi), &x| (lo.min(x), hi.max(x)));
            let scale = if max > min { (max - min) / 15.0 } else { 1.0 };
            out.extend_from_slice(&scale.to_le_bytes());
            out.extend_from_slice(&min.to_le_bytes());
            for (i, &x) in group.iter().enumerate() {
                let q = ((x - min) / scale).round().clamp(0.0, 15.0) as u8;
                codes[start + i] = q;
            }
        }
        for pair in codes.chunks(2) {
            let lo = pair[0];
            let hi = pair.get(1).copied().unwrap_or(0);
            out.push(lo | (hi << 4));
        }
    }

    std::fs::write(path.as_ref(), &out)
        .map_err(|e| ForgeError::Fzm(format!("{}: {e}", path.as_ref().display())))
}

fn take<'a>(bytes: &'a [u8], pos: &mut usize, n: usize) -> Result<&'a [u8]> {
    let end = pos
        .checked_add(n)
        .ok_or_else(|| ForgeError::Fzm("truncated".into()))?;
    let slice = bytes
        .get(*pos..end)
        .ok_or_else(|| ForgeError::Fzm("truncated".into()))?;
    *pos = end;
    Ok(slice)
}

/// Load a `.fzm` file from disk, dequantized to f32.
pub fn load_fzm_q4_file(path: impl AsRef<Path>) -> Result<TensorEntries> {
    let bytes = std::fs::read(path.as_ref())
        .map_err(|e| ForgeError::Fzm(format!("{}: {e}", path.as_ref().display())))?;
    load_fzm_q4(&bytes)
}

/// Parse `.fzm` bytes into `(name, shape, host f32 data)`, dequantized on
/// load — same shape the safetensors `host()` closure produces, so callers
/// (e.g. [`crate::models::gpt2::Gpt2::from_fzm_bytes`]) plug it in the same way.
pub fn load_fzm_q4(bytes: &[u8]) -> Result<TensorEntries> {
    let mut pos = 0usize;

    if take(bytes, &mut pos, 4)? != MAGIC {
        return Err(ForgeError::Fzm("bad magic".into()));
    }
    let tensor_count = u32::from_le_bytes(take(bytes, &mut pos, 4)?.try_into().unwrap());

    let mut out = Vec::with_capacity(tensor_count as usize);
    for _ in 0..tensor_count {
        let name_len = u16::from_le_bytes(take(bytes, &mut pos, 2)?.try_into().unwrap()) as usize;
        let name = String::from_utf8(take(bytes, &mut pos, name_len)?.to_vec())
            .map_err(|_| ForgeError::Fzm("bad utf8 name".into()))?;
        let ndim = take(bytes, &mut pos, 1)?[0] as usize;
        let mut shape = Vec::with_capacity(ndim);
        for _ in 0..ndim {
            shape.push(u32::from_le_bytes(take(bytes, &mut pos, 4)?.try_into().unwrap()) as usize);
        }
        let group_size = u32::from_le_bytes(take(bytes, &mut pos, 4)?.try_into().unwrap()) as usize;
        let n_groups = u32::from_le_bytes(take(bytes, &mut pos, 4)?.try_into().unwrap()) as usize;
        let numel: usize = shape.iter().product();

        let mut groups = Vec::with_capacity(n_groups);
        for _ in 0..n_groups {
            let scale = f32::from_le_bytes(take(bytes, &mut pos, 4)?.try_into().unwrap());
            let min = f32::from_le_bytes(take(bytes, &mut pos, 4)?.try_into().unwrap());
            groups.push((scale, min));
        }

        let packed_len = numel.div_ceil(2);
        let packed = take(bytes, &mut pos, packed_len)?;
        let mut data = Vec::with_capacity(numel);
        for i in 0..numel {
            let byte = packed[i / 2];
            let q = if i % 2 == 0 { byte & 0x0F } else { byte >> 4 };
            let (scale, min) = groups[i / group_size];
            data.push(q as f32 * scale + min);
        }
        out.push((name, shape, data));
    }
    Ok(out)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn round_trips_within_q4_error() {
        let data: Vec<f32> = (0..200).map(|i| (i as f32 * 0.037).sin()).collect();
        let entries = vec![("w".to_string(), vec![200], data.clone())];
        let dir = std::env::temp_dir().join("forge_fzm_test.fzm");
        save_fzm_q4(&dir, &entries).unwrap();
        let loaded = load_fzm_q4_file(&dir).unwrap();
        std::fs::remove_file(&dir).ok();

        assert_eq!(loaded.len(), 1);
        let (name, shape, out) = &loaded[0];
        assert_eq!(name, "w");
        assert_eq!(shape, &vec![200]);
        assert_eq!(out.len(), data.len());
        // q4 over a [-1, 1] range: step is at most 2/15 ≈ 0.133 per group.
        for (a, b) in data.iter().zip(out.iter()) {
            assert!((a - b).abs() < 0.15, "a={a} b={b}");
        }
    }

    #[test]
    fn rejects_bad_magic() {
        assert!(load_fzm_q4(b"NOPE").is_err());
    }
}
