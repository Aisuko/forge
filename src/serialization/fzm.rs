use std::path::Path;

use serde::{Deserialize, Serialize};
use serde_json::Value;

use crate::dtype::{Q4, Q4_GROUP};
use crate::error::{ForgeError, Result};

const MAGIC_V1: &[u8; 4] = b"FZM1";
const MAGIC_V2: &[u8; 4] = b"FZM2";

pub const GROUP_SIZE: usize = Q4_GROUP;

pub type TensorEntries = Vec<(String, Vec<usize>, Vec<f32>)>;

pub type Assets = Vec<(String, Vec<u8>)>;

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum FzmDtype {
    Q4,
    F32,
    Bf16,
}

impl FzmDtype {
    fn code(self) -> u8 {
        match self {
            FzmDtype::Q4 => 0,
            FzmDtype::F32 => 1,
            FzmDtype::Bf16 => 2,
        }
    }

    fn from_code(code: u8) -> Result<Self> {
        match code {
            0 => Ok(FzmDtype::Q4),
            1 => Ok(FzmDtype::F32),
            2 => Ok(FzmDtype::Bf16),
            _ => Err(ForgeError::Fzm(format!("unknown dtype code {code}"))),
        }
    }

    pub fn name(self) -> &'static str {
        match self {
            FzmDtype::Q4 => "q4",
            FzmDtype::F32 => "f32",
            FzmDtype::Bf16 => "bf16",
        }
    }
}

#[derive(Clone, Debug, Default, PartialEq, Serialize, Deserialize)]
pub struct Quantization {
    #[serde(rename = "type")]
    pub kind: String,
    #[serde(default)]
    pub group_size: usize,
}

#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
pub struct FzmManifest {
    pub format: String,
    pub version: u32,
    pub architecture: String,
    #[serde(default)]
    pub task: String,
    #[serde(default)]
    pub model: Value,
    #[serde(default)]
    pub components: Value,
    #[serde(default)]
    pub quantization: Quantization,
    #[serde(default)]
    pub config: Value,
    #[serde(default)]
    pub provenance: Value,
}

impl FzmManifest {
    pub fn new(architecture: impl Into<String>, config: Value) -> Self {
        FzmManifest {
            format: "fzm".into(),
            version: 2,
            architecture: architecture.into(),
            task: String::new(),
            model: Value::Null,
            components: Value::Null,
            quantization: Quantization::default(),
            config,
            provenance: Value::Null,
        }
    }
}

#[derive(Clone, Debug, PartialEq)]
pub enum Body {
    Q4(Q4),
    F32(Vec<f32>),
}

impl Body {
    pub fn into_f32(self) -> Vec<f32> {
        match self {
            Body::Q4(q) => q.dequant(),
            Body::F32(v) => v,
        }
    }
}

pub type RawEntries = Vec<(String, Vec<usize>, Body)>;

pub struct Fzm {
    pub version: u32,
    pub manifest: Option<FzmManifest>,
    pub entries: RawEntries,
    pub assets: Assets,
}

impl Fzm {
    pub fn into_f32_entries(self) -> TensorEntries {
        self.entries
            .into_iter()
            .map(|(n, s, b)| (n, s, b.into_f32()))
            .collect()
    }
}

pub fn is_fzm(bytes: &[u8]) -> bool {
    bytes.starts_with(MAGIC_V1) || bytes.starts_with(MAGIC_V2)
}

fn u32_len(what: &str, n: usize) -> Result<[u8; 4]> {
    u32::try_from(n)
        .map(u32::to_le_bytes)
        .map_err(|_| ForgeError::Fzm(format!("{what}: length {n} exceeds u32")))
}

fn put_name_shape(out: &mut Vec<u8>, name: &str, shape: &[usize], numel: usize) -> Result<()> {
    if name.len() > u16::MAX as usize {
        return Err(ForgeError::Fzm(format!("{name}: name too long")));
    }
    if shape.len() > u8::MAX as usize {
        return Err(ForgeError::Fzm(format!("{name}: too many dims")));
    }
    if shape.iter().product::<usize>() != numel {
        return Err(ForgeError::Fzm(format!(
            "{name}: data length {numel} does not match shape {shape:?}"
        )));
    }
    out.extend_from_slice(&(name.len() as u16).to_le_bytes());
    out.extend_from_slice(name.as_bytes());
    out.push(shape.len() as u8);
    for &d in shape {
        out.extend_from_slice(&u32_len(name, d)?);
    }
    Ok(())
}

fn put_q4(out: &mut Vec<u8>, name: &str, data: &[f32]) -> Result<()> {
    out.extend_from_slice(&(GROUP_SIZE as u32).to_le_bytes());
    out.extend_from_slice(&u32_len(name, Q4::groups(data.len()))?);
    Q4::quantize(data).write_fzm(out);
    Ok(())
}

fn bf16(x: f32) -> u16 {
    let b = x.to_bits();
    if x.is_nan() {
        return ((b >> 16) | 0x40) as u16;
    }
    ((b + 0x7FFF + ((b >> 16) & 1)) >> 16) as u16
}

fn write_file(path: &Path, out: &[u8]) -> Result<()> {
    std::fs::write(path, out).map_err(|e| ForgeError::Fzm(format!("{}: {e}", path.display())))
}

pub fn save_fzm_q4(
    path: impl AsRef<Path>,
    entries: &[(String, Vec<usize>, Vec<f32>)],
) -> Result<()> {
    let mut out = Vec::new();
    out.extend_from_slice(MAGIC_V1);
    out.extend_from_slice(&u32_len("tensor count", entries.len())?);
    for (name, shape, data) in entries {
        put_name_shape(&mut out, name, shape, data.len())?;
        put_q4(&mut out, name, data)?;
    }
    write_file(path.as_ref(), &out)
}

pub fn encode_fzm_v2(
    manifest: &FzmManifest,
    entries: &[(String, Vec<usize>, Vec<f32>)],
    dtype: FzmDtype,
    assets: &[(String, Vec<u8>)],
) -> Result<Vec<u8>> {
    let mut manifest = manifest.clone();
    manifest.format = "fzm".into();
    manifest.version = 2;
    manifest.quantization = Quantization {
        kind: dtype.name().into(),
        group_size: if dtype == FzmDtype::Q4 { GROUP_SIZE } else { 0 },
    };
    let json = serde_json::to_vec(&manifest)?;

    let mut out = Vec::new();
    out.extend_from_slice(MAGIC_V2);
    out.extend_from_slice(&u32_len("manifest", json.len())?);
    out.extend_from_slice(&json);

    out.extend_from_slice(&u32_len("asset count", assets.len())?);
    for (name, data) in assets {
        if name.len() > u16::MAX as usize {
            return Err(ForgeError::Fzm(format!("{name}: asset name too long")));
        }
        out.extend_from_slice(&(name.len() as u16).to_le_bytes());
        out.extend_from_slice(name.as_bytes());
        out.extend_from_slice(&u32_len(name, data.len())?);
        out.extend_from_slice(data);
    }

    out.extend_from_slice(&u32_len("tensor count", entries.len())?);
    for (name, shape, data) in entries {
        put_name_shape(&mut out, name, shape, data.len())?;
        out.push(dtype.code());
        match dtype {
            FzmDtype::Q4 => put_q4(&mut out, name, data)?,
            FzmDtype::F32 => {
                out.extend_from_slice(&[0; 8]);
                for &x in data {
                    out.extend_from_slice(&x.to_le_bytes());
                }
            }
            FzmDtype::Bf16 => {
                out.extend_from_slice(&[0; 8]);
                for &x in data {
                    out.extend_from_slice(&bf16(x).to_le_bytes());
                }
            }
        }
    }
    Ok(out)
}

pub fn save_fzm_v2(
    path: impl AsRef<Path>,
    manifest: &FzmManifest,
    entries: &[(String, Vec<usize>, Vec<f32>)],
    dtype: FzmDtype,
    assets: &[(String, Vec<u8>)],
) -> Result<()> {
    write_file(
        path.as_ref(),
        &encode_fzm_v2(manifest, entries, dtype, assets)?,
    )
}

fn truncated() -> ForgeError {
    ForgeError::Fzm("truncated".into())
}

fn take<'a>(bytes: &'a [u8], pos: &mut usize, n: usize) -> Result<&'a [u8]> {
    let end = pos.checked_add(n).ok_or_else(truncated)?;
    let slice = bytes.get(*pos..end).ok_or_else(truncated)?;
    *pos = end;
    Ok(slice)
}

fn take_u16(bytes: &[u8], pos: &mut usize) -> Result<usize> {
    Ok(u16::from_le_bytes(take(bytes, pos, 2)?.try_into().unwrap()) as usize)
}

fn take_u32(bytes: &[u8], pos: &mut usize) -> Result<usize> {
    Ok(u32::from_le_bytes(take(bytes, pos, 4)?.try_into().unwrap()) as usize)
}

fn take_string(bytes: &[u8], pos: &mut usize, n: usize) -> Result<String> {
    String::from_utf8(take(bytes, pos, n)?.to_vec())
        .map_err(|_| ForgeError::Fzm("bad utf8 name".into()))
}

fn bytes_for(name: &str, numel: usize, width: usize) -> Result<usize> {
    numel
        .checked_mul(width)
        .ok_or_else(|| ForgeError::Fzm(format!("{name}: tensor too large")))
}

struct TensorHeader {
    name: String,
    shape: Vec<usize>,
    dtype: FzmDtype,
    numel: usize,
}

fn read_tensor_header(bytes: &[u8], pos: &mut usize, version: u32) -> Result<TensorHeader> {
    let name_len = take_u16(bytes, pos)?;
    let name = take_string(bytes, pos, name_len)?;
    let ndim = take(bytes, pos, 1)?[0] as usize;
    let mut shape = Vec::with_capacity(ndim);
    for _ in 0..ndim {
        shape.push(take_u32(bytes, pos)?);
    }
    let numel = shape
        .iter()
        .try_fold(1usize, |acc, &d| acc.checked_mul(d))
        .ok_or_else(|| ForgeError::Fzm(format!("{name}: shape {shape:?} overflows")))?;
    let dtype = if version == 1 {
        FzmDtype::Q4
    } else {
        FzmDtype::from_code(take(bytes, pos, 1)?[0])?
    };
    let group_size = take_u32(bytes, pos)?;
    let n_groups = take_u32(bytes, pos)?;
    if dtype == FzmDtype::Q4 && (group_size != GROUP_SIZE || n_groups != Q4::groups(numel)) {
        return Err(ForgeError::Fzm(format!(
            "{name}: q4 group size {group_size} x {n_groups} groups does not fit {numel} elements"
        )));
    }
    Ok(TensorHeader {
        name,
        shape,
        dtype,
        numel,
    })
}

fn tensor_body<'a>(
    bytes: &'a [u8],
    pos: &mut usize,
    h: &TensorHeader,
) -> Result<(&'a [u8], &'a [u8])> {
    match h.dtype {
        FzmDtype::Q4 => Ok((
            take(bytes, pos, Q4::groups(h.numel) * 8)?,
            take(bytes, pos, h.numel.div_ceil(2))?,
        )),
        FzmDtype::F32 => Ok((take(bytes, pos, bytes_for(&h.name, h.numel, 4)?)?, &[])),
        FzmDtype::Bf16 => Ok((take(bytes, pos, bytes_for(&h.name, h.numel, 2)?)?, &[])),
    }
}

fn read_tensor_body(bytes: &[u8], pos: &mut usize, h: &TensorHeader) -> Result<Body> {
    let (a, b) = tensor_body(bytes, pos, h)?;
    Ok(match h.dtype {
        FzmDtype::Q4 => Body::Q4(Q4::from_fzm(h.numel, a, b)),
        FzmDtype::F32 => Body::F32(
            a.chunks_exact(4)
                .map(|c| f32::from_le_bytes(c.try_into().unwrap()))
                .collect(),
        ),
        FzmDtype::Bf16 => Body::F32(
            a.chunks_exact(2)
                .map(|c| f32::from_bits((u16::from_le_bytes(c.try_into().unwrap()) as u32) << 16))
                .collect(),
        ),
    })
}

struct Preamble<'a> {
    version: u32,
    manifest: Option<FzmManifest>,
    assets: Vec<(String, &'a [u8])>,
    tensor_count: usize,
}

fn read_preamble<'a>(bytes: &'a [u8], pos: &mut usize) -> Result<Preamble<'a>> {
    let magic = take(bytes, pos, 4)?;
    if magic == MAGIC_V1 {
        let tensor_count = take_u32(bytes, pos)?;
        return Ok(Preamble {
            version: 1,
            manifest: None,
            assets: Vec::new(),
            tensor_count,
        });
    }
    if magic != MAGIC_V2 {
        return Err(ForgeError::Fzm("bad magic".into()));
    }
    let manifest_len = take_u32(bytes, pos)?;
    let manifest: FzmManifest = serde_json::from_slice(take(bytes, pos, manifest_len)?)?;
    if manifest.version != 2 {
        return Err(ForgeError::Fzm(format!(
            "manifest version {} in an FZM2 file",
            manifest.version
        )));
    }
    let asset_count = take_u32(bytes, pos)?;
    let mut assets = Vec::new();
    for _ in 0..asset_count {
        let name_len = take_u16(bytes, pos)?;
        let name = take_string(bytes, pos, name_len)?;
        let len = take_u32(bytes, pos)?;
        assets.push((name, take(bytes, pos, len)?));
    }
    let tensor_count = take_u32(bytes, pos)?;
    Ok(Preamble {
        version: 2,
        manifest: Some(manifest),
        assets,
        tensor_count,
    })
}

pub fn manifest(bytes: &[u8]) -> Result<Option<FzmManifest>> {
    Ok(read_preamble(bytes, &mut 0)?.manifest)
}

pub fn check_architecture(manifest: Option<&FzmManifest>, expected: &str) -> Result<()> {
    match manifest {
        Some(m) if m.architecture != expected => Err(ForgeError::Fzm(format!(
            "checkpoint architecture is {}, not {expected}",
            m.architecture
        ))),
        _ => Ok(()),
    }
}

pub fn read_fzm_header(bytes: &[u8]) -> Result<Vec<(String, Vec<usize>, FzmDtype)>> {
    let mut pos = 0usize;
    let pre = read_preamble(bytes, &mut pos)?;
    let mut out = Vec::new();
    for _ in 0..pre.tensor_count {
        let h = read_tensor_header(bytes, &mut pos, pre.version)?;
        tensor_body(bytes, &mut pos, &h)?;
        out.push((h.name, h.shape, h.dtype));
    }
    Ok(out)
}

pub fn load_fzm(bytes: &[u8]) -> Result<Fzm> {
    let mut pos = 0usize;
    let pre = read_preamble(bytes, &mut pos)?;
    let mut entries = Vec::new();
    for _ in 0..pre.tensor_count {
        let h = read_tensor_header(bytes, &mut pos, pre.version)?;
        let body = read_tensor_body(bytes, &mut pos, &h)?;
        entries.push((h.name, h.shape, body));
    }
    Ok(Fzm {
        version: pre.version,
        manifest: pre.manifest,
        entries,
        assets: pre
            .assets
            .into_iter()
            .map(|(n, d)| (n, d.to_vec()))
            .collect(),
    })
}

pub fn load_fzm_file(path: impl AsRef<Path>) -> Result<Fzm> {
    let bytes = std::fs::read(path.as_ref())
        .map_err(|e| ForgeError::Fzm(format!("{}: {e}", path.as_ref().display())))?;
    load_fzm(&bytes)
}

pub fn load_fzm_q4_file(path: impl AsRef<Path>) -> Result<TensorEntries> {
    Ok(load_fzm_file(path)?.into_f32_entries())
}

pub fn load_fzm_q4(bytes: &[u8]) -> Result<TensorEntries> {
    Ok(load_fzm(bytes)?.into_f32_entries())
}

#[cfg(test)]
mod tests {
    use super::*;

    fn sample() -> TensorEntries {
        vec![
            (
                "a".to_string(),
                vec![3, 7],
                (0..21).map(|i| (i as f32 * 0.31).sin()).collect(),
            ),
            (
                "b.weight".to_string(),
                vec![130],
                (0..130).map(|i| (i as f32 * 0.037).cos()).collect(),
            ),
        ]
    }

    #[test]
    fn round_trips_within_q4_error() {
        let data: Vec<f32> = (0..200).map(|i| (i as f32 * 0.037).sin()).collect();
        let entries = vec![("w".to_string(), vec![200], data.clone())];
        let path = std::env::temp_dir().join("forge_fzm_test.fzm");
        save_fzm_q4(&path, &entries).unwrap();
        let loaded = load_fzm_q4_file(&path).unwrap();
        std::fs::remove_file(&path).ok();

        assert_eq!(loaded.len(), 1);
        let (name, shape, out) = &loaded[0];
        assert_eq!(name, "w");
        assert_eq!(shape, &vec![200]);
        for (a, b) in data.iter().zip(out.iter()) {
            assert!((a - b).abs() < 0.15, "a={a} b={b}");
        }
    }

    #[test]
    fn rejects_bad_magic() {
        assert!(load_fzm(b"NOPE").is_err());
        assert!(read_fzm_header(b"NOPE").is_err());
        assert!(!is_fzm(b"NOPE"));
    }

    #[test]
    fn header_walk_matches_a_full_load() {
        let path = std::env::temp_dir().join("forge_fzm_header_test.fzm");
        save_fzm_q4(&path, &sample()).unwrap();
        let bytes = std::fs::read(&path).unwrap();
        std::fs::remove_file(&path).ok();

        let header = read_fzm_header(&bytes).unwrap();
        let full = load_fzm(&bytes).unwrap();
        assert_eq!(full.version, 1);
        assert!(full.manifest.is_none());
        assert_eq!(header.len(), full.entries.len());
        for ((hn, hs, hd), (fname, fshape, _)) in header.iter().zip(&full.entries) {
            assert_eq!(hn, fname);
            assert_eq!(hs, fshape);
            assert_eq!(*hd, FzmDtype::Q4);
        }
    }

    #[test]
    fn v2_round_trips_each_dtype() {
        let entries = sample();
        let manifest = FzmManifest::new("test_arch", serde_json::json!({ "d": 4 }));
        for (dtype, tol) in [
            (FzmDtype::Q4, 0.15),
            (FzmDtype::F32, 0.0),
            (FzmDtype::Bf16, 1e-2),
        ] {
            let bytes = encode_fzm_v2(&manifest, &entries, dtype, &[]).unwrap();
            assert!(is_fzm(&bytes));
            let fzm = load_fzm(&bytes).unwrap();
            assert_eq!(fzm.version, 2);
            let m = fzm.manifest.clone().unwrap();
            assert_eq!(m.architecture, "test_arch");
            assert_eq!(m.quantization.kind, dtype.name());
            assert_eq!(fzm.entries.len(), entries.len());
            for ((n0, s0, d0), (n1, s1, d1)) in entries.iter().zip(&fzm.into_f32_entries()) {
                assert_eq!(n0, n1);
                assert_eq!(s0, s1);
                for (a, b) in d0.iter().zip(d1) {
                    assert!((a - b).abs() <= tol, "{dtype:?}: a={a} b={b}");
                }
            }
            let header = read_fzm_header(&bytes).unwrap();
            assert_eq!(header.len(), entries.len());
            assert!(header.iter().all(|(_, _, d)| *d == dtype));
        }
    }

    #[test]
    fn v2_carries_manifest_and_assets() {
        let mut manifest = FzmManifest::new("encoder_decoder", serde_json::json!({ "d_model": 8 }));
        manifest.task = "intent_aware_summarisation".into();
        manifest.provenance = serde_json::json!({ "source": "unit" });
        let assets = vec![("tokenizer/tokenizer.json".to_string(), b"{}".to_vec())];
        let bytes = encode_fzm_v2(&manifest, &sample(), FzmDtype::F32, &assets).unwrap();
        let fzm = load_fzm(&bytes).unwrap();
        let m = fzm.manifest.unwrap();
        assert_eq!(m.task, manifest.task);
        assert_eq!(m.config, manifest.config);
        assert_eq!(m.provenance, manifest.provenance);
        assert_eq!(fzm.assets, assets);
    }

    #[test]
    fn v2_rejects_wrong_manifest_version_and_truncation() {
        let mut manifest = FzmManifest::new("x", Value::Null);
        let mut bytes = encode_fzm_v2(&manifest, &sample(), FzmDtype::F32, &[]).unwrap();
        bytes.truncate(bytes.len() - 3);
        assert!(load_fzm(&bytes).is_err());

        manifest.version = 3;
        let json = serde_json::to_vec(&manifest).unwrap();
        let mut forged = MAGIC_V2.to_vec();
        forged.extend_from_slice(&(json.len() as u32).to_le_bytes());
        forged.extend_from_slice(&json);
        forged.extend_from_slice(&0u32.to_le_bytes());
        forged.extend_from_slice(&0u32.to_le_bytes());
        assert!(load_fzm(&forged).is_err());
    }

    #[test]
    fn huge_counts_fail_without_allocating() {
        let mut v1 = MAGIC_V1.to_vec();
        v1.extend_from_slice(&u32::MAX.to_le_bytes());
        assert!(load_fzm(&v1).is_err());
        assert!(read_fzm_header(&v1).is_err());

        let json = serde_json::to_vec(&FzmManifest::new("x", Value::Null)).unwrap();
        let mut v2 = MAGIC_V2.to_vec();
        v2.extend_from_slice(&(json.len() as u32).to_le_bytes());
        v2.extend_from_slice(&json);
        v2.extend_from_slice(&u32::MAX.to_le_bytes());
        assert!(load_fzm(&v2).is_err());

        let mut groups = MAGIC_V1.to_vec();
        groups.extend_from_slice(&1u32.to_le_bytes());
        put_name_shape(&mut groups, "w", &[4], 4).unwrap();
        groups.extend_from_slice(&64u32.to_le_bytes());
        groups.extend_from_slice(&u32::MAX.to_le_bytes());
        assert!(load_fzm(&groups).is_err());

        let mut shape = MAGIC_V1.to_vec();
        shape.extend_from_slice(&1u32.to_le_bytes());
        shape.extend_from_slice(&1u16.to_le_bytes());
        shape.push(b'w');
        shape.push(4);
        for _ in 0..4 {
            shape.extend_from_slice(&u32::MAX.to_le_bytes());
        }
        assert!(load_fzm(&shape).is_err());
    }

    #[test]
    fn oversized_lengths_are_rejected_on_write() {
        assert!(u32_len("x", u32::MAX as usize).is_ok());
        #[cfg(target_pointer_width = "64")]
        assert!(u32_len("x", u32::MAX as usize + 1).is_err());
        let mut out = Vec::new();
        #[cfg(target_pointer_width = "64")]
        assert!(put_name_shape(&mut out, "w", &[1 << 33], 1 << 33).is_err());
        assert!(put_name_shape(&mut out, "w", &[2, 3], 5).is_err());
    }

    #[test]
    fn bf16_rounds_to_nearest_even() {
        let up = f32::from_bits(0x3F80_8001);
        assert_eq!(bf16(up), 0x3F81);
        let tie_even = f32::from_bits(0x3F80_8000);
        assert_eq!(bf16(tie_even), 0x3F80);
        let tie_odd = f32::from_bits(0x3F81_8000);
        assert_eq!(bf16(tie_odd), 0x3F82);
        assert_eq!(bf16(f32::from_bits(0x3F80_7FFF)), 0x3F80);
        let nan = f32::from_bits(0x7F80_0001);
        assert!(f32::from_bits((bf16(nan) as u32) << 16).is_nan());
        assert_eq!(bf16(f32::INFINITY), 0x7F80);
        assert_eq!(bf16(f32::MAX), 0x7F80);
    }

    #[test]
    fn architecture_check_and_peek() {
        let manifest = FzmManifest::new("gpt2", Value::Null);
        let bytes = encode_fzm_v2(&manifest, &sample(), FzmDtype::F32, &[]).unwrap();
        let m = super::manifest(&bytes).unwrap();
        assert_eq!(m.as_ref().unwrap().architecture, "gpt2");
        assert!(check_architecture(m.as_ref(), "gpt2").is_ok());
        assert!(check_architecture(m.as_ref(), "encoder_decoder").is_err());
        assert!(check_architecture(None, "gpt2").is_ok());
    }

    #[test]
    fn q4_blocks_match_the_fzm_layout() {
        let data: Vec<f32> = (0..131).map(|i| (i as f32 * 0.21).sin()).collect();
        let q = Q4::quantize(&data);
        let mut bytes = Vec::new();
        q.write_fzm(&mut bytes);
        let groups = Q4::groups(data.len());
        let back = Q4::from_fzm(data.len(), &bytes[..groups * 8], &bytes[groups * 8..]);
        assert_eq!(back, q);
        assert_eq!(bytes.len(), groups * 8 + data.len().div_ceil(2));
        let tail = q.slice(64, 67).dequant();
        assert_eq!(tail, q.dequant()[64..].to_vec());
    }
}
