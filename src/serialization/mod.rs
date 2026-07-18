//! SafeTensors loading and saving: file <-> named tensors on a device.

use std::collections::HashMap;
use std::path::Path;

use safetensors::SafeTensors;
use safetensors::tensor::{Dtype, TensorView};

use crate::device::Device;
use crate::error::{ForgeError, Result};
use crate::tensor::Tensor;

/// Load every f32 tensor from a .safetensors file onto `device`.
/// Non-f32 tensors are skipped (GPT-2 checkpoints are pure f32).
pub fn load_safetensors(
    path: impl AsRef<Path>,
    device: &Device,
) -> Result<HashMap<String, Tensor>> {
    let bytes = std::fs::read(path.as_ref())?;
    load_safetensors_bytes(&bytes, device)
        .map_err(|e| ForgeError::SafeTensors(format!("{}: {e}", path.as_ref().display())))
}

/// Load every f32 tensor from in-memory .safetensors bytes onto `device` —
/// the primary form; on wasm the bytes arrive from an HTTP fetch.
pub fn load_safetensors_bytes(bytes: &[u8], device: &Device) -> Result<HashMap<String, Tensor>> {
    let st = SafeTensors::deserialize(bytes)
        .map_err(|e| ForgeError::SafeTensors(format!("deserialize: {e}")))?;
    let mut out = HashMap::new();
    for (name, view) in st.tensors() {
        if view.dtype() != Dtype::F32 {
            continue;
        }
        // pod_collect_to_vec copies, which also fixes up any misalignment of
        // the raw byte slice within the file buffer.
        let data: Vec<f32> = bytemuck::pod_collect_to_vec(view.data());
        let tensor = Tensor::from_f32(&data, view.shape(), device)?;
        out.insert(name, tensor);
    }
    Ok(out)
}

/// Save named f32 tensors to a .safetensors file (checkpoints).
/// Entries: (name, shape, host data).
pub fn save_safetensors(
    path: impl AsRef<Path>,
    entries: &[(String, Vec<usize>, Vec<f32>)],
) -> Result<()> {
    let views: Vec<(String, TensorView<'_>)> = entries
        .iter()
        .map(|(name, shape, data)| {
            TensorView::new(Dtype::F32, shape.clone(), bytemuck::cast_slice(data))
                .map(|v| (name.clone(), v))
                .map_err(|e| ForgeError::SafeTensors(format!("{name}: {e}")))
        })
        .collect::<Result<_>>()?;
    safetensors::serialize_to_file(views, &None, path.as_ref())
        .map_err(|e| ForgeError::SafeTensors(format!("{}: {e}", path.as_ref().display())))
}
