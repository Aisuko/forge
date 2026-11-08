use std::path::{Path, PathBuf};

use safetensors::tensor::{Dtype, TensorView};

use crate::error::{ForgeError, Result};

pub mod fzm;

pub fn checkpoint_in_dir(dir: impl AsRef<Path>) -> Result<PathBuf> {
    let dir = dir.as_ref();
    for name in ["model.fzm", "model.safetensors"] {
        let path = dir.join(name);
        if path.is_file() {
            return Ok(path);
        }
    }
    Err(ForgeError::Fzm(format!(
        "{}: no model.fzm or model.safetensors",
        dir.display()
    )))
}

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

pub(crate) fn view_f32(name: &str, view: &TensorView<'_>) -> Result<Vec<f32>> {
    match view.dtype() {
        Dtype::F32 => Ok(bytemuck::pod_collect_to_vec(view.data())),
        Dtype::BF16 => Ok(view
            .data()
            .chunks_exact(2)
            .map(|c| f32::from_bits((u16::from_le_bytes([c[0], c[1]]) as u32) << 16))
            .collect()),
        other => Err(ForgeError::SafeTensors(format!(
            "{name}: unsupported dtype {other:?}"
        ))),
    }
}

pub fn load_safetensors(bytes: &[u8]) -> Result<fzm::TensorEntries> {
    let st = safetensors::SafeTensors::deserialize(bytes)
        .map_err(|e| ForgeError::SafeTensors(format!("deserialize: {e}")))?;
    st.tensors()
        .into_iter()
        .map(|(name, view)| {
            let data = view_f32(&name, &view)?;
            Ok((name, view.shape().to_vec(), data))
        })
        .collect()
}
