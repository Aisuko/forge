//! SafeTensors saving: named tensors -> file.
//!
//! There is deliberately no generic loader here. Model loading goes through
//! [`crate::models::gpt2::Gpt2::from_safetensors_bytes`], which deserializes
//! straight into the model's own parameter layout rather than materializing an
//! intermediate `HashMap<String, Tensor>` of every tensor in the file.
//!
//! [`fzm`] is the quantized sibling format — see its module docs.

use std::path::{Path, PathBuf};

use safetensors::tensor::{Dtype, TensorView};

use crate::error::{ForgeError, Result};

pub mod fzm;

/// The checkpoint inside a model directory, whichever format it is in.
///
/// A model directory holds `config.json`, `vocab.json` and one set of weights
/// named `model.fzm` or `model.safetensors`. `.fzm` wins when both are present:
/// it is what the shipped assets and the site now carry, and a leftover f32
/// file beside it should not silently change which bytes get loaded.
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
