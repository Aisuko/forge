//! SafeTensors saving: named tensors -> file.
//!
//! There is deliberately no generic loader here. Model loading goes through
//! [`crate::models::gpt2::Gpt2::from_safetensors_bytes`], which deserializes
//! straight into the model's own parameter layout rather than materializing an
//! intermediate `HashMap<String, Tensor>` of every tensor in the file.

use std::path::Path;

use safetensors::tensor::{Dtype, TensorView};

use crate::error::{ForgeError, Result};

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
