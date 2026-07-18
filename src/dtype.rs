/// Element types supported by Forge 1.0: f32 compute, u32 indices.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum DType {
    F32,
    U32,
}

impl DType {
    pub fn size_bytes(self) -> usize {
        match self {
            DType::F32 | DType::U32 => 4,
        }
    }
}
