#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum DType {
    F32,
    U32,
    Q4,
}

pub const Q4_GROUP: usize = 64;

const Q4_BLOCK: usize = 10;

#[derive(Clone, Debug, PartialEq)]
pub struct Q4 {
    numel: usize,
    blocks: Vec<u32>,
}

impl Q4 {
    pub fn groups(numel: usize) -> usize {
        numel.div_ceil(Q4_GROUP).max(1)
    }

    pub fn device_bytes(numel: usize) -> usize {
        Self::groups(numel) * Q4_BLOCK * 4
    }

    pub fn quantize(data: &[f32]) -> Q4 {
        let n = data.len();
        let mut blocks = vec![0u32; Self::groups(n) * Q4_BLOCK];
        for (g, block) in blocks.chunks_exact_mut(Q4_BLOCK).enumerate() {
            let group = &data[(g * Q4_GROUP).min(n)..((g + 1) * Q4_GROUP).min(n)];
            let (min, max) = group
                .iter()
                .fold((f32::MAX, f32::MIN), |(lo, hi), &x| (lo.min(x), hi.max(x)));
            let scale = if max > min { (max - min) / 15.0 } else { 1.0 };
            block[0] = scale.to_bits();
            block[1] = min.to_bits();
            for (j, &x) in group.iter().enumerate() {
                let q = ((x - min) / scale).round().clamp(0.0, 15.0) as u32;
                block[2 + j / 8] |= q << (j % 8 * 4);
            }
        }
        Q4 { numel: n, blocks }
    }

    pub fn from_fzm(numel: usize, groups: &[u8], packed: &[u8]) -> Q4 {
        let mut blocks = vec![0u32; Self::groups(numel) * Q4_BLOCK];
        for (g, block) in blocks.chunks_exact_mut(Q4_BLOCK).enumerate() {
            let word = |i: usize| u32::from_le_bytes(groups[i..i + 4].try_into().unwrap());
            block[0] = word(g * 8);
            block[1] = word(g * 8 + 4);
            let bytes = &packed[(g * 32).min(packed.len())..((g + 1) * 32).min(packed.len())];
            for (i, &b) in bytes.iter().enumerate() {
                block[2 + i / 4] |= (b as u32) << (i % 4 * 8);
            }
        }
        Q4 { numel, blocks }
    }

    pub fn write_fzm(&self, out: &mut Vec<u8>) {
        for block in self.blocks.chunks_exact(Q4_BLOCK) {
            out.extend_from_slice(&block[0].to_le_bytes());
            out.extend_from_slice(&block[1].to_le_bytes());
        }
        let packed: Vec<u8> = self
            .blocks
            .chunks_exact(Q4_BLOCK)
            .flat_map(|b| b[2..].iter().flat_map(|w| w.to_le_bytes()))
            .take(self.numel.div_ceil(2))
            .collect();
        out.extend_from_slice(&packed);
    }

    pub(crate) fn from_blocks(numel: usize, blocks: Vec<u32>) -> Q4 {
        Q4 { numel, blocks }
    }

    pub fn numel(&self) -> usize {
        self.numel
    }

    pub fn blocks(&self) -> &[u32] {
        &self.blocks
    }

    pub fn dequant(&self) -> Vec<f32> {
        (0..self.numel)
            .map(|i| {
                let b = &self.blocks[i / Q4_GROUP * Q4_BLOCK..];
                let q = (b[2 + i % Q4_GROUP / 8] >> (i % 8 * 4)) & 15;
                q as f32 * f32::from_bits(b[0]) + f32::from_bits(b[1])
            })
            .collect()
    }

    pub fn slice(&self, start: usize, len: usize) -> Q4 {
        assert!(start.is_multiple_of(Q4_GROUP) && start + len <= self.numel);
        let first = start / Q4_GROUP;
        let end = first + Self::groups(len);
        Q4 {
            numel: len,
            blocks: self.blocks[first * Q4_BLOCK..end * Q4_BLOCK].to_vec(),
        }
    }
}
