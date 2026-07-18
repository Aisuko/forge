use std::fmt;

/// Row-major, contiguous tensor shape.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct Shape(pub Vec<usize>);

impl Shape {
    pub fn new(dims: &[usize]) -> Self {
        Shape(dims.to_vec())
    }

    pub fn dims(&self) -> &[usize] {
        &self.0
    }

    pub fn rank(&self) -> usize {
        self.0.len()
    }

    /// Total number of elements.
    pub fn numel(&self) -> usize {
        self.0.iter().product()
    }

    pub fn dim(&self, i: usize) -> usize {
        self.0[i]
    }

    /// Row-major strides in elements.
    pub fn strides(&self) -> Vec<usize> {
        let mut strides = vec![1; self.0.len()];
        for i in (0..self.0.len().saturating_sub(1)).rev() {
            strides[i] = strides[i + 1] * self.0[i + 1];
        }
        strides
    }
}

impl fmt::Display for Shape {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "{:?}", self.0)
    }
}

impl From<&[usize]> for Shape {
    fn from(d: &[usize]) -> Self {
        Shape(d.to_vec())
    }
}

impl From<Vec<usize>> for Shape {
    fn from(d: Vec<usize>) -> Self {
        Shape(d)
    }
}

impl<const N: usize> From<[usize; N]> for Shape {
    fn from(d: [usize; N]) -> Self {
        Shape(d.to_vec())
    }
}
