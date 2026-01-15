//! Error types for MLX operations

use thiserror::Error;

use crate::DType;

/// Result type alias for MLX operations
pub type Result<T> = std::result::Result<T, Error>;

/// Errors that can occur during MLX operations
#[derive(Error, Debug)]
pub enum Error {
    /// Array creation failed
    #[error("Failed to create array: {0}")]
    ArrayCreation(String),

    /// Invalid shape specified
    #[error("Invalid shape: {0}")]
    InvalidShape(String),

    /// Shape mismatch between arrays
    #[error("Shape mismatch: expected {expected:?}, got {actual:?}")]
    ShapeMismatch {
        expected: Vec<i32>,
        actual: Vec<i32>,
    },

    /// Data type mismatch
    #[error("DType mismatch: expected {expected:?}, got {actual:?}")]
    DTypeMismatch { expected: DType, actual: DType },

    /// Invalid data type for operation
    #[error("Invalid dtype {0:?} for operation: {1}")]
    InvalidDType(DType, String),

    /// Index out of bounds
    #[error("Index out of bounds: {index} >= {size}")]
    IndexOutOfBounds { index: usize, size: usize },

    /// Dimension mismatch
    #[error("Dimension mismatch: expected {expected}, got {actual}")]
    DimensionMismatch { expected: usize, actual: usize },

    /// Invalid axis specified
    #[error("Invalid axis {axis} for array with {ndim} dimensions")]
    InvalidAxis { axis: i32, ndim: usize },

    /// Operation not supported
    #[error("Operation not supported: {0}")]
    NotSupported(String),

    /// Device error
    #[error("Device error: {0}")]
    DeviceError(String),

    /// Stream error
    #[error("Stream error: {0}")]
    StreamError(String),

    /// Memory allocation failed
    #[error("Memory allocation failed: {0}")]
    AllocationFailed(String),

    /// Null pointer encountered
    #[error("Null pointer encountered: {0}")]
    NullPointer(String),

    /// Conversion error
    #[error("Conversion error: {0}")]
    ConversionError(String),

    /// Internal MLX error
    #[error("Internal MLX error: {0}")]
    Internal(String),
}

impl Error {
    /// Create an error for invalid shape
    pub fn invalid_shape<S: Into<String>>(msg: S) -> Self {
        Error::InvalidShape(msg.into())
    }

    /// Create a shape mismatch error
    pub fn shape_mismatch(expected: &[i32], actual: &[i32]) -> Self {
        Error::ShapeMismatch {
            expected: expected.to_vec(),
            actual: actual.to_vec(),
        }
    }

    /// Create a dimension mismatch error
    pub fn dimension_mismatch(expected: usize, actual: usize) -> Self {
        Error::DimensionMismatch { expected, actual }
    }

    /// Create an invalid axis error
    pub fn invalid_axis(axis: i32, ndim: usize) -> Self {
        Error::InvalidAxis { axis, ndim }
    }
}
