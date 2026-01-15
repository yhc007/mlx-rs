//! Data types for MLX arrays

use std::fmt;
use mlx_sys::mlx_dtype;

/// Data types supported by MLX arrays
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
#[repr(u32)]
pub enum DType {
    /// Boolean type
    Bool = 0,
    /// Unsigned 8-bit integer
    UInt8 = 1,
    /// Unsigned 16-bit integer
    UInt16 = 2,
    /// Unsigned 32-bit integer
    UInt32 = 3,
    /// Unsigned 64-bit integer
    UInt64 = 4,
    /// Signed 8-bit integer
    Int8 = 5,
    /// Signed 16-bit integer
    Int16 = 6,
    /// Signed 32-bit integer
    Int32 = 7,
    /// Signed 64-bit integer
    Int64 = 8,
    /// 16-bit floating point (IEEE 754)
    Float16 = 9,
    /// 32-bit floating point
    Float32 = 10,
    /// 64-bit floating point
    Float64 = 11,
    /// Brain floating point (16-bit)
    BFloat16 = 12,
    /// Complex 64-bit (2x 32-bit float)
    Complex64 = 13,
}

impl DType {
    /// Get the size in bytes of this data type
    pub fn size_bytes(&self) -> usize {
        match self {
            DType::Bool | DType::UInt8 | DType::Int8 => 1,
            DType::UInt16 | DType::Int16 | DType::Float16 | DType::BFloat16 => 2,
            DType::UInt32 | DType::Int32 | DType::Float32 => 4,
            DType::UInt64 | DType::Int64 | DType::Float64 | DType::Complex64 => 8,
        }
    }

    /// Check if this is a floating point type
    pub fn is_floating_point(&self) -> bool {
        matches!(
            self,
            DType::Float16 | DType::Float32 | DType::Float64 | DType::BFloat16 | DType::Complex64
        )
    }

    /// Check if this is an integer type
    pub fn is_integer(&self) -> bool {
        matches!(
            self,
            DType::Int8
                | DType::Int16
                | DType::Int32
                | DType::Int64
                | DType::UInt8
                | DType::UInt16
                | DType::UInt32
                | DType::UInt64
        )
    }

    /// Check if this is a signed type
    pub fn is_signed(&self) -> bool {
        matches!(
            self,
            DType::Int8
                | DType::Int16
                | DType::Int32
                | DType::Int64
                | DType::Float16
                | DType::Float32
                | DType::Float64
                | DType::BFloat16
                | DType::Complex64
        )
    }

    /// Check if this is an unsigned type
    pub fn is_unsigned(&self) -> bool {
        matches!(
            self,
            DType::UInt8 | DType::UInt16 | DType::UInt32 | DType::UInt64 | DType::Bool
        )
    }

    /// Check if this is a complex type
    pub fn is_complex(&self) -> bool {
        matches!(self, DType::Complex64)
    }
}

impl fmt::Display for DType {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        let name = match self {
            DType::Bool => "bool",
            DType::UInt8 => "uint8",
            DType::UInt16 => "uint16",
            DType::UInt32 => "uint32",
            DType::UInt64 => "uint64",
            DType::Int8 => "int8",
            DType::Int16 => "int16",
            DType::Int32 => "int32",
            DType::Int64 => "int64",
            DType::Float16 => "float16",
            DType::Float32 => "float32",
            DType::Float64 => "float64",
            DType::BFloat16 => "bfloat16",
            DType::Complex64 => "complex64",
        };
        write!(f, "{}", name)
    }
}

impl From<DType> for mlx_dtype {
    fn from(dtype: DType) -> Self {
        dtype as mlx_dtype
    }
}

impl From<mlx_dtype> for DType {
    fn from(dtype: mlx_dtype) -> Self {
        match dtype {
            0 => DType::Bool,
            1 => DType::UInt8,
            2 => DType::UInt16,
            3 => DType::UInt32,
            4 => DType::UInt64,
            5 => DType::Int8,
            6 => DType::Int16,
            7 => DType::Int32,
            8 => DType::Int64,
            9 => DType::Float16,
            10 => DType::Float32,
            11 => DType::Float64,
            12 => DType::BFloat16,
            13 => DType::Complex64,
            _ => DType::Float32, // Default fallback
        }
    }
}

/// Trait for types that can be converted to MLX arrays
pub trait ArrayElement: Copy + Default + Send + Sync + 'static {
    /// The MLX dtype for this element type
    const DTYPE: DType;
}

impl ArrayElement for bool {
    const DTYPE: DType = DType::Bool;
}

impl ArrayElement for u8 {
    const DTYPE: DType = DType::UInt8;
}

impl ArrayElement for u16 {
    const DTYPE: DType = DType::UInt16;
}

impl ArrayElement for u32 {
    const DTYPE: DType = DType::UInt32;
}

impl ArrayElement for u64 {
    const DTYPE: DType = DType::UInt64;
}

impl ArrayElement for i8 {
    const DTYPE: DType = DType::Int8;
}

impl ArrayElement for i16 {
    const DTYPE: DType = DType::Int16;
}

impl ArrayElement for i32 {
    const DTYPE: DType = DType::Int32;
}

impl ArrayElement for i64 {
    const DTYPE: DType = DType::Int64;
}

impl ArrayElement for f32 {
    const DTYPE: DType = DType::Float32;
}

impl ArrayElement for f64 {
    const DTYPE: DType = DType::Float64;
}
