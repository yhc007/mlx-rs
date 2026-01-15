//! Array type and operations
//!
//! The `Array` type is the core data structure in MLX, representing
//! multi-dimensional arrays with lazy evaluation.

use std::fmt;
use std::ops::{Add, Div, Mul, Neg, Sub};

use mlx_sys;

use crate::dtype::{ArrayElement, DType};
use crate::error::{Error, Result};
use crate::stream::Stream;

/// A multi-dimensional array with lazy evaluation
///
/// `Array` is the fundamental data structure in MLX. It supports:
/// - Multiple data types (float32, int32, etc.)
/// - Arbitrary dimensions
/// - Lazy evaluation (operations are computed only when needed)
/// - Automatic memory management
pub struct Array {
    inner: mlx_sys::mlx_array,
}

impl Array {
    // ========================================================================
    // Constructors
    // ========================================================================

    /// Create an array from a raw MLX array handle
    #[allow(dead_code)]
    pub(crate) unsafe fn from_raw(inner: mlx_sys::mlx_array) -> Self {
        Self { inner }
    }

    /// Create an uninitialized array (for use with out-parameter APIs)
    pub(crate) fn new_uninit() -> Self {
        let inner = unsafe { mlx_sys::mlx_array_new() };
        Self { inner }
    }

    /// Create from raw pointer (for internal use)
    #[allow(dead_code)]
    pub(crate) unsafe fn from_ptr(inner: mlx_sys::mlx_array) -> Self {
        Self { inner }
    }

    /// Get the raw MLX array handle
    pub(crate) fn as_raw(&self) -> mlx_sys::mlx_array {
        self.inner
    }

    /// Get mutable pointer for out-parameter APIs
    pub(crate) fn as_mut_ptr(&mut self) -> *mut mlx_sys::mlx_array {
        &mut self.inner
    }

    /// Create an array from a slice of data
    pub fn from_slice<T: ArrayElement>(data: &[T], shape: &[i32]) -> Result<Self> {
        let expected_size: i32 = shape.iter().product();
        if data.len() != expected_size as usize {
            return Err(Error::InvalidShape(format!(
                "Data length {} does not match shape {:?} (expected {})",
                data.len(),
                shape,
                expected_size
            )));
        }

        let inner = unsafe {
            mlx_sys::mlx_array_new_data(
                data.as_ptr() as *const std::ffi::c_void,
                shape.as_ptr(),
                shape.len() as i32,
                T::DTYPE.into(),
            )
        };

        Ok(Self { inner })
    }

    /// Create a scalar array from a single value
    pub fn scalar<T: ArrayElement>(value: T) -> Result<Self> {
        Self::from_slice(&[value], &[])
    }

    /// Create a scalar from f32
    pub fn from_float(value: f32) -> Self {
        let inner = unsafe { mlx_sys::mlx_array_new_float(value) };
        Self { inner }
    }

    /// Create a scalar from i32
    pub fn from_int(value: i32) -> Self {
        let inner = unsafe { mlx_sys::mlx_array_new_int(value) };
        Self { inner }
    }

    /// Create a scalar from bool
    pub fn from_bool(value: bool) -> Self {
        let inner = unsafe { mlx_sys::mlx_array_new_bool(value) };
        Self { inner }
    }

    /// Create an array filled with zeros
    pub fn zeros<T: ArrayElement>(shape: &[i32]) -> Result<Self> {
        let stream = Stream::default();
        let mut result = Self::new_uninit();

        let status = unsafe {
            mlx_sys::mlx_zeros(
                result.as_mut_ptr(),
                shape.as_ptr(),
                shape.len(),
                T::DTYPE.into(),
                stream.as_raw(),
            )
        };

        if status != 0 {
            return Err(Error::ArrayCreation("Failed to create zeros array".into()));
        }

        Ok(result)
    }

    /// Create an array filled with ones
    pub fn ones<T: ArrayElement>(shape: &[i32]) -> Result<Self> {
        let stream = Stream::default();
        let mut result = Self::new_uninit();

        let status = unsafe {
            mlx_sys::mlx_ones(
                result.as_mut_ptr(),
                shape.as_ptr(),
                shape.len(),
                T::DTYPE.into(),
                stream.as_raw(),
            )
        };

        if status != 0 {
            return Err(Error::ArrayCreation("Failed to create ones array".into()));
        }

        Ok(result)
    }

    /// Create an identity matrix
    pub fn eye<T: ArrayElement>(n: i32, m: Option<i32>, k: i32) -> Result<Self> {
        let m = m.unwrap_or(n);
        let stream = Stream::default();
        let mut result = Self::new_uninit();

        let status = unsafe {
            mlx_sys::mlx_eye(
                result.as_mut_ptr(),
                n,
                m,
                k,
                T::DTYPE.into(),
                stream.as_raw(),
            )
        };

        if status != 0 {
            return Err(Error::ArrayCreation("Failed to create identity matrix".into()));
        }

        Ok(result)
    }

    /// Create a 1D array with evenly spaced values
    pub fn arange<T: ArrayElement>(start: f64, stop: f64, step: f64) -> Result<Self> {
        let stream = Stream::default();
        let mut result = Self::new_uninit();

        let status = unsafe {
            mlx_sys::mlx_arange(
                result.as_mut_ptr(),
                start,
                stop,
                step,
                T::DTYPE.into(),
                stream.as_raw(),
            )
        };

        if status != 0 {
            return Err(Error::ArrayCreation("Failed to create arange array".into()));
        }

        Ok(result)
    }

    /// Create a 1D array with evenly spaced values (inclusive)
    pub fn linspace<T: ArrayElement>(start: f64, stop: f64, num: i32) -> Result<Self> {
        let stream = Stream::default();
        let mut result = Self::new_uninit();

        let status = unsafe {
            mlx_sys::mlx_linspace(
                result.as_mut_ptr(),
                start,
                stop,
                num,
                T::DTYPE.into(),
                stream.as_raw(),
            )
        };

        if status != 0 {
            return Err(Error::ArrayCreation("Failed to create linspace array".into()));
        }

        Ok(result)
    }

    // ========================================================================
    // Properties
    // ========================================================================

    /// Get the number of dimensions
    pub fn ndim(&self) -> usize {
        unsafe { mlx_sys::mlx_array_ndim(self.inner) }
    }

    /// Get the shape of the array
    pub fn shape(&self) -> Vec<i32> {
        let ndim = self.ndim();
        if ndim == 0 {
            return vec![];
        }

        let shape_ptr = unsafe { mlx_sys::mlx_array_shape(self.inner) };
        if shape_ptr.is_null() {
            return vec![];
        }

        let shape = unsafe { std::slice::from_raw_parts(shape_ptr, ndim) };
        shape.to_vec()
    }

    /// Get the total number of elements
    pub fn size(&self) -> usize {
        unsafe { mlx_sys::mlx_array_size(self.inner) }
    }

    /// Get the data type
    pub fn dtype(&self) -> DType {
        unsafe { mlx_sys::mlx_array_dtype(self.inner).into() }
    }

    /// Get the size of each element in bytes
    pub fn itemsize(&self) -> usize {
        unsafe { mlx_sys::mlx_array_itemsize(self.inner) }
    }

    /// Get the total number of bytes
    pub fn nbytes(&self) -> usize {
        unsafe { mlx_sys::mlx_array_nbytes(self.inner) }
    }

    /// Check if this is a scalar (0-dimensional array)
    pub fn is_scalar(&self) -> bool {
        self.ndim() == 0
    }

    // ========================================================================
    // Evaluation
    // ========================================================================

    /// Evaluate the array, forcing computation
    pub fn eval(&self) {
        unsafe {
            mlx_sys::mlx_array_eval(self.inner);
        }
    }

    // ========================================================================
    // Data Access
    // ========================================================================

    /// Get the data as a Vec
    pub fn to_vec<T: ArrayElement>(&self) -> Result<Vec<T>> {
        if self.dtype() != T::DTYPE {
            return Err(Error::DTypeMismatch {
                expected: T::DTYPE,
                actual: self.dtype(),
            });
        }

        self.eval();

        let size = self.size();
        let data_ptr = self.data_ptr::<T>()?;

        let slice = unsafe { std::slice::from_raw_parts(data_ptr, size) };
        Ok(slice.to_vec())
    }

    /// Get raw data pointer for a specific type
    fn data_ptr<T: ArrayElement>(&self) -> Result<*const T> {
        let ptr = match T::DTYPE {
            DType::Bool => unsafe { mlx_sys::mlx_array_data_bool(self.inner) as *const T },
            DType::UInt8 => unsafe { mlx_sys::mlx_array_data_uint8(self.inner) as *const T },
            DType::UInt16 => unsafe { mlx_sys::mlx_array_data_uint16(self.inner) as *const T },
            DType::UInt32 => unsafe { mlx_sys::mlx_array_data_uint32(self.inner) as *const T },
            DType::UInt64 => unsafe { mlx_sys::mlx_array_data_uint64(self.inner) as *const T },
            DType::Int8 => unsafe { mlx_sys::mlx_array_data_int8(self.inner) as *const T },
            DType::Int16 => unsafe { mlx_sys::mlx_array_data_int16(self.inner) as *const T },
            DType::Int32 => unsafe { mlx_sys::mlx_array_data_int32(self.inner) as *const T },
            DType::Int64 => unsafe { mlx_sys::mlx_array_data_int64(self.inner) as *const T },
            DType::Float32 => unsafe { mlx_sys::mlx_array_data_float32(self.inner) as *const T },
            _ => return Err(Error::InvalidDType(T::DTYPE, "data access".into())),
        };

        if ptr.is_null() {
            return Err(Error::NullPointer("Array data pointer is null".into()));
        }

        Ok(ptr)
    }

    /// Get a scalar value from a 0-dimensional array
    pub fn item<T: ArrayElement>(&self) -> Result<T> {
        if !self.is_scalar() && self.size() != 1 {
            return Err(Error::InvalidShape(format!(
                "Cannot get item from array with shape {:?}",
                self.shape()
            )));
        }

        let vec = self.to_vec::<T>()?;
        Ok(vec[0])
    }

    /// Get a scalar float32 value
    pub fn item_float32(&self) -> Result<f32> {
        self.eval();
        let mut result: f32 = 0.0;
        let status = unsafe { mlx_sys::mlx_array_item_float32(&mut result, self.inner) };
        if status != 0 {
            return Err(Error::ConversionError("Failed to get float32 item".into()));
        }
        Ok(result)
    }

    /// Get a scalar int32 value
    pub fn item_int32(&self) -> Result<i32> {
        self.eval();
        let mut result: i32 = 0;
        let status = unsafe { mlx_sys::mlx_array_item_int32(&mut result, self.inner) };
        if status != 0 {
            return Err(Error::ConversionError("Failed to get int32 item".into()));
        }
        Ok(result)
    }

    /// Get a scalar bool value
    pub fn item_bool(&self) -> Result<bool> {
        self.eval();
        let mut result: bool = false;
        let status = unsafe { mlx_sys::mlx_array_item_bool(&mut result, self.inner) };
        if status != 0 {
            return Err(Error::ConversionError("Failed to get bool item".into()));
        }
        Ok(result)
    }

    // ========================================================================
    // Shape operations
    // ========================================================================

    /// Reshape the array to a new shape
    pub fn reshape(&self, shape: &[i32]) -> Result<Self> {
        let stream = Stream::default();
        let mut result = Self::new_uninit();

        let status = unsafe {
            mlx_sys::mlx_reshape(
                result.as_mut_ptr(),
                self.inner,
                shape.as_ptr(),
                shape.len(),
                stream.as_raw(),
            )
        };

        if status != 0 {
            return Err(Error::InvalidShape(format!(
                "Cannot reshape array to shape {:?}",
                shape
            )));
        }

        Ok(result)
    }

    /// Transpose the array
    pub fn transpose(&self) -> Result<Self> {
        let stream = Stream::default();
        let mut result = Self::new_uninit();

        let status = unsafe {
            mlx_sys::mlx_transpose(
                result.as_mut_ptr(),
                self.inner,
                stream.as_raw(),
            )
        };

        if status != 0 {
            return Err(Error::ArrayCreation("Failed to transpose array".into()));
        }

        Ok(result)
    }

    /// Shorthand for transpose
    pub fn t(&self) -> Result<Self> {
        self.transpose()
    }

    /// Transpose the array along specified axes
    ///
    /// # Arguments
    /// * `axes` - The order of axes to transpose to
    ///
    /// # Example
    /// ```ignore
    /// let a = Array::from_slice(&[1, 2, 3, 4, 5, 6], &[2, 3])?;
    /// let b = a.transpose_axes(&[1, 0])?; // Swap dimensions
    /// ```
    pub fn transpose_axes(&self, axes: &[i32]) -> Result<Self> {
        let stream = Stream::default();
        let mut result = Self::new_uninit();

        let status = unsafe {
            mlx_sys::mlx_transpose_axes(
                result.as_mut_ptr(),
                self.inner,
                axes.as_ptr(),
                axes.len(),
                stream.as_raw(),
            )
        };

        if status != 0 {
            return Err(Error::ArrayCreation("Failed to transpose array with axes".into()));
        }

        Ok(result)
    }

    /// Expand dimensions at the given axis
    pub fn expand_dims(&self, axis: i32) -> Result<Self> {
        let stream = Stream::default();
        let mut result = Self::new_uninit();

        let status = unsafe {
            mlx_sys::mlx_expand_dims(
                result.as_mut_ptr(),
                self.inner,
                axis,
                stream.as_raw(),
            )
        };

        if status != 0 {
            return Err(Error::ArrayCreation("Failed to expand dims".into()));
        }

        Ok(result)
    }

    /// Squeeze dimensions (remove axes with size 1)
    pub fn squeeze(&self) -> Result<Self> {
        let stream = Stream::default();
        let mut result = Self::new_uninit();

        let status = unsafe {
            mlx_sys::mlx_squeeze(result.as_mut_ptr(), self.inner, stream.as_raw())
        };

        if status != 0 {
            return Err(Error::ArrayCreation("Failed to squeeze".into()));
        }

        Ok(result)
    }

    /// Squeeze a specific axis
    pub fn squeeze_axis(&self, axis: i32) -> Result<Self> {
        let stream = Stream::default();
        let mut result = Self::new_uninit();

        let status = unsafe {
            mlx_sys::mlx_squeeze_axis(result.as_mut_ptr(), self.inner, axis, stream.as_raw())
        };

        if status != 0 {
            return Err(Error::ArrayCreation("Failed to squeeze axis".into()));
        }

        Ok(result)
    }

    /// Flatten the array to 1D
    pub fn flatten(&self) -> Result<Self> {
        let stream = Stream::default();
        let mut result = Self::new_uninit();

        let status = unsafe {
            mlx_sys::mlx_flatten(
                result.as_mut_ptr(),
                self.inner,
                0,
                -1,
                stream.as_raw(),
            )
        };

        if status != 0 {
            return Err(Error::ArrayCreation("Failed to flatten".into()));
        }

        Ok(result)
    }

    // ========================================================================
    // Indexing and slicing operations
    // ========================================================================

    /// Slice the array with start, stop, and stride for each dimension
    ///
    /// # Arguments
    /// * `start` - Starting indices for each dimension
    /// * `stop` - Stopping indices for each dimension
    /// * `strides` - Step size for each dimension (optional, defaults to 1)
    pub fn slice(&self, start: &[i32], stop: &[i32], strides: Option<&[i32]>) -> Result<Self> {
        let stream = Stream::default();
        let mut result = Self::new_uninit();

        let default_strides: Vec<i32> = vec![1; start.len()];
        let strides = strides.unwrap_or(&default_strides);

        if start.len() != stop.len() || start.len() != strides.len() {
            return Err(Error::InvalidShape(
                "start, stop, and strides must have the same length".into(),
            ));
        }

        let status = unsafe {
            mlx_sys::mlx_slice(
                result.as_mut_ptr(),
                self.inner,
                start.as_ptr(),
                start.len(),
                stop.as_ptr(),
                stop.len(),
                strides.as_ptr(),
                strides.len(),
                stream.as_raw(),
            )
        };

        if status != 0 {
            return Err(Error::ArrayCreation("Failed to slice array".into()));
        }

        Ok(result)
    }

    /// Take elements from an array using indices
    ///
    /// # Arguments
    /// * `indices` - Array of indices to take
    pub fn take(&self, indices: &Array) -> Result<Self> {
        let stream = Stream::default();
        let mut result = Self::new_uninit();

        let status = unsafe {
            mlx_sys::mlx_take(
                result.as_mut_ptr(),
                self.inner,
                indices.inner,
                stream.as_raw(),
            )
        };

        if status != 0 {
            return Err(Error::ArrayCreation("Failed to take elements".into()));
        }

        Ok(result)
    }

    /// Take elements along an axis using indices
    ///
    /// # Arguments
    /// * `indices` - Array of indices to take
    /// * `axis` - Axis along which to take
    pub fn take_along_axis(&self, indices: &Array, axis: i32) -> Result<Self> {
        let stream = Stream::default();
        let mut result = Self::new_uninit();

        let status = unsafe {
            mlx_sys::mlx_take_along_axis(
                result.as_mut_ptr(),
                self.inner,
                indices.inner,
                axis,
                stream.as_raw(),
            )
        };

        if status != 0 {
            return Err(Error::ArrayCreation("Failed to take along axis".into()));
        }

        Ok(result)
    }

    // ========================================================================
    // Reduction operations
    // ========================================================================

    /// Sum all elements
    pub fn sum_all(&self, keepdims: bool) -> Result<Self> {
        let stream = Stream::default();
        let mut result = Self::new_uninit();

        let status = unsafe {
            mlx_sys::mlx_sum(
                result.as_mut_ptr(),
                self.inner,
                keepdims,
                stream.as_raw(),
            )
        };

        if status != 0 {
            return Err(Error::ArrayCreation("Failed to compute sum".into()));
        }

        Ok(result)
    }

    /// Sum along specified axes
    pub fn sum_axes(&self, axes: &[i32], keepdims: bool) -> Result<Self> {
        let stream = Stream::default();
        let mut result = Self::new_uninit();

        let status = unsafe {
            mlx_sys::mlx_sum_axes(
                result.as_mut_ptr(),
                self.inner,
                axes.as_ptr(),
                axes.len(),
                keepdims,
                stream.as_raw(),
            )
        };

        if status != 0 {
            return Err(Error::ArrayCreation("Failed to compute sum".into()));
        }

        Ok(result)
    }

    /// Mean of all elements
    pub fn mean_all(&self, keepdims: bool) -> Result<Self> {
        let stream = Stream::default();
        let mut result = Self::new_uninit();

        let status = unsafe {
            mlx_sys::mlx_mean(
                result.as_mut_ptr(),
                self.inner,
                keepdims,
                stream.as_raw(),
            )
        };

        if status != 0 {
            return Err(Error::ArrayCreation("Failed to compute mean".into()));
        }

        Ok(result)
    }

    /// Maximum of all elements
    pub fn max_all(&self, keepdims: bool) -> Result<Self> {
        let stream = Stream::default();
        let mut result = Self::new_uninit();

        let status = unsafe {
            mlx_sys::mlx_max(
                result.as_mut_ptr(),
                self.inner,
                keepdims,
                stream.as_raw(),
            )
        };

        if status != 0 {
            return Err(Error::ArrayCreation("Failed to compute max".into()));
        }

        Ok(result)
    }

    /// Minimum of all elements
    pub fn min_all(&self, keepdims: bool) -> Result<Self> {
        let stream = Stream::default();
        let mut result = Self::new_uninit();

        let status = unsafe {
            mlx_sys::mlx_min(
                result.as_mut_ptr(),
                self.inner,
                keepdims,
                stream.as_raw(),
            )
        };

        if status != 0 {
            return Err(Error::ArrayCreation("Failed to compute min".into()));
        }

        Ok(result)
    }

    // ========================================================================
    // Math operations
    // ========================================================================

    /// Element-wise absolute value
    pub fn abs(&self) -> Result<Self> {
        let stream = Stream::default();
        let mut result = Self::new_uninit();

        let status = unsafe {
            mlx_sys::mlx_abs(result.as_mut_ptr(), self.inner, stream.as_raw())
        };

        if status != 0 {
            return Err(Error::ArrayCreation("Failed to compute abs".into()));
        }

        Ok(result)
    }

    /// Element-wise square root
    pub fn sqrt(&self) -> Result<Self> {
        let stream = Stream::default();
        let mut result = Self::new_uninit();

        let status = unsafe {
            mlx_sys::mlx_sqrt(result.as_mut_ptr(), self.inner, stream.as_raw())
        };

        if status != 0 {
            return Err(Error::ArrayCreation("Failed to compute sqrt".into()));
        }

        Ok(result)
    }

    /// Element-wise exponential
    pub fn exp(&self) -> Result<Self> {
        let stream = Stream::default();
        let mut result = Self::new_uninit();

        let status = unsafe {
            mlx_sys::mlx_exp(result.as_mut_ptr(), self.inner, stream.as_raw())
        };

        if status != 0 {
            return Err(Error::ArrayCreation("Failed to compute exp".into()));
        }

        Ok(result)
    }

    /// Element-wise natural logarithm
    pub fn log(&self) -> Result<Self> {
        let stream = Stream::default();
        let mut result = Self::new_uninit();

        let status = unsafe {
            mlx_sys::mlx_log(result.as_mut_ptr(), self.inner, stream.as_raw())
        };

        if status != 0 {
            return Err(Error::ArrayCreation("Failed to compute log".into()));
        }

        Ok(result)
    }

    /// Element-wise sine
    pub fn sin(&self) -> Result<Self> {
        let stream = Stream::default();
        let mut result = Self::new_uninit();

        let status = unsafe {
            mlx_sys::mlx_sin(result.as_mut_ptr(), self.inner, stream.as_raw())
        };

        if status != 0 {
            return Err(Error::ArrayCreation("Failed to compute sin".into()));
        }

        Ok(result)
    }

    /// Element-wise cosine
    pub fn cos(&self) -> Result<Self> {
        let stream = Stream::default();
        let mut result = Self::new_uninit();

        let status = unsafe {
            mlx_sys::mlx_cos(result.as_mut_ptr(), self.inner, stream.as_raw())
        };

        if status != 0 {
            return Err(Error::ArrayCreation("Failed to compute cos".into()));
        }

        Ok(result)
    }

    /// Matrix multiplication
    pub fn matmul(&self, other: &Array) -> Result<Self> {
        let stream = Stream::default();
        let mut result = Self::new_uninit();

        let status = unsafe {
            mlx_sys::mlx_matmul(
                result.as_mut_ptr(),
                self.inner,
                other.inner,
                stream.as_raw(),
            )
        };

        if status != 0 {
            return Err(Error::ArrayCreation("Failed to compute matmul".into()));
        }

        Ok(result)
    }

    // ========================================================================
    // Comparison operations
    // ========================================================================

    /// Element-wise equality comparison
    ///
    /// Returns a boolean array where each element is true if the corresponding
    /// elements in self and other are equal.
    pub fn eq(&self, other: &Array) -> Result<Self> {
        let stream = Stream::default();
        let mut result = Self::new_uninit();

        let status = unsafe {
            mlx_sys::mlx_equal(
                result.as_mut_ptr(),
                self.inner,
                other.inner,
                stream.as_raw(),
            )
        };

        if status != 0 {
            return Err(Error::ArrayCreation("Failed to compute equal".into()));
        }

        Ok(result)
    }

    /// Element-wise inequality comparison
    pub fn ne(&self, other: &Array) -> Result<Self> {
        let stream = Stream::default();
        let mut result = Self::new_uninit();

        let status = unsafe {
            mlx_sys::mlx_not_equal(
                result.as_mut_ptr(),
                self.inner,
                other.inner,
                stream.as_raw(),
            )
        };

        if status != 0 {
            return Err(Error::ArrayCreation("Failed to compute not_equal".into()));
        }

        Ok(result)
    }

    /// Element-wise less than comparison
    pub fn lt(&self, other: &Array) -> Result<Self> {
        let stream = Stream::default();
        let mut result = Self::new_uninit();

        let status = unsafe {
            mlx_sys::mlx_less(
                result.as_mut_ptr(),
                self.inner,
                other.inner,
                stream.as_raw(),
            )
        };

        if status != 0 {
            return Err(Error::ArrayCreation("Failed to compute less".into()));
        }

        Ok(result)
    }

    /// Element-wise less than or equal comparison
    pub fn le(&self, other: &Array) -> Result<Self> {
        let stream = Stream::default();
        let mut result = Self::new_uninit();

        let status = unsafe {
            mlx_sys::mlx_less_equal(
                result.as_mut_ptr(),
                self.inner,
                other.inner,
                stream.as_raw(),
            )
        };

        if status != 0 {
            return Err(Error::ArrayCreation("Failed to compute less_equal".into()));
        }

        Ok(result)
    }

    /// Element-wise greater than comparison
    pub fn gt(&self, other: &Array) -> Result<Self> {
        let stream = Stream::default();
        let mut result = Self::new_uninit();

        let status = unsafe {
            mlx_sys::mlx_greater(
                result.as_mut_ptr(),
                self.inner,
                other.inner,
                stream.as_raw(),
            )
        };

        if status != 0 {
            return Err(Error::ArrayCreation("Failed to compute greater".into()));
        }

        Ok(result)
    }

    /// Element-wise greater than or equal comparison
    pub fn ge(&self, other: &Array) -> Result<Self> {
        let stream = Stream::default();
        let mut result = Self::new_uninit();

        let status = unsafe {
            mlx_sys::mlx_greater_equal(
                result.as_mut_ptr(),
                self.inner,
                other.inner,
                stream.as_raw(),
            )
        };

        if status != 0 {
            return Err(Error::ArrayCreation("Failed to compute greater_equal".into()));
        }

        Ok(result)
    }

    // ========================================================================
    // Logical operations
    // ========================================================================

    /// Element-wise logical NOT
    pub fn logical_not(&self) -> Result<Self> {
        let stream = Stream::default();
        let mut result = Self::new_uninit();

        let status = unsafe {
            mlx_sys::mlx_logical_not(result.as_mut_ptr(), self.inner, stream.as_raw())
        };

        if status != 0 {
            return Err(Error::ArrayCreation("Failed to compute logical_not".into()));
        }

        Ok(result)
    }

    /// Element-wise logical AND
    pub fn logical_and(&self, other: &Array) -> Result<Self> {
        let stream = Stream::default();
        let mut result = Self::new_uninit();

        let status = unsafe {
            mlx_sys::mlx_logical_and(
                result.as_mut_ptr(),
                self.inner,
                other.inner,
                stream.as_raw(),
            )
        };

        if status != 0 {
            return Err(Error::ArrayCreation("Failed to compute logical_and".into()));
        }

        Ok(result)
    }

    /// Element-wise logical OR
    pub fn logical_or(&self, other: &Array) -> Result<Self> {
        let stream = Stream::default();
        let mut result = Self::new_uninit();

        let status = unsafe {
            mlx_sys::mlx_logical_or(
                result.as_mut_ptr(),
                self.inner,
                other.inner,
                stream.as_raw(),
            )
        };

        if status != 0 {
            return Err(Error::ArrayCreation("Failed to compute logical_or".into()));
        }

        Ok(result)
    }

    // ========================================================================
    // Advanced indexing operations
    // ========================================================================

    /// Get the index of the maximum value in the flattened array
    ///
    /// # Arguments
    /// * `keepdims` - If true, the reduced axis is kept with size 1
    ///
    /// # Returns
    /// Array containing the index of the maximum value
    pub fn argmax(&self, keepdims: bool) -> Result<Self> {
        let stream = Stream::default();
        let mut result = Self::new_uninit();

        let status = unsafe {
            mlx_sys::mlx_argmax(
                result.as_mut_ptr(),
                self.inner,
                keepdims,
                stream.as_raw(),
            )
        };

        if status != 0 {
            return Err(Error::ArrayCreation("Failed to compute argmax".into()));
        }

        Ok(result)
    }

    /// Get the indices of the maximum values along an axis
    ///
    /// # Arguments
    /// * `axis` - Axis along which to find the maximum
    /// * `keepdims` - If true, the reduced axis is kept with size 1
    ///
    /// # Returns
    /// Array containing the indices of maximum values
    pub fn argmax_axis(&self, axis: i32, keepdims: bool) -> Result<Self> {
        let stream = Stream::default();
        let mut result = Self::new_uninit();

        let status = unsafe {
            mlx_sys::mlx_argmax_axis(
                result.as_mut_ptr(),
                self.inner,
                axis,
                keepdims,
                stream.as_raw(),
            )
        };

        if status != 0 {
            return Err(Error::ArrayCreation("Failed to compute argmax_axis".into()));
        }

        Ok(result)
    }

    /// Get the index of the minimum value in the flattened array
    ///
    /// # Arguments
    /// * `keepdims` - If true, the reduced axis is kept with size 1
    ///
    /// # Returns
    /// Array containing the index of the minimum value
    pub fn argmin(&self, keepdims: bool) -> Result<Self> {
        let stream = Stream::default();
        let mut result = Self::new_uninit();

        let status = unsafe {
            mlx_sys::mlx_argmin(
                result.as_mut_ptr(),
                self.inner,
                keepdims,
                stream.as_raw(),
            )
        };

        if status != 0 {
            return Err(Error::ArrayCreation("Failed to compute argmin".into()));
        }

        Ok(result)
    }

    /// Get the indices of the minimum values along an axis
    ///
    /// # Arguments
    /// * `axis` - Axis along which to find the minimum
    /// * `keepdims` - If true, the reduced axis is kept with size 1
    ///
    /// # Returns
    /// Array containing the indices of minimum values
    pub fn argmin_axis(&self, axis: i32, keepdims: bool) -> Result<Self> {
        let stream = Stream::default();
        let mut result = Self::new_uninit();

        let status = unsafe {
            mlx_sys::mlx_argmin_axis(
                result.as_mut_ptr(),
                self.inner,
                axis,
                keepdims,
                stream.as_raw(),
            )
        };

        if status != 0 {
            return Err(Error::ArrayCreation("Failed to compute argmin_axis".into()));
        }

        Ok(result)
    }

    /// Get the indices that would sort the flattened array
    ///
    /// # Returns
    /// Array of indices that sort the array
    pub fn argsort(&self) -> Result<Self> {
        let stream = Stream::default();
        let mut result = Self::new_uninit();

        let status = unsafe {
            mlx_sys::mlx_argsort(result.as_mut_ptr(), self.inner, stream.as_raw())
        };

        if status != 0 {
            return Err(Error::ArrayCreation("Failed to compute argsort".into()));
        }

        Ok(result)
    }

    /// Get the indices that would sort along an axis
    ///
    /// # Arguments
    /// * `axis` - Axis along which to sort
    ///
    /// # Returns
    /// Array of indices that sort along the axis
    pub fn argsort_axis(&self, axis: i32) -> Result<Self> {
        let stream = Stream::default();
        let mut result = Self::new_uninit();

        let status = unsafe {
            mlx_sys::mlx_argsort_axis(
                result.as_mut_ptr(),
                self.inner,
                axis,
                stream.as_raw(),
            )
        };

        if status != 0 {
            return Err(Error::ArrayCreation("Failed to compute argsort_axis".into()));
        }

        Ok(result)
    }

    /// Sort the flattened array
    ///
    /// # Returns
    /// Sorted array
    pub fn sort(&self) -> Result<Self> {
        let stream = Stream::default();
        let mut result = Self::new_uninit();

        let status = unsafe {
            mlx_sys::mlx_sort(result.as_mut_ptr(), self.inner, stream.as_raw())
        };

        if status != 0 {
            return Err(Error::ArrayCreation("Failed to sort".into()));
        }

        Ok(result)
    }

    /// Sort along an axis
    ///
    /// # Arguments
    /// * `axis` - Axis along which to sort
    ///
    /// # Returns
    /// Sorted array
    pub fn sort_axis(&self, axis: i32) -> Result<Self> {
        let stream = Stream::default();
        let mut result = Self::new_uninit();

        let status = unsafe {
            mlx_sys::mlx_sort_axis(
                result.as_mut_ptr(),
                self.inner,
                axis,
                stream.as_raw(),
            )
        };

        if status != 0 {
            return Err(Error::ArrayCreation("Failed to sort_axis".into()));
        }

        Ok(result)
    }

    /// Get the top k largest elements from the flattened array
    ///
    /// # Arguments
    /// * `k` - Number of top elements to return
    ///
    /// # Returns
    /// Array containing the k largest elements (not sorted)
    pub fn topk(&self, k: i32) -> Result<Self> {
        let stream = Stream::default();
        let mut result = Self::new_uninit();

        let status = unsafe {
            mlx_sys::mlx_topk(result.as_mut_ptr(), self.inner, k, stream.as_raw())
        };

        if status != 0 {
            return Err(Error::ArrayCreation("Failed to compute topk".into()));
        }

        Ok(result)
    }

    /// Get the top k largest elements along an axis
    ///
    /// # Arguments
    /// * `k` - Number of top elements to return
    /// * `axis` - Axis along which to find top k
    ///
    /// # Returns
    /// Array containing the k largest elements along the axis (not sorted)
    pub fn topk_axis(&self, k: i32, axis: i32) -> Result<Self> {
        let stream = Stream::default();
        let mut result = Self::new_uninit();

        let status = unsafe {
            mlx_sys::mlx_topk_axis(
                result.as_mut_ptr(),
                self.inner,
                k,
                axis,
                stream.as_raw(),
            )
        };

        if status != 0 {
            return Err(Error::ArrayCreation("Failed to compute topk_axis".into()));
        }

        Ok(result)
    }
}

// ============================================================================
// Memory management
// ============================================================================

impl Drop for Array {
    fn drop(&mut self) {
        unsafe {
            mlx_sys::mlx_array_free(self.inner);
        }
    }
}

impl Clone for Array {
    fn clone(&self) -> Self {
        let mut result = Self::new_uninit();
        unsafe {
            mlx_sys::mlx_array_set(result.as_mut_ptr(), self.inner);
        }
        result
    }
}

// Array is Send and Sync because MLX handles thread safety
unsafe impl Send for Array {}
unsafe impl Sync for Array {}

// ============================================================================
// Display
// ============================================================================

impl fmt::Debug for Array {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.debug_struct("Array")
            .field("shape", &self.shape())
            .field("dtype", &self.dtype())
            .finish()
    }
}

impl fmt::Display for Array {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "Array(shape={:?}, dtype={})", self.shape(), self.dtype())
    }
}

// ============================================================================
// Arithmetic operators
// ============================================================================

/// Helper function for binary operations
fn binary_op(
    a: &Array,
    b: &Array,
    op: unsafe extern "C" fn(
        *mut mlx_sys::mlx_array,
        mlx_sys::mlx_array,
        mlx_sys::mlx_array,
        mlx_sys::mlx_stream,
    ) -> i32,
) -> Array {
    let stream = Stream::default();
    let mut result = Array::new_uninit();
    unsafe {
        op(result.as_mut_ptr(), a.as_raw(), b.as_raw(), stream.as_raw());
    }
    result
}

impl Add for &Array {
    type Output = Array;

    fn add(self, rhs: Self) -> Self::Output {
        binary_op(self, rhs, mlx_sys::mlx_add)
    }
}

impl Add<Array> for &Array {
    type Output = Array;

    fn add(self, rhs: Array) -> Self::Output {
        self + &rhs
    }
}

impl Add<&Array> for Array {
    type Output = Array;

    fn add(self, rhs: &Array) -> Self::Output {
        &self + rhs
    }
}

impl Add for Array {
    type Output = Array;

    fn add(self, rhs: Self) -> Self::Output {
        &self + &rhs
    }
}

impl Sub for &Array {
    type Output = Array;

    fn sub(self, rhs: Self) -> Self::Output {
        binary_op(self, rhs, mlx_sys::mlx_subtract)
    }
}

impl Sub<Array> for &Array {
    type Output = Array;

    fn sub(self, rhs: Array) -> Self::Output {
        self - &rhs
    }
}

impl Sub<&Array> for Array {
    type Output = Array;

    fn sub(self, rhs: &Array) -> Self::Output {
        &self - rhs
    }
}

impl Sub for Array {
    type Output = Array;

    fn sub(self, rhs: Self) -> Self::Output {
        &self - &rhs
    }
}

impl Mul for &Array {
    type Output = Array;

    fn mul(self, rhs: Self) -> Self::Output {
        binary_op(self, rhs, mlx_sys::mlx_multiply)
    }
}

impl Mul<Array> for &Array {
    type Output = Array;

    fn mul(self, rhs: Array) -> Self::Output {
        self * &rhs
    }
}

impl Mul<&Array> for Array {
    type Output = Array;

    fn mul(self, rhs: &Array) -> Self::Output {
        &self * rhs
    }
}

impl Mul for Array {
    type Output = Array;

    fn mul(self, rhs: Self) -> Self::Output {
        &self * &rhs
    }
}

impl Div for &Array {
    type Output = Array;

    fn div(self, rhs: Self) -> Self::Output {
        binary_op(self, rhs, mlx_sys::mlx_divide)
    }
}

impl Div<Array> for &Array {
    type Output = Array;

    fn div(self, rhs: Array) -> Self::Output {
        self / &rhs
    }
}

impl Div<&Array> for Array {
    type Output = Array;

    fn div(self, rhs: &Array) -> Self::Output {
        &self / rhs
    }
}

impl Div for Array {
    type Output = Array;

    fn div(self, rhs: Self) -> Self::Output {
        &self / &rhs
    }
}

impl Neg for &Array {
    type Output = Array;

    fn neg(self) -> Self::Output {
        let stream = Stream::default();
        let mut result = Array::new_uninit();
        unsafe {
            mlx_sys::mlx_negative(result.as_mut_ptr(), self.inner, stream.as_raw());
        }
        result
    }
}

impl Neg for Array {
    type Output = Array;

    fn neg(self) -> Self::Output {
        -&self
    }
}

// ============================================================================
// From implementations
// ============================================================================

impl From<f32> for Array {
    fn from(val: f32) -> Self {
        Array::from_float(val)
    }
}

impl From<i32> for Array {
    fn from(val: i32) -> Self {
        Array::from_int(val)
    }
}

impl From<bool> for Array {
    fn from(val: bool) -> Self {
        Array::from_bool(val)
    }
}
