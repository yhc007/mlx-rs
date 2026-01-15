//! Additional operations for MLX arrays

use mlx_sys;

use crate::array::Array;
use crate::dtype::ArrayElement;
use crate::error::{Error, Result};
use crate::stream::Stream;

// ============================================================================
// Array creation functions
// ============================================================================

/// Create an array filled with zeros
pub fn zeros<T: ArrayElement>(shape: &[i32]) -> Result<Array> {
    Array::zeros::<T>(shape)
}

/// Create an array filled with ones
pub fn ones<T: ArrayElement>(shape: &[i32]) -> Result<Array> {
    Array::ones::<T>(shape)
}

/// Create an identity matrix
pub fn eye<T: ArrayElement>(n: i32) -> Result<Array> {
    Array::eye::<T>(n, None, 0)
}

/// Create a 1D array with evenly spaced values
pub fn arange<T: ArrayElement>(start: f64, stop: f64, step: f64) -> Result<Array> {
    Array::arange::<T>(start, stop, step)
}

/// Create a 1D array with evenly spaced values (inclusive)
pub fn linspace<T: ArrayElement>(start: f64, stop: f64, num: i32) -> Result<Array> {
    Array::linspace::<T>(start, stop, num)
}

// ============================================================================
// Arithmetic operations
// ============================================================================

/// Element-wise addition
pub fn add(a: &Array, b: &Array) -> Array {
    a + b
}

/// Element-wise subtraction
pub fn subtract(a: &Array, b: &Array) -> Array {
    a - b
}

/// Element-wise multiplication
pub fn multiply(a: &Array, b: &Array) -> Array {
    a * b
}

/// Element-wise division
pub fn divide(a: &Array, b: &Array) -> Array {
    a / b
}

/// Element-wise negation
pub fn negative(a: &Array) -> Array {
    -a
}

// ============================================================================
// Mathematical functions
// ============================================================================

/// Element-wise absolute value
pub fn abs(a: &Array) -> Result<Array> {
    a.abs()
}

/// Element-wise square root
pub fn sqrt(a: &Array) -> Result<Array> {
    a.sqrt()
}

/// Element-wise exponential
pub fn exp(a: &Array) -> Result<Array> {
    a.exp()
}

/// Element-wise natural logarithm
pub fn log(a: &Array) -> Result<Array> {
    a.log()
}

/// Element-wise sine
pub fn sin(a: &Array) -> Result<Array> {
    a.sin()
}

/// Element-wise cosine
pub fn cos(a: &Array) -> Result<Array> {
    a.cos()
}

// ============================================================================
// Linear algebra
// ============================================================================

/// Matrix multiplication
pub fn matmul(a: &Array, b: &Array) -> Result<Array> {
    a.matmul(b)
}

/// Transpose
pub fn transpose(a: &Array) -> Result<Array> {
    a.transpose()
}

// ============================================================================
// Reduction operations
// ============================================================================

/// Sum of all elements
pub fn sum(a: &Array) -> Result<Array> {
    a.sum_all(false)
}

/// Mean of all elements
pub fn mean(a: &Array) -> Result<Array> {
    a.mean_all(false)
}

/// Maximum of all elements
pub fn max(a: &Array) -> Result<Array> {
    a.max_all(false)
}

/// Minimum of all elements
pub fn min(a: &Array) -> Result<Array> {
    a.min_all(false)
}

// ============================================================================
// Shape operations
// ============================================================================

/// Reshape an array
pub fn reshape(a: &Array, shape: &[i32]) -> Result<Array> {
    a.reshape(shape)
}

/// Expand dimensions at the given axis
pub fn expand_dims(a: &Array, axis: i32) -> Result<Array> {
    a.expand_dims(axis)
}

// ============================================================================
// Evaluation
// ============================================================================

/// Evaluate an array
pub fn eval(a: &Array) {
    a.eval();
}

// ============================================================================
// Concatenation and stacking
// ============================================================================

/// Concatenate arrays along an axis
///
/// # Arguments
/// * `arrays` - Slice of arrays to concatenate
/// * `axis` - Axis along which to concatenate
pub fn concatenate(arrays: &[&Array], axis: i32) -> Result<Array> {
    if arrays.is_empty() {
        return Err(Error::InvalidShape("Cannot concatenate empty array list".into()));
    }

    let stream = Stream::default();
    let mut result = Array::new_uninit();

    // Create vector of raw array pointers
    let raw_arrays: Vec<_> = arrays.iter().map(|a| a.as_raw()).collect();
    let vec = unsafe { mlx_sys::mlx_vector_array_new_data(raw_arrays.as_ptr(), raw_arrays.len()) };

    let status = unsafe {
        mlx_sys::mlx_concatenate_axis(
            result.as_mut_ptr(),
            vec,
            axis,
            stream.as_raw(),
        )
    };

    unsafe { mlx_sys::mlx_vector_array_free(vec) };

    if status != 0 {
        return Err(Error::ArrayCreation("Failed to concatenate arrays".into()));
    }

    Ok(result)
}

/// Stack arrays along a new axis
///
/// # Arguments
/// * `arrays` - Slice of arrays to stack
/// * `axis` - Axis along which to stack (new axis is created)
pub fn stack(arrays: &[&Array], axis: i32) -> Result<Array> {
    if arrays.is_empty() {
        return Err(Error::InvalidShape("Cannot stack empty array list".into()));
    }

    let stream = Stream::default();
    let mut result = Array::new_uninit();

    // Create vector of raw array pointers
    let raw_arrays: Vec<_> = arrays.iter().map(|a| a.as_raw()).collect();
    let vec = unsafe { mlx_sys::mlx_vector_array_new_data(raw_arrays.as_ptr(), raw_arrays.len()) };

    let status = unsafe {
        mlx_sys::mlx_stack_axis(
            result.as_mut_ptr(),
            vec,
            axis,
            stream.as_raw(),
        )
    };

    unsafe { mlx_sys::mlx_vector_array_free(vec) };

    if status != 0 {
        return Err(Error::ArrayCreation("Failed to stack arrays".into()));
    }

    Ok(result)
}

/// Split an array into multiple sub-arrays
///
/// # Arguments
/// * `array` - Array to split
/// * `num_splits` - Number of equal splits
/// * `axis` - Axis along which to split
pub fn split(array: &Array, num_splits: i32, axis: i32) -> Result<Vec<Array>> {
    let stream = Stream::default();
    let mut result_vec = unsafe { mlx_sys::mlx_vector_array_new() };

    let status = unsafe {
        mlx_sys::mlx_split(
            &mut result_vec,
            array.as_raw(),
            num_splits,
            axis,
            stream.as_raw(),
        )
    };

    if status != 0 {
        unsafe { mlx_sys::mlx_vector_array_free(result_vec) };
        return Err(Error::ArrayCreation("Failed to split array".into()));
    }

    // Extract arrays from the vector
    let size = unsafe { mlx_sys::mlx_vector_array_size(result_vec) };
    let mut arrays = Vec::with_capacity(size);

    for i in 0..size {
        let mut arr = Array::new_uninit();
        unsafe {
            mlx_sys::mlx_vector_array_get(arr.as_mut_ptr(), result_vec, i);
        }
        arrays.push(arr);
    }

    unsafe { mlx_sys::mlx_vector_array_free(result_vec) };

    Ok(arrays)
}

// ============================================================================
// Indexing operations
// ============================================================================

/// Slice an array
///
/// # Arguments
/// * `array` - Array to slice
/// * `start` - Starting indices for each dimension
/// * `stop` - Stopping indices for each dimension
/// * `strides` - Step size for each dimension (optional)
pub fn slice(array: &Array, start: &[i32], stop: &[i32], strides: Option<&[i32]>) -> Result<Array> {
    array.slice(start, stop, strides)
}

/// Take elements from an array using indices
pub fn take(array: &Array, indices: &Array) -> Result<Array> {
    array.take(indices)
}

/// Take elements along an axis using indices
pub fn take_along_axis(array: &Array, indices: &Array, axis: i32) -> Result<Array> {
    array.take_along_axis(indices, axis)
}

// ============================================================================
// Additional shape operations
// ============================================================================

/// Squeeze dimensions (remove axes with size 1)
pub fn squeeze(a: &Array) -> Result<Array> {
    a.squeeze()
}

/// Flatten an array to 1D
pub fn flatten(a: &Array) -> Result<Array> {
    a.flatten()
}

// ============================================================================
// Comparison operations
// ============================================================================

/// Element-wise equality comparison
pub fn equal(a: &Array, b: &Array) -> Result<Array> {
    a.eq(b)
}

/// Element-wise inequality comparison
pub fn not_equal(a: &Array, b: &Array) -> Result<Array> {
    a.ne(b)
}

/// Element-wise less than comparison
pub fn less(a: &Array, b: &Array) -> Result<Array> {
    a.lt(b)
}

/// Element-wise less than or equal comparison
pub fn less_equal(a: &Array, b: &Array) -> Result<Array> {
    a.le(b)
}

/// Element-wise greater than comparison
pub fn greater(a: &Array, b: &Array) -> Result<Array> {
    a.gt(b)
}

/// Element-wise greater than or equal comparison
pub fn greater_equal(a: &Array, b: &Array) -> Result<Array> {
    a.ge(b)
}

// ============================================================================
// Logical operations
// ============================================================================

/// Element-wise logical NOT
pub fn logical_not(a: &Array) -> Result<Array> {
    a.logical_not()
}

/// Element-wise logical AND
pub fn logical_and(a: &Array, b: &Array) -> Result<Array> {
    a.logical_and(b)
}

/// Element-wise logical OR
pub fn logical_or(a: &Array, b: &Array) -> Result<Array> {
    a.logical_or(b)
}

/// Select elements from x or y based on condition
///
/// # Arguments
/// * `condition` - Boolean array for selection
/// * `x` - Values to select when condition is true
/// * `y` - Values to select when condition is false
///
/// # Returns
/// Array with elements from x where condition is true, else from y
pub fn where_cond(condition: &Array, x: &Array, y: &Array) -> Result<Array> {
    let stream = Stream::default();
    let mut result = Array::new_uninit();

    let status = unsafe {
        mlx_sys::mlx_where(
            result.as_mut_ptr(),
            condition.as_raw(),
            x.as_raw(),
            y.as_raw(),
            stream.as_raw(),
        )
    };

    if status != 0 {
        return Err(Error::ArrayCreation("Failed to compute where".into()));
    }

    Ok(result)
}

// ============================================================================
// Advanced indexing operations
// ============================================================================

/// Get the index of the maximum value in the flattened array
pub fn argmax(a: &Array) -> Result<Array> {
    a.argmax(false)
}

/// Get the indices of the maximum values along an axis
pub fn argmax_axis(a: &Array, axis: i32) -> Result<Array> {
    a.argmax_axis(axis, false)
}

/// Get the index of the minimum value in the flattened array
pub fn argmin(a: &Array) -> Result<Array> {
    a.argmin(false)
}

/// Get the indices of the minimum values along an axis
pub fn argmin_axis(a: &Array, axis: i32) -> Result<Array> {
    a.argmin_axis(axis, false)
}

/// Get the indices that would sort the flattened array
pub fn argsort(a: &Array) -> Result<Array> {
    a.argsort()
}

/// Get the indices that would sort along an axis
pub fn argsort_axis(a: &Array, axis: i32) -> Result<Array> {
    a.argsort_axis(axis)
}

/// Sort the flattened array
pub fn sort(a: &Array) -> Result<Array> {
    a.sort()
}

/// Sort along an axis
pub fn sort_axis(a: &Array, axis: i32) -> Result<Array> {
    a.sort_axis(axis)
}

/// Get the top k largest elements from the flattened array
pub fn topk(a: &Array, k: i32) -> Result<Array> {
    a.topk(k)
}

/// Get the top k largest elements along an axis
pub fn topk_axis(a: &Array, k: i32, axis: i32) -> Result<Array> {
    a.topk_axis(k, axis)
}
