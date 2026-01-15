//! Automatic differentiation and function transforms
//!
//! This module provides gradient computation and function transformations.
//!
//! # Overview
//!
//! MLX supports automatic differentiation through several key functions:
//!
//! - [`grad`] - Compute gradients of scalar-valued functions
//! - [`value_and_grad`] - Compute both function value and gradients
//! - [`vjp`] - Vector-Jacobian product (reverse-mode autodiff)
//! - [`jvp`] - Jacobian-vector product (forward-mode autodiff)
//!
//! # Example
//!
//! ```ignore
//! use mlx_rs::{Array, transforms};
//!
//! // Define a simple function: f(x) = x^2
//! let x = Array::from_float(3.0);
//!
//! // Compute gradient using vjp
//! let (values, grads) = transforms::vjp(
//!     |inputs| {
//!         let x = &inputs[0];
//!         vec![x * x]
//!     },
//!     &[x.clone()],
//!     &[Array::from_float(1.0)],  // cotangent (seed)
//! ).unwrap();
//!
//! // grads[0] should be 2*x = 6.0
//! ```

use mlx_sys;
use std::ffi::c_void;

use crate::array::Array;
use crate::error::{Error, Result};
use crate::stream::Stream;

// ============================================================================
// Evaluation
// ============================================================================

/// Evaluate multiple arrays at once
pub fn eval(arrays: &[&Array]) {
    if arrays.is_empty() {
        return;
    }

    let raw_arrays: Vec<_> = arrays.iter().map(|a| a.as_raw()).collect();
    let vec = unsafe { mlx_sys::mlx_vector_array_new_data(raw_arrays.as_ptr(), raw_arrays.len()) };

    unsafe {
        mlx_sys::mlx_eval(vec);
        mlx_sys::mlx_vector_array_free(vec);
    }
}

// ============================================================================
// Utility transforms
// ============================================================================

/// Stop gradient - returns a new array that doesn't track gradients
pub fn stop_gradient(x: &Array) -> Result<Array> {
    let stream = Stream::default();
    let mut result = Array::new_uninit();

    let status = unsafe {
        mlx_sys::mlx_stop_gradient(result.as_mut_ptr(), x.as_raw(), stream.as_raw())
    };

    if status != 0 {
        return Err(Error::ArrayCreation("Failed to stop gradient".into()));
    }

    Ok(result)
}

// ============================================================================
// Async evaluation
// ============================================================================

/// Asynchronously evaluate arrays
pub fn async_eval(arrays: &[&Array]) {
    // In MLX, eval is already asynchronous by default
    // This function is provided for API compatibility
    eval(arrays);
}

/// Wait for all pending computations to complete
pub fn synchronize() {
    crate::synchronize();
}

// ============================================================================
// Gradient computation helpers
// ============================================================================

/// Compute numerical gradient for testing (finite differences)
pub fn numerical_gradient(
    f: impl Fn(&Array) -> Result<Array>,
    x: &Array,
    eps: f32,
) -> Result<Array> {
    let eps_arr = Array::from_float(eps);

    // f(x + eps)
    let x_plus = x + &eps_arr;
    let y_plus = f(&x_plus)?;

    // f(x - eps)
    let x_minus = x - &eps_arr;
    let y_minus = f(&x_minus)?;

    // (f(x + eps) - f(x - eps)) / (2 * eps)
    let diff = &y_plus - &y_minus;
    let two_eps = Array::from_float(2.0 * eps);
    let grad = &diff / &two_eps;

    Ok(grad)
}

/// Simple gradient computation for scalar functions using finite differences
///
/// For a scalar-valued function f, computes df/dx
pub fn grad_scalar(f: impl Fn(&Array) -> Result<Array>, x: &Array) -> Result<Array> {
    // Use numerical gradient as a fallback
    // A full implementation would use MLX's autodiff
    numerical_gradient(f, x, 1e-5)
}

// ============================================================================
// JIT Compilation (placeholder)
// ============================================================================

/// Check if compilation is enabled
pub fn compile_enabled() -> bool {
    true
}

// ============================================================================
// Closure types for auto-diff
// ============================================================================

/// Type alias for functions that transform arrays
pub type ArrayFn = Box<dyn Fn(&[Array]) -> Vec<Array> + Send + Sync>;

/// Internal wrapper for passing Rust closures to C
struct ClosureWrapper {
    func: ArrayFn,
}

/// Trampoline function that bridges C callbacks to Rust closures
unsafe extern "C" fn closure_trampoline(
    result: *mut mlx_sys::mlx_vector_array,
    input: mlx_sys::mlx_vector_array,
    payload: *mut c_void,
) -> std::ffi::c_int {
    // Get the closure from payload
    let wrapper = &*(payload as *const ClosureWrapper);

    // Extract input arrays
    let input_size = mlx_sys::mlx_vector_array_size(input);
    let mut inputs = Vec::with_capacity(input_size);
    for i in 0..input_size {
        let mut arr = Array::new_uninit();
        mlx_sys::mlx_vector_array_get(arr.as_mut_ptr(), input, i);
        inputs.push(arr);
    }

    // Call the Rust function
    let outputs = (wrapper.func)(&inputs);

    // Create output vector
    let raw_outputs: Vec<_> = outputs.iter().map(|a| a.as_raw()).collect();
    let output_vec = mlx_sys::mlx_vector_array_new_data(raw_outputs.as_ptr(), raw_outputs.len());
    mlx_sys::mlx_vector_array_set(result, output_vec);
    mlx_sys::mlx_vector_array_free(output_vec);

    // Prevent outputs from being freed (ownership transferred)
    for output in outputs {
        std::mem::forget(output);
    }

    0 // Success
}

/// Destructor for the closure wrapper - called by MLX when closure is freed
unsafe extern "C" fn closure_destructor(payload: *mut c_void) {
    if !payload.is_null() {
        // Reconstruct the Box and drop it
        let _ = Box::from_raw(payload as *mut ClosureWrapper);
    }
}

/// Create an MLX closure from a Rust function
fn create_closure<F>(f: F) -> mlx_sys::mlx_closure
where
    F: Fn(&[Array]) -> Vec<Array> + Send + Sync + 'static,
{
    let wrapper = Box::new(ClosureWrapper {
        func: Box::new(f),
    });
    let wrapper_ptr = Box::into_raw(wrapper);

    unsafe {
        mlx_sys::mlx_closure_new_func_payload(
            Some(closure_trampoline),
            wrapper_ptr as *mut c_void,
            Some(closure_destructor),
        )
    }
}

// ============================================================================
// Vector-Jacobian Product (VJP) - Reverse-mode autodiff
// ============================================================================

/// Compute the vector-Jacobian product (reverse-mode autodiff)
///
/// Given a function `f: R^n -> R^m`, the VJP computes:
/// - The function output `f(primals)`
/// - The gradients `v^T * J` where `J` is the Jacobian and `v` is the cotangent
///
/// # Arguments
/// * `f` - Function to differentiate
/// * `primals` - Input values at which to compute the VJP
/// * `cotangents` - Cotangent vectors (seeds for reverse-mode)
///
/// # Returns
/// A tuple of (outputs, gradients) where:
/// - outputs: The function values `f(primals)`
/// - gradients: The VJP result `v^T * J`
///
/// # Example
/// ```ignore
/// // Compute gradient of f(x) = x^2 at x = 3
/// let x = Array::from_float(3.0);
/// let (outputs, grads) = vjp(
///     |inputs| vec![&inputs[0] * &inputs[0]],
///     &[x],
///     &[Array::from_float(1.0)],
/// ).unwrap();
/// // grads[0] = 2 * 3 = 6
/// ```
pub fn vjp<F>(f: F, primals: &[Array], cotangents: &[Array]) -> Result<(Vec<Array>, Vec<Array>)>
where
    F: Fn(&[Array]) -> Vec<Array> + Send + Sync + 'static,
{
    let closure = create_closure(f);

    // Create input vectors
    let primals_raw: Vec<_> = primals.iter().map(|a| a.as_raw()).collect();
    let primals_vec = unsafe {
        mlx_sys::mlx_vector_array_new_data(primals_raw.as_ptr(), primals_raw.len())
    };

    let cotangents_raw: Vec<_> = cotangents.iter().map(|a| a.as_raw()).collect();
    let cotangents_vec = unsafe {
        mlx_sys::mlx_vector_array_new_data(cotangents_raw.as_ptr(), cotangents_raw.len())
    };

    // Result vectors
    let mut outputs_vec = unsafe { mlx_sys::mlx_vector_array_new() };
    let mut grads_vec = unsafe { mlx_sys::mlx_vector_array_new() };

    // Call VJP
    let status = unsafe {
        mlx_sys::mlx_vjp(
            &mut outputs_vec,
            &mut grads_vec,
            closure,
            primals_vec,
            cotangents_vec,
        )
    };

    // Cleanup input vectors and closure (destructor handles wrapper cleanup)
    unsafe {
        mlx_sys::mlx_vector_array_free(primals_vec);
        mlx_sys::mlx_vector_array_free(cotangents_vec);
        mlx_sys::mlx_closure_free(closure);
    }

    if status != 0 {
        unsafe {
            mlx_sys::mlx_vector_array_free(outputs_vec);
            mlx_sys::mlx_vector_array_free(grads_vec);
        }
        return Err(Error::ArrayCreation("VJP computation failed".into()));
    }

    // Extract results
    let outputs = extract_vector_array(outputs_vec);
    let grads = extract_vector_array(grads_vec);

    unsafe {
        mlx_sys::mlx_vector_array_free(outputs_vec);
        mlx_sys::mlx_vector_array_free(grads_vec);
    }

    Ok((outputs, grads))
}

// ============================================================================
// Jacobian-Vector Product (JVP) - Forward-mode autodiff
// ============================================================================

/// Compute the Jacobian-vector product (forward-mode autodiff)
///
/// Given a function `f: R^n -> R^m`, the JVP computes:
/// - The function output `f(primals)`
/// - The directional derivative `J * v` where `J` is the Jacobian and `v` is the tangent
///
/// # Arguments
/// * `f` - Function to differentiate
/// * `primals` - Input values at which to compute the JVP
/// * `tangents` - Tangent vectors (directions for differentiation)
///
/// # Returns
/// A tuple of (outputs, jvp_result) where:
/// - outputs: The function values `f(primals)`
/// - jvp_result: The directional derivatives `J * v`
///
/// # Example
/// ```ignore
/// // Compute directional derivative of f(x) = x^2 at x = 3 in direction v = 1
/// let x = Array::from_float(3.0);
/// let (outputs, jvps) = jvp(
///     |inputs| vec![&inputs[0] * &inputs[0]],
///     &[x],
///     &[Array::from_float(1.0)],
/// ).unwrap();
/// // jvps[0] = 2 * 3 * 1 = 6
/// ```
pub fn jvp<F>(f: F, primals: &[Array], tangents: &[Array]) -> Result<(Vec<Array>, Vec<Array>)>
where
    F: Fn(&[Array]) -> Vec<Array> + Send + Sync + 'static,
{
    let closure = create_closure(f);

    // Create input vectors
    let primals_raw: Vec<_> = primals.iter().map(|a| a.as_raw()).collect();
    let primals_vec = unsafe {
        mlx_sys::mlx_vector_array_new_data(primals_raw.as_ptr(), primals_raw.len())
    };

    let tangents_raw: Vec<_> = tangents.iter().map(|a| a.as_raw()).collect();
    let tangents_vec = unsafe {
        mlx_sys::mlx_vector_array_new_data(tangents_raw.as_ptr(), tangents_raw.len())
    };

    // Result vectors
    let mut outputs_vec = unsafe { mlx_sys::mlx_vector_array_new() };
    let mut jvps_vec = unsafe { mlx_sys::mlx_vector_array_new() };

    // Call JVP
    let status = unsafe {
        mlx_sys::mlx_jvp(
            &mut outputs_vec,
            &mut jvps_vec,
            closure,
            primals_vec,
            tangents_vec,
        )
    };

    // Cleanup (destructor handles wrapper cleanup)
    unsafe {
        mlx_sys::mlx_vector_array_free(primals_vec);
        mlx_sys::mlx_vector_array_free(tangents_vec);
        mlx_sys::mlx_closure_free(closure);
    }

    if status != 0 {
        unsafe {
            mlx_sys::mlx_vector_array_free(outputs_vec);
            mlx_sys::mlx_vector_array_free(jvps_vec);
        }
        return Err(Error::ArrayCreation("JVP computation failed".into()));
    }

    // Extract results
    let outputs = extract_vector_array(outputs_vec);
    let jvps = extract_vector_array(jvps_vec);

    unsafe {
        mlx_sys::mlx_vector_array_free(outputs_vec);
        mlx_sys::mlx_vector_array_free(jvps_vec);
    }

    Ok((outputs, jvps))
}

// ============================================================================
// Value and Gradient
// ============================================================================

/// Compute both the value and gradient of a scalar-valued function
///
/// This is a convenience wrapper around `vjp` for scalar functions.
///
/// # Arguments
/// * `f` - Scalar-valued function (returns a single array)
/// * `inputs` - Input arrays
/// * `argnums` - Indices of arguments to differentiate with respect to
///
/// # Returns
/// A tuple of (value, gradients) where:
/// - value: The function output
/// - gradients: Gradients with respect to the specified arguments
///
/// # Example
/// ```ignore
/// // f(x, y) = x * y, compute gradient w.r.t. x at (2, 3)
/// let x = Array::from_float(2.0);
/// let y = Array::from_float(3.0);
/// let (value, grads) = value_and_grad(
///     |inputs| vec![&inputs[0] * &inputs[1]],
///     &[x, y],
///     &[0],  // differentiate w.r.t. first argument
/// ).unwrap();
/// // value = 6.0, grads[0] = 3.0 (derivative w.r.t. x is y)
/// ```
pub fn value_and_grad<F>(
    f: F,
    inputs: &[Array],
    argnums: &[i32],
) -> Result<(Vec<Array>, Vec<Array>)>
where
    F: Fn(&[Array]) -> Vec<Array> + Send + Sync + 'static,
{
    let closure = create_closure(f);

    // Create the value_and_grad closure
    let mut vag_closure = unsafe { mlx_sys::mlx_closure_value_and_grad_new() };

    let status = unsafe {
        mlx_sys::mlx_value_and_grad(
            &mut vag_closure,
            closure,
            argnums.as_ptr(),
            argnums.len(),
        )
    };

    if status != 0 {
        unsafe {
            mlx_sys::mlx_closure_free(closure);
            mlx_sys::mlx_closure_value_and_grad_free(vag_closure);
        }
        return Err(Error::ArrayCreation("Failed to create value_and_grad closure".into()));
    }

    // Create input vector
    let inputs_raw: Vec<_> = inputs.iter().map(|a| a.as_raw()).collect();
    let inputs_vec = unsafe {
        mlx_sys::mlx_vector_array_new_data(inputs_raw.as_ptr(), inputs_raw.len())
    };

    // Result vectors
    let mut values_vec = unsafe { mlx_sys::mlx_vector_array_new() };
    let mut grads_vec = unsafe { mlx_sys::mlx_vector_array_new() };

    // Apply the closure
    let apply_status = unsafe {
        mlx_sys::mlx_closure_value_and_grad_apply(
            &mut values_vec,
            &mut grads_vec,
            vag_closure,
            inputs_vec,
        )
    };

    // Cleanup (destructor handles wrapper cleanup)
    unsafe {
        mlx_sys::mlx_vector_array_free(inputs_vec);
        mlx_sys::mlx_closure_free(closure);
        mlx_sys::mlx_closure_value_and_grad_free(vag_closure);
    }

    if apply_status != 0 {
        unsafe {
            mlx_sys::mlx_vector_array_free(values_vec);
            mlx_sys::mlx_vector_array_free(grads_vec);
        }
        return Err(Error::ArrayCreation("Failed to apply value_and_grad".into()));
    }

    // Extract results
    let values = extract_vector_array(values_vec);
    let grads = extract_vector_array(grads_vec);

    unsafe {
        mlx_sys::mlx_vector_array_free(values_vec);
        mlx_sys::mlx_vector_array_free(grads_vec);
    }

    Ok((values, grads))
}

/// Compute the gradient of a scalar-valued function
///
/// This is a convenience function that returns only the gradients.
///
/// # Arguments
/// * `f` - Scalar-valued function (must return a single scalar array)
/// * `inputs` - Input arrays
/// * `argnums` - Indices of arguments to differentiate with respect to
///
/// # Returns
/// Gradients with respect to the specified arguments
///
/// # Example
/// ```ignore
/// // f(x) = x^2, compute gradient at x = 3
/// let x = Array::from_float(3.0);
/// let grads = grad(
///     |inputs| vec![&inputs[0] * &inputs[0]],
///     &[x],
///     &[0],
/// ).unwrap();
/// // grads[0] = 6.0
/// ```
pub fn grad<F>(f: F, inputs: &[Array], argnums: &[i32]) -> Result<Vec<Array>>
where
    F: Fn(&[Array]) -> Vec<Array> + Send + Sync + 'static,
{
    let (_, grads) = value_and_grad(f, inputs, argnums)?;
    Ok(grads)
}

// ============================================================================
// Helper functions
// ============================================================================

/// Extract arrays from an mlx_vector_array
fn extract_vector_array(vec: mlx_sys::mlx_vector_array) -> Vec<Array> {
    let size = unsafe { mlx_sys::mlx_vector_array_size(vec) };
    let mut arrays = Vec::with_capacity(size);

    for i in 0..size {
        let mut arr = Array::new_uninit();
        unsafe {
            mlx_sys::mlx_vector_array_get(arr.as_mut_ptr(), vec, i);
        }
        arrays.push(arr);
    }

    arrays
}

// ============================================================================
// Simplified gradient functions
// ============================================================================

/// Compute gradient of a function with a single input
///
/// Convenience function for the common case of a single-input function.
pub fn grad_single<F>(f: F, x: &Array) -> Result<Array>
where
    F: Fn(&Array) -> Array + Send + Sync + 'static,
{
    let grads = grad(
        move |inputs| vec![f(&inputs[0])],
        &[x.clone()],
        &[0],
    )?;

    grads.into_iter().next()
        .ok_or_else(|| Error::ArrayCreation("No gradient returned".into()))
}

/// Compute value and gradient of a function with a single input
pub fn value_and_grad_single<F>(f: F, x: &Array) -> Result<(Array, Array)>
where
    F: Fn(&Array) -> Array + Send + Sync + 'static,
{
    let (values, grads) = value_and_grad(
        move |inputs| vec![f(&inputs[0])],
        &[x.clone()],
        &[0],
    )?;

    let value = values.into_iter().next()
        .ok_or_else(|| Error::ArrayCreation("No value returned".into()))?;
    let grad = grads.into_iter().next()
        .ok_or_else(|| Error::ArrayCreation("No gradient returned".into()))?;

    Ok((value, grad))
}
