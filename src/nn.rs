//! Neural network operations
//!
//! This module provides common neural network activation functions
//! and building blocks.

use mlx_sys;

use crate::array::Array;
use crate::error::{Error, Result};
use crate::random;
use crate::stream::Stream;

// ============================================================================
// Activation Functions
// ============================================================================

/// Rectified Linear Unit (ReLU)
///
/// f(x) = max(0, x)
pub fn relu(x: &Array) -> Result<Array> {
    let stream = Stream::default();
    let mut result = Array::new_uninit();
    let zero = Array::from_float(0.0);

    let status = unsafe {
        mlx_sys::mlx_maximum(
            result.as_mut_ptr(),
            x.as_raw(),
            zero.as_raw(),
            stream.as_raw(),
        )
    };

    if status != 0 {
        return Err(Error::ArrayCreation("Failed to compute relu".into()));
    }

    Ok(result)
}

/// Leaky ReLU
///
/// f(x) = x if x > 0, else alpha * x
pub fn leaky_relu(x: &Array, alpha: f32) -> Result<Array> {
    // Implement using: max(x, alpha * x)
    let stream = Stream::default();
    let alpha_arr = Array::from_float(alpha);

    // Compute alpha * x
    let mut scaled = Array::new_uninit();
    let status = unsafe {
        mlx_sys::mlx_multiply(
            scaled.as_mut_ptr(),
            x.as_raw(),
            alpha_arr.as_raw(),
            stream.as_raw(),
        )
    };
    if status != 0 {
        return Err(Error::ArrayCreation("Failed to compute leaky_relu".into()));
    }

    // Compute max(x, alpha * x)
    let mut result = Array::new_uninit();
    let status = unsafe {
        mlx_sys::mlx_maximum(
            result.as_mut_ptr(),
            x.as_raw(),
            scaled.as_raw(),
            stream.as_raw(),
        )
    };
    if status != 0 {
        return Err(Error::ArrayCreation("Failed to compute leaky_relu".into()));
    }

    Ok(result)
}

/// ReLU6
///
/// f(x) = min(max(0, x), 6)
pub fn relu6(x: &Array) -> Result<Array> {
    let stream = Stream::default();
    let zero = Array::from_float(0.0);
    let six = Array::from_float(6.0);

    // max(0, x)
    let mut max_result = Array::new_uninit();
    let status = unsafe {
        mlx_sys::mlx_maximum(
            max_result.as_mut_ptr(),
            x.as_raw(),
            zero.as_raw(),
            stream.as_raw(),
        )
    };
    if status != 0 {
        return Err(Error::ArrayCreation("Failed to compute relu6".into()));
    }

    // min(max(0, x), 6)
    let mut result = Array::new_uninit();
    let status = unsafe {
        mlx_sys::mlx_minimum(
            result.as_mut_ptr(),
            max_result.as_raw(),
            six.as_raw(),
            stream.as_raw(),
        )
    };
    if status != 0 {
        return Err(Error::ArrayCreation("Failed to compute relu6".into()));
    }

    Ok(result)
}

/// Softmax
///
/// Applies softmax along the specified axis
pub fn softmax(x: &Array, axis: i32) -> Result<Array> {
    let stream = Stream::default();
    let mut result = Array::new_uninit();

    let status = unsafe {
        mlx_sys::mlx_softmax_axis(
            result.as_mut_ptr(),
            x.as_raw(),
            axis,
            true, // precise
            stream.as_raw(),
        )
    };

    if status != 0 {
        return Err(Error::ArrayCreation("Failed to compute softmax".into()));
    }

    Ok(result)
}

/// Sigmoid
///
/// f(x) = 1 / (1 + exp(-x))
pub fn sigmoid(x: &Array) -> Result<Array> {
    let stream = Stream::default();
    let mut result = Array::new_uninit();

    let status =
        unsafe { mlx_sys::mlx_sigmoid(result.as_mut_ptr(), x.as_raw(), stream.as_raw()) };

    if status != 0 {
        return Err(Error::ArrayCreation("Failed to compute sigmoid".into()));
    }

    Ok(result)
}

/// Hyperbolic tangent (tanh)
pub fn tanh(x: &Array) -> Result<Array> {
    let stream = Stream::default();
    let mut result = Array::new_uninit();

    let status = unsafe { mlx_sys::mlx_tanh(result.as_mut_ptr(), x.as_raw(), stream.as_raw()) };

    if status != 0 {
        return Err(Error::ArrayCreation("Failed to compute tanh".into()));
    }

    Ok(result)
}

/// SiLU (Sigmoid Linear Unit) / Swish
///
/// f(x) = x * sigmoid(x)
pub fn silu(x: &Array) -> Result<Array> {
    let stream = Stream::default();

    // Compute sigmoid(x)
    let sig = sigmoid(x)?;

    // Compute x * sigmoid(x)
    let mut result = Array::new_uninit();
    let status = unsafe {
        mlx_sys::mlx_multiply(
            result.as_mut_ptr(),
            x.as_raw(),
            sig.as_raw(),
            stream.as_raw(),
        )
    };

    if status != 0 {
        return Err(Error::ArrayCreation("Failed to compute silu".into()));
    }

    Ok(result)
}

/// Softplus
///
/// f(x) = log(1 + exp(x))
pub fn softplus(x: &Array) -> Result<Array> {
    let stream = Stream::default();

    // Compute exp(x)
    let mut exp_x = Array::new_uninit();
    let status = unsafe {
        mlx_sys::mlx_exp(exp_x.as_mut_ptr(), x.as_raw(), stream.as_raw())
    };
    if status != 0 {
        return Err(Error::ArrayCreation("Failed to compute softplus".into()));
    }

    // Compute log(1 + exp(x)) = log1p(exp(x))
    let mut result = Array::new_uninit();
    let status = unsafe {
        mlx_sys::mlx_log1p(result.as_mut_ptr(), exp_x.as_raw(), stream.as_raw())
    };
    if status != 0 {
        return Err(Error::ArrayCreation("Failed to compute softplus".into()));
    }

    Ok(result)
}

/// Softsign
///
/// f(x) = x / (1 + |x|)
pub fn softsign(x: &Array) -> Result<Array> {
    let stream = Stream::default();

    // Compute |x|
    let mut abs_x = Array::new_uninit();
    let status = unsafe {
        mlx_sys::mlx_abs(abs_x.as_mut_ptr(), x.as_raw(), stream.as_raw())
    };
    if status != 0 {
        return Err(Error::ArrayCreation("Failed to compute softsign".into()));
    }

    // Compute 1 + |x|
    let one = Array::from_float(1.0);
    let mut denom = Array::new_uninit();
    let status = unsafe {
        mlx_sys::mlx_add(
            denom.as_mut_ptr(),
            one.as_raw(),
            abs_x.as_raw(),
            stream.as_raw(),
        )
    };
    if status != 0 {
        return Err(Error::ArrayCreation("Failed to compute softsign".into()));
    }

    // Compute x / (1 + |x|)
    let mut result = Array::new_uninit();
    let status = unsafe {
        mlx_sys::mlx_divide(
            result.as_mut_ptr(),
            x.as_raw(),
            denom.as_raw(),
            stream.as_raw(),
        )
    };
    if status != 0 {
        return Err(Error::ArrayCreation("Failed to compute softsign".into()));
    }

    Ok(result)
}

/// GELU (Gaussian Error Linear Unit)
///
/// f(x) = x * 0.5 * (1 + erf(x / sqrt(2)))
///
/// This is the exact GELU used in BERT, GPT-2, etc.
pub fn gelu(x: &Array) -> Result<Array> {
    let stream = Stream::default();

    // x / sqrt(2)
    let sqrt2 = Array::from_float(std::f32::consts::SQRT_2);
    let mut x_scaled = Array::new_uninit();
    let status = unsafe {
        mlx_sys::mlx_divide(x_scaled.as_mut_ptr(), x.as_raw(), sqrt2.as_raw(), stream.as_raw())
    };
    if status != 0 {
        return Err(Error::ArrayCreation("Failed to compute gelu".into()));
    }

    // erf(x / sqrt(2))
    let mut erf_val = Array::new_uninit();
    let status = unsafe {
        mlx_sys::mlx_erf(erf_val.as_mut_ptr(), x_scaled.as_raw(), stream.as_raw())
    };
    if status != 0 {
        return Err(Error::ArrayCreation("Failed to compute erf".into()));
    }

    // 1 + erf(...)
    let one = Array::from_float(1.0);
    let mut one_plus_erf = Array::new_uninit();
    let status = unsafe {
        mlx_sys::mlx_add(one_plus_erf.as_mut_ptr(), one.as_raw(), erf_val.as_raw(), stream.as_raw())
    };
    if status != 0 {
        return Err(Error::ArrayCreation("Failed to compute gelu".into()));
    }

    // 0.5 * (1 + erf(...))
    let half = Array::from_float(0.5);
    let mut half_scaled = Array::new_uninit();
    let status = unsafe {
        mlx_sys::mlx_multiply(half_scaled.as_mut_ptr(), half.as_raw(), one_plus_erf.as_raw(), stream.as_raw())
    };
    if status != 0 {
        return Err(Error::ArrayCreation("Failed to compute gelu".into()));
    }

    // x * 0.5 * (1 + erf(...))
    let mut result = Array::new_uninit();
    let status = unsafe {
        mlx_sys::mlx_multiply(result.as_mut_ptr(), x.as_raw(), half_scaled.as_raw(), stream.as_raw())
    };
    if status != 0 {
        return Err(Error::ArrayCreation("Failed to compute gelu".into()));
    }

    Ok(result)
}

/// GELU Approximate (fast approximation using tanh)
///
/// f(x) = 0.5 * x * (1 + tanh(sqrt(2/pi) * (x + 0.044715 * x^3)))
///
/// This is faster than exact GELU and used in GPT-3, etc.
pub fn gelu_approx(x: &Array) -> Result<Array> {
    let stream = Stream::default();

    // Constants
    let sqrt_2_over_pi = (2.0_f32 / std::f32::consts::PI).sqrt(); // ~0.7978845608
    let coeff = 0.044715_f32;

    // x^3
    let mut x_cubed = Array::new_uninit();
    let status = unsafe {
        mlx_sys::mlx_multiply(x_cubed.as_mut_ptr(), x.as_raw(), x.as_raw(), stream.as_raw())
    };
    if status != 0 {
        return Err(Error::ArrayCreation("Failed to compute gelu_approx".into()));
    }
    let mut x_cubed_final = Array::new_uninit();
    let status = unsafe {
        mlx_sys::mlx_multiply(x_cubed_final.as_mut_ptr(), x_cubed.as_raw(), x.as_raw(), stream.as_raw())
    };
    if status != 0 {
        return Err(Error::ArrayCreation("Failed to compute gelu_approx".into()));
    }

    // 0.044715 * x^3
    let coeff_arr = Array::from_float(coeff);
    let mut scaled_x_cubed = Array::new_uninit();
    let status = unsafe {
        mlx_sys::mlx_multiply(scaled_x_cubed.as_mut_ptr(), coeff_arr.as_raw(), x_cubed_final.as_raw(), stream.as_raw())
    };
    if status != 0 {
        return Err(Error::ArrayCreation("Failed to compute gelu_approx".into()));
    }

    // x + 0.044715 * x^3
    let mut x_plus_cubed = Array::new_uninit();
    let status = unsafe {
        mlx_sys::mlx_add(x_plus_cubed.as_mut_ptr(), x.as_raw(), scaled_x_cubed.as_raw(), stream.as_raw())
    };
    if status != 0 {
        return Err(Error::ArrayCreation("Failed to compute gelu_approx".into()));
    }

    // sqrt(2/pi) * (x + 0.044715 * x^3)
    let sqrt_arr = Array::from_float(sqrt_2_over_pi);
    let mut tanh_arg = Array::new_uninit();
    let status = unsafe {
        mlx_sys::mlx_multiply(tanh_arg.as_mut_ptr(), sqrt_arr.as_raw(), x_plus_cubed.as_raw(), stream.as_raw())
    };
    if status != 0 {
        return Err(Error::ArrayCreation("Failed to compute gelu_approx".into()));
    }

    // tanh(...)
    let mut tanh_val = Array::new_uninit();
    let status = unsafe {
        mlx_sys::mlx_tanh(tanh_val.as_mut_ptr(), tanh_arg.as_raw(), stream.as_raw())
    };
    if status != 0 {
        return Err(Error::ArrayCreation("Failed to compute tanh".into()));
    }

    // 1 + tanh(...)
    let one = Array::from_float(1.0);
    let mut one_plus_tanh = Array::new_uninit();
    let status = unsafe {
        mlx_sys::mlx_add(one_plus_tanh.as_mut_ptr(), one.as_raw(), tanh_val.as_raw(), stream.as_raw())
    };
    if status != 0 {
        return Err(Error::ArrayCreation("Failed to compute gelu_approx".into()));
    }

    // x * (1 + tanh(...))
    let mut x_times = Array::new_uninit();
    let status = unsafe {
        mlx_sys::mlx_multiply(x_times.as_mut_ptr(), x.as_raw(), one_plus_tanh.as_raw(), stream.as_raw())
    };
    if status != 0 {
        return Err(Error::ArrayCreation("Failed to compute gelu_approx".into()));
    }

    // 0.5 * x * (1 + tanh(...))
    let half = Array::from_float(0.5);
    let mut result = Array::new_uninit();
    let status = unsafe {
        mlx_sys::mlx_multiply(result.as_mut_ptr(), half.as_raw(), x_times.as_raw(), stream.as_raw())
    };
    if status != 0 {
        return Err(Error::ArrayCreation("Failed to compute gelu_approx".into()));
    }

    Ok(result)
}

/// ELU (Exponential Linear Unit)
///
/// f(x) = x if x > 0, else alpha * (exp(x) - 1)
///
/// # Arguments
/// * `x` - Input array
/// * `alpha` - Scale for negative values (default: 1.0)
pub fn elu(x: &Array, alpha: f32) -> Result<Array> {
    let stream = Stream::default();

    // exp(x)
    let mut exp_x = Array::new_uninit();
    let status = unsafe {
        mlx_sys::mlx_exp(exp_x.as_mut_ptr(), x.as_raw(), stream.as_raw())
    };
    if status != 0 {
        return Err(Error::ArrayCreation("Failed to compute exp".into()));
    }

    // exp(x) - 1
    let one = Array::from_float(1.0);
    let mut exp_minus_1 = Array::new_uninit();
    let status = unsafe {
        mlx_sys::mlx_subtract(exp_minus_1.as_mut_ptr(), exp_x.as_raw(), one.as_raw(), stream.as_raw())
    };
    if status != 0 {
        return Err(Error::ArrayCreation("Failed to compute elu".into()));
    }

    // alpha * (exp(x) - 1)
    let alpha_arr = Array::from_float(alpha);
    let mut negative_part = Array::new_uninit();
    let status = unsafe {
        mlx_sys::mlx_multiply(negative_part.as_mut_ptr(), alpha_arr.as_raw(), exp_minus_1.as_raw(), stream.as_raw())
    };
    if status != 0 {
        return Err(Error::ArrayCreation("Failed to compute elu".into()));
    }

    // x > 0 condition
    let zero = Array::from_float(0.0);
    let mut condition = Array::new_uninit();
    let status = unsafe {
        mlx_sys::mlx_greater(condition.as_mut_ptr(), x.as_raw(), zero.as_raw(), stream.as_raw())
    };
    if status != 0 {
        return Err(Error::ArrayCreation("Failed to compute condition".into()));
    }

    // where(x > 0, x, alpha * (exp(x) - 1))
    let mut result = Array::new_uninit();
    let status = unsafe {
        mlx_sys::mlx_where(result.as_mut_ptr(), condition.as_raw(), x.as_raw(), negative_part.as_raw(), stream.as_raw())
    };
    if status != 0 {
        return Err(Error::ArrayCreation("Failed to compute elu".into()));
    }

    Ok(result)
}

/// SELU (Scaled Exponential Linear Unit)
///
/// f(x) = scale * (x if x > 0 else alpha * (exp(x) - 1))
///
/// Uses the self-normalizing constants:
/// - alpha ≈ 1.6732632423543772
/// - scale ≈ 1.0507009873554805
pub fn selu(x: &Array) -> Result<Array> {
    let stream = Stream::default();

    // SELU constants
    let alpha: f32 = 1.6732632423543772;
    let scale: f32 = 1.0507009873554805;

    // First compute ELU with alpha
    let elu_result = elu(x, alpha)?;

    // Then scale
    let scale_arr = Array::from_float(scale);
    let mut result = Array::new_uninit();
    let status = unsafe {
        mlx_sys::mlx_multiply(result.as_mut_ptr(), scale_arr.as_raw(), elu_result.as_raw(), stream.as_raw())
    };
    if status != 0 {
        return Err(Error::ArrayCreation("Failed to compute selu".into()));
    }

    Ok(result)
}

/// CELU (Continuously Differentiable ELU)
///
/// f(x) = max(0, x) + min(0, alpha * (exp(x/alpha) - 1))
///
/// # Arguments
/// * `x` - Input array
/// * `alpha` - Scale parameter (default: 1.0)
pub fn celu(x: &Array, alpha: f32) -> Result<Array> {
    let stream = Stream::default();
    let alpha_arr = Array::from_float(alpha);
    let zero = Array::from_float(0.0);
    let one = Array::from_float(1.0);

    // x / alpha
    let mut x_div_alpha = Array::new_uninit();
    let status = unsafe {
        mlx_sys::mlx_divide(x_div_alpha.as_mut_ptr(), x.as_raw(), alpha_arr.as_raw(), stream.as_raw())
    };
    if status != 0 {
        return Err(Error::ArrayCreation("Failed to compute celu".into()));
    }

    // exp(x / alpha)
    let mut exp_val = Array::new_uninit();
    let status = unsafe {
        mlx_sys::mlx_exp(exp_val.as_mut_ptr(), x_div_alpha.as_raw(), stream.as_raw())
    };
    if status != 0 {
        return Err(Error::ArrayCreation("Failed to compute exp".into()));
    }

    // exp(x/alpha) - 1
    let mut exp_minus_1 = Array::new_uninit();
    let status = unsafe {
        mlx_sys::mlx_subtract(exp_minus_1.as_mut_ptr(), exp_val.as_raw(), one.as_raw(), stream.as_raw())
    };
    if status != 0 {
        return Err(Error::ArrayCreation("Failed to compute celu".into()));
    }

    // alpha * (exp(x/alpha) - 1)
    let mut negative_part = Array::new_uninit();
    let status = unsafe {
        mlx_sys::mlx_multiply(negative_part.as_mut_ptr(), alpha_arr.as_raw(), exp_minus_1.as_raw(), stream.as_raw())
    };
    if status != 0 {
        return Err(Error::ArrayCreation("Failed to compute celu".into()));
    }

    // max(0, x)
    let mut positive_part = Array::new_uninit();
    let status = unsafe {
        mlx_sys::mlx_maximum(positive_part.as_mut_ptr(), zero.as_raw(), x.as_raw(), stream.as_raw())
    };
    if status != 0 {
        return Err(Error::ArrayCreation("Failed to compute max".into()));
    }

    // min(0, alpha * (exp(x/alpha) - 1))
    let mut clamped_negative = Array::new_uninit();
    let status = unsafe {
        mlx_sys::mlx_minimum(clamped_negative.as_mut_ptr(), zero.as_raw(), negative_part.as_raw(), stream.as_raw())
    };
    if status != 0 {
        return Err(Error::ArrayCreation("Failed to compute min".into()));
    }

    // max(0, x) + min(0, ...)
    let mut result = Array::new_uninit();
    let status = unsafe {
        mlx_sys::mlx_add(result.as_mut_ptr(), positive_part.as_raw(), clamped_negative.as_raw(), stream.as_raw())
    };
    if status != 0 {
        return Err(Error::ArrayCreation("Failed to compute celu".into()));
    }

    Ok(result)
}

/// Mish activation
///
/// f(x) = x * tanh(softplus(x)) = x * tanh(log(1 + exp(x)))
pub fn mish(x: &Array) -> Result<Array> {
    let stream = Stream::default();

    // softplus(x) = log(1 + exp(x))
    let mut exp_x = Array::new_uninit();
    let status = unsafe {
        mlx_sys::mlx_exp(exp_x.as_mut_ptr(), x.as_raw(), stream.as_raw())
    };
    if status != 0 {
        return Err(Error::ArrayCreation("Failed to compute exp".into()));
    }

    let mut log1p_exp = Array::new_uninit();
    let status = unsafe {
        mlx_sys::mlx_log1p(log1p_exp.as_mut_ptr(), exp_x.as_raw(), stream.as_raw())
    };
    if status != 0 {
        return Err(Error::ArrayCreation("Failed to compute log1p".into()));
    }

    // tanh(softplus(x))
    let mut tanh_val = Array::new_uninit();
    let status = unsafe {
        mlx_sys::mlx_tanh(tanh_val.as_mut_ptr(), log1p_exp.as_raw(), stream.as_raw())
    };
    if status != 0 {
        return Err(Error::ArrayCreation("Failed to compute tanh".into()));
    }

    // x * tanh(softplus(x))
    let mut result = Array::new_uninit();
    let status = unsafe {
        mlx_sys::mlx_multiply(result.as_mut_ptr(), x.as_raw(), tanh_val.as_raw(), stream.as_raw())
    };
    if status != 0 {
        return Err(Error::ArrayCreation("Failed to compute mish".into()));
    }

    Ok(result)
}

/// Hardswish activation
///
/// f(x) = x * relu6(x + 3) / 6
///
/// Used in MobileNetV3.
pub fn hardswish(x: &Array) -> Result<Array> {
    let stream = Stream::default();
    let three = Array::from_float(3.0);
    let six = Array::from_float(6.0);
    let zero = Array::from_float(0.0);

    // x + 3
    let mut x_plus_3 = Array::new_uninit();
    let status = unsafe {
        mlx_sys::mlx_add(x_plus_3.as_mut_ptr(), x.as_raw(), three.as_raw(), stream.as_raw())
    };
    if status != 0 {
        return Err(Error::ArrayCreation("Failed to compute hardswish".into()));
    }

    // relu6(x + 3) = min(max(0, x+3), 6)
    let mut relu_part = Array::new_uninit();
    let status = unsafe {
        mlx_sys::mlx_maximum(relu_part.as_mut_ptr(), zero.as_raw(), x_plus_3.as_raw(), stream.as_raw())
    };
    if status != 0 {
        return Err(Error::ArrayCreation("Failed to compute hardswish".into()));
    }

    let mut relu6_part = Array::new_uninit();
    let status = unsafe {
        mlx_sys::mlx_minimum(relu6_part.as_mut_ptr(), relu_part.as_raw(), six.as_raw(), stream.as_raw())
    };
    if status != 0 {
        return Err(Error::ArrayCreation("Failed to compute hardswish".into()));
    }

    // x * relu6(x + 3)
    let mut x_times_relu6 = Array::new_uninit();
    let status = unsafe {
        mlx_sys::mlx_multiply(x_times_relu6.as_mut_ptr(), x.as_raw(), relu6_part.as_raw(), stream.as_raw())
    };
    if status != 0 {
        return Err(Error::ArrayCreation("Failed to compute hardswish".into()));
    }

    // / 6
    let mut result = Array::new_uninit();
    let status = unsafe {
        mlx_sys::mlx_divide(result.as_mut_ptr(), x_times_relu6.as_raw(), six.as_raw(), stream.as_raw())
    };
    if status != 0 {
        return Err(Error::ArrayCreation("Failed to compute hardswish".into()));
    }

    Ok(result)
}

/// Hardsigmoid activation
///
/// f(x) = relu6(x + 3) / 6
///
/// A fast approximation to sigmoid.
pub fn hardsigmoid(x: &Array) -> Result<Array> {
    let stream = Stream::default();
    let three = Array::from_float(3.0);
    let six = Array::from_float(6.0);
    let zero = Array::from_float(0.0);

    // x + 3
    let mut x_plus_3 = Array::new_uninit();
    let status = unsafe {
        mlx_sys::mlx_add(x_plus_3.as_mut_ptr(), x.as_raw(), three.as_raw(), stream.as_raw())
    };
    if status != 0 {
        return Err(Error::ArrayCreation("Failed to compute hardsigmoid".into()));
    }

    // relu6(x + 3) = min(max(0, x+3), 6)
    let mut relu_part = Array::new_uninit();
    let status = unsafe {
        mlx_sys::mlx_maximum(relu_part.as_mut_ptr(), zero.as_raw(), x_plus_3.as_raw(), stream.as_raw())
    };
    if status != 0 {
        return Err(Error::ArrayCreation("Failed to compute hardsigmoid".into()));
    }

    let mut relu6_part = Array::new_uninit();
    let status = unsafe {
        mlx_sys::mlx_minimum(relu6_part.as_mut_ptr(), relu_part.as_raw(), six.as_raw(), stream.as_raw())
    };
    if status != 0 {
        return Err(Error::ArrayCreation("Failed to compute hardsigmoid".into()));
    }

    // / 6
    let mut result = Array::new_uninit();
    let status = unsafe {
        mlx_sys::mlx_divide(result.as_mut_ptr(), relu6_part.as_raw(), six.as_raw(), stream.as_raw())
    };
    if status != 0 {
        return Err(Error::ArrayCreation("Failed to compute hardsigmoid".into()));
    }

    Ok(result)
}

/// Hardtanh activation
///
/// f(x) = clip(x, min_val, max_val)
///
/// # Arguments
/// * `x` - Input array
/// * `min_val` - Minimum value (default: -1.0)
/// * `max_val` - Maximum value (default: 1.0)
pub fn hardtanh(x: &Array, min_val: f32, max_val: f32) -> Result<Array> {
    let stream = Stream::default();
    let min_arr = Array::from_float(min_val);
    let max_arr = Array::from_float(max_val);

    let mut result = Array::new_uninit();
    let status = unsafe {
        mlx_sys::mlx_clip(
            result.as_mut_ptr(),
            x.as_raw(),
            min_arr.as_raw(),
            max_arr.as_raw(),
            stream.as_raw(),
        )
    };
    if status != 0 {
        return Err(Error::ArrayCreation("Failed to compute hardtanh".into()));
    }

    Ok(result)
}

/// Log Softmax
///
/// f(x) = log(softmax(x)) = x - log(sum(exp(x)))
///
/// More numerically stable than computing softmax then log.
pub fn log_softmax(x: &Array, axis: i32) -> Result<Array> {
    let stream = Stream::default();

    // For numerical stability: log_softmax(x) = x - logsumexp(x)
    // logsumexp(x) = max(x) + log(sum(exp(x - max(x))))

    // max(x) along axis
    let mut max_x = Array::new_uninit();
    let status = unsafe {
        mlx_sys::mlx_max_axis(max_x.as_mut_ptr(), x.as_raw(), axis, true, stream.as_raw())
    };
    if status != 0 {
        return Err(Error::ArrayCreation("Failed to compute max".into()));
    }

    // x - max(x)
    let mut x_shifted = Array::new_uninit();
    let status = unsafe {
        mlx_sys::mlx_subtract(x_shifted.as_mut_ptr(), x.as_raw(), max_x.as_raw(), stream.as_raw())
    };
    if status != 0 {
        return Err(Error::ArrayCreation("Failed to subtract".into()));
    }

    // exp(x - max(x))
    let mut exp_shifted = Array::new_uninit();
    let status = unsafe {
        mlx_sys::mlx_exp(exp_shifted.as_mut_ptr(), x_shifted.as_raw(), stream.as_raw())
    };
    if status != 0 {
        return Err(Error::ArrayCreation("Failed to compute exp".into()));
    }

    // sum(exp(x - max(x)))
    let mut sum_exp = Array::new_uninit();
    let status = unsafe {
        mlx_sys::mlx_sum_axis(sum_exp.as_mut_ptr(), exp_shifted.as_raw(), axis, true, stream.as_raw())
    };
    if status != 0 {
        return Err(Error::ArrayCreation("Failed to compute sum".into()));
    }

    // log(sum(exp(x - max(x))))
    let mut log_sum_exp = Array::new_uninit();
    let status = unsafe {
        mlx_sys::mlx_log(log_sum_exp.as_mut_ptr(), sum_exp.as_raw(), stream.as_raw())
    };
    if status != 0 {
        return Err(Error::ArrayCreation("Failed to compute log".into()));
    }

    // logsumexp = max(x) + log(sum(exp(x - max(x))))
    let mut logsumexp = Array::new_uninit();
    let status = unsafe {
        mlx_sys::mlx_add(logsumexp.as_mut_ptr(), max_x.as_raw(), log_sum_exp.as_raw(), stream.as_raw())
    };
    if status != 0 {
        return Err(Error::ArrayCreation("Failed to compute logsumexp".into()));
    }

    // x - logsumexp(x)
    let mut result = Array::new_uninit();
    let status = unsafe {
        mlx_sys::mlx_subtract(result.as_mut_ptr(), x.as_raw(), logsumexp.as_raw(), stream.as_raw())
    };
    if status != 0 {
        return Err(Error::ArrayCreation("Failed to compute log_softmax".into()));
    }

    Ok(result)
}

/// GLU (Gated Linear Unit)
///
/// Splits the input in half along the specified axis and applies:
/// f(x) = a * sigmoid(b) where [a, b] = split(x)
///
/// # Arguments
/// * `x` - Input array (must have even size along specified axis)
/// * `axis` - Axis to split along (default: -1)
pub fn glu(x: &Array, axis: i32) -> Result<Array> {
    let stream = Stream::default();
    let shape = x.shape();

    // Normalize axis
    let ndim = shape.len() as i32;
    let axis = if axis < 0 { axis + ndim } else { axis };

    if axis < 0 || axis >= ndim {
        return Err(Error::InvalidShape("Invalid axis for GLU".into()));
    }

    let dim_size = shape[axis as usize];
    if dim_size % 2 != 0 {
        return Err(Error::InvalidShape(
            "GLU requires even size along split axis".into(),
        ));
    }

    // Split into two halves
    let mut result_vec = unsafe { mlx_sys::mlx_vector_array_new() };
    let status = unsafe {
        mlx_sys::mlx_split(
            &mut result_vec,
            x.as_raw(),
            2,
            axis,
            stream.as_raw(),
        )
    };
    if status != 0 {
        unsafe { mlx_sys::mlx_vector_array_free(result_vec) };
        return Err(Error::ArrayCreation("Failed to split for GLU".into()));
    }

    // Get the two halves
    let mut a = Array::new_uninit();
    let mut b = Array::new_uninit();
    unsafe {
        mlx_sys::mlx_vector_array_get(a.as_mut_ptr(), result_vec, 0);
        mlx_sys::mlx_vector_array_get(b.as_mut_ptr(), result_vec, 1);
        mlx_sys::mlx_vector_array_free(result_vec);
    }

    // sigmoid(b)
    let mut sig_b = Array::new_uninit();
    let status = unsafe {
        mlx_sys::mlx_sigmoid(sig_b.as_mut_ptr(), b.as_raw(), stream.as_raw())
    };
    if status != 0 {
        return Err(Error::ArrayCreation("Failed to compute sigmoid".into()));
    }

    // a * sigmoid(b)
    let mut result = Array::new_uninit();
    let status = unsafe {
        mlx_sys::mlx_multiply(result.as_mut_ptr(), a.as_raw(), sig_b.as_raw(), stream.as_raw())
    };
    if status != 0 {
        return Err(Error::ArrayCreation("Failed to compute GLU".into()));
    }

    Ok(result)
}

// ============================================================================
// Loss Functions
// ============================================================================

/// Mean squared error loss
pub fn mse_loss(predictions: &Array, targets: &Array, reduction: &str) -> Result<Array> {
    let diff = predictions - targets;
    let squared = &diff * &diff;

    match reduction {
        "mean" => squared.mean_all(false),
        "sum" => squared.sum_all(false),
        "none" => Ok(squared),
        _ => Err(Error::NotSupported(format!("Unknown reduction: {}", reduction))),
    }
}

/// L1 loss (Mean Absolute Error)
pub fn l1_loss(predictions: &Array, targets: &Array, reduction: &str) -> Result<Array> {
    let diff = predictions - targets;
    let abs_diff = diff.abs()?;

    match reduction {
        "mean" => abs_diff.mean_all(false),
        "sum" => abs_diff.sum_all(false),
        "none" => Ok(abs_diff),
        _ => Err(Error::NotSupported(format!("Unknown reduction: {}", reduction))),
    }
}

/// Binary cross entropy loss
///
/// For predictions p and targets y:
/// BCE = -[y * log(p) + (1 - y) * log(1 - p)]
pub fn binary_cross_entropy(
    predictions: &Array,
    targets: &Array,
    reduction: &str,
) -> Result<Array> {
    let stream = Stream::default();
    let one = Array::from_float(1.0);
    let eps = Array::from_float(1e-7);

    // Clamp predictions to avoid log(0)
    let mut p_clamped = Array::new_uninit();
    let status = unsafe {
        mlx_sys::mlx_maximum(
            p_clamped.as_mut_ptr(),
            predictions.as_raw(),
            eps.as_raw(),
            stream.as_raw(),
        )
    };
    if status != 0 {
        return Err(Error::ArrayCreation("Failed to compute BCE".into()));
    }

    let one_minus_eps = &one - &eps;
    let mut p_safe = Array::new_uninit();
    let status = unsafe {
        mlx_sys::mlx_minimum(
            p_safe.as_mut_ptr(),
            p_clamped.as_raw(),
            one_minus_eps.as_raw(),
            stream.as_raw(),
        )
    };
    if status != 0 {
        return Err(Error::ArrayCreation("Failed to compute BCE".into()));
    }

    // log(p)
    let mut log_p = Array::new_uninit();
    let status = unsafe {
        mlx_sys::mlx_log(log_p.as_mut_ptr(), p_safe.as_raw(), stream.as_raw())
    };
    if status != 0 {
        return Err(Error::ArrayCreation("Failed to compute BCE".into()));
    }

    // 1 - p
    let one_minus_p = &one - &p_safe;

    // log(1 - p)
    let mut log_one_minus_p = Array::new_uninit();
    let status = unsafe {
        mlx_sys::mlx_log(log_one_minus_p.as_mut_ptr(), one_minus_p.as_raw(), stream.as_raw())
    };
    if status != 0 {
        return Err(Error::ArrayCreation("Failed to compute BCE".into()));
    }

    // y * log(p)
    let term1 = targets * &log_p;

    // (1 - y)
    let one_minus_y = &one - targets;

    // (1 - y) * log(1 - p)
    let term2 = &one_minus_y * &log_one_minus_p;

    // -[term1 + term2]
    let sum_terms = &term1 + &term2;
    let loss = -&sum_terms;

    match reduction {
        "mean" => loss.mean_all(false),
        "sum" => loss.sum_all(false),
        "none" => Ok(loss),
        _ => Err(Error::NotSupported(format!("Unknown reduction: {}", reduction))),
    }
}

// ============================================================================
// Convolution Operations
// ============================================================================

/// 1D Convolution
///
/// MLX uses **channels-last** format for convolutions.
///
/// # Arguments
/// * `input` - Input array of shape (N, L, C_in) - batch, length, channels
/// * `weight` - Weight array of shape (C_out, K, C_in/groups) - out_channels, kernel_size, in_channels
/// * `stride` - Stride of the convolution (default: 1)
/// * `padding` - Padding on both sides (default: 0)
/// * `dilation` - Dilation of the kernel (default: 1)
/// * `groups` - Number of groups (default: 1)
///
/// # Returns
/// Output array of shape (N, L_out, C_out)
pub fn conv1d(
    input: &Array,
    weight: &Array,
    stride: i32,
    padding: i32,
    dilation: i32,
    groups: i32,
) -> Result<Array> {
    let stream = Stream::default();
    let mut result = Array::new_uninit();

    let status = unsafe {
        mlx_sys::mlx_conv1d(
            result.as_mut_ptr(),
            input.as_raw(),
            weight.as_raw(),
            stride,
            padding,
            dilation,
            groups,
            stream.as_raw(),
        )
    };

    if status != 0 {
        return Err(Error::ArrayCreation("Failed to compute conv1d".into()));
    }

    Ok(result)
}

/// 1D Convolution with default parameters
///
/// Convenience function with stride=1, padding=0, dilation=1, groups=1
pub fn conv1d_simple(input: &Array, weight: &Array) -> Result<Array> {
    conv1d(input, weight, 1, 0, 1, 1)
}

/// 2D Convolution
///
/// MLX uses **channels-last** format for convolutions.
///
/// # Arguments
/// * `input` - Input array of shape (N, H, W, C_in) - batch, height, width, channels
/// * `weight` - Weight array of shape (C_out, kH, kW, C_in/groups) - out_channels, kernel_h, kernel_w, in_channels
/// * `stride` - Stride as (stride_h, stride_w) (default: (1, 1))
/// * `padding` - Padding as (pad_h, pad_w) (default: (0, 0))
/// * `dilation` - Dilation as (dilation_h, dilation_w) (default: (1, 1))
/// * `groups` - Number of groups (default: 1)
///
/// # Returns
/// Output array of shape (N, H_out, W_out, C_out)
pub fn conv2d(
    input: &Array,
    weight: &Array,
    stride: (i32, i32),
    padding: (i32, i32),
    dilation: (i32, i32),
    groups: i32,
) -> Result<Array> {
    let stream = Stream::default();
    let mut result = Array::new_uninit();

    let status = unsafe {
        mlx_sys::mlx_conv2d(
            result.as_mut_ptr(),
            input.as_raw(),
            weight.as_raw(),
            stride.0,
            stride.1,
            padding.0,
            padding.1,
            dilation.0,
            dilation.1,
            groups,
            stream.as_raw(),
        )
    };

    if status != 0 {
        return Err(Error::ArrayCreation("Failed to compute conv2d".into()));
    }

    Ok(result)
}

/// 2D Convolution with default parameters
///
/// Convenience function with stride=(1,1), padding=(0,0), dilation=(1,1), groups=1
pub fn conv2d_simple(input: &Array, weight: &Array) -> Result<Array> {
    conv2d(input, weight, (1, 1), (0, 0), (1, 1), 1)
}

// ============================================================================
// Pooling Operations (implemented using as_strided)
// ============================================================================

/// 1D Max Pooling
///
/// MLX uses **channels-last** format: Input shape is (N, L, C).
///
/// # Arguments
/// * `input` - Input array of shape (N, L, C) - batch, length, channels
/// * `kernel_size` - Size of the pooling window
/// * `stride` - Stride of the pooling (defaults to kernel_size if None)
///
/// # Returns
/// Output array of shape (N, L_out, C)
pub fn max_pool1d(input: &Array, kernel_size: i32, stride: Option<i32>) -> Result<Array> {
    let stride = stride.unwrap_or(kernel_size);
    let shape = input.shape();

    if shape.len() != 3 {
        return Err(Error::InvalidShape(
            "max_pool1d requires 3D input (N, L, C)".into(),
        ));
    }

    let batch = shape[0];
    let length = shape[1];
    let channels = shape[2];

    let out_len = (length - kernel_size) / stride + 1;
    if out_len <= 0 {
        return Err(Error::InvalidShape(
            "Pooling kernel is larger than input".into(),
        ));
    }

    let stream = Stream::default();

    // Use as_strided to create sliding windows
    // New shape: (N, out_len, kernel_size, C)
    let new_shape = [batch, out_len, kernel_size, channels];
    let new_strides = [
        (length * channels) as i64,
        (stride * channels) as i64,
        channels as i64,
        1i64,
    ];

    let mut windowed = Array::new_uninit();
    let status = unsafe {
        mlx_sys::mlx_as_strided(
            windowed.as_mut_ptr(),
            input.as_raw(),
            new_shape.as_ptr(),
            new_shape.len(),
            new_strides.as_ptr(),
            new_strides.len(),
            0,
            stream.as_raw(),
        )
    };
    if status != 0 {
        return Err(Error::ArrayCreation("Failed to create strided view".into()));
    }

    // Max over the kernel dimension (axis=2)
    let mut result = Array::new_uninit();
    let status = unsafe {
        mlx_sys::mlx_max_axis(
            result.as_mut_ptr(),
            windowed.as_raw(),
            2,
            false,
            stream.as_raw(),
        )
    };
    if status != 0 {
        return Err(Error::ArrayCreation("Failed to compute max_pool1d".into()));
    }

    Ok(result)
}

/// 1D Average Pooling
///
/// MLX uses **channels-last** format: Input shape is (N, L, C).
///
/// # Arguments
/// * `input` - Input array of shape (N, L, C) - batch, length, channels
/// * `kernel_size` - Size of the pooling window
/// * `stride` - Stride of the pooling (defaults to kernel_size if None)
///
/// # Returns
/// Output array of shape (N, L_out, C)
pub fn avg_pool1d(input: &Array, kernel_size: i32, stride: Option<i32>) -> Result<Array> {
    let stride = stride.unwrap_or(kernel_size);
    let shape = input.shape();

    if shape.len() != 3 {
        return Err(Error::InvalidShape(
            "avg_pool1d requires 3D input (N, L, C)".into(),
        ));
    }

    let batch = shape[0];
    let length = shape[1];
    let channels = shape[2];

    let out_len = (length - kernel_size) / stride + 1;
    if out_len <= 0 {
        return Err(Error::InvalidShape(
            "Pooling kernel is larger than input".into(),
        ));
    }

    let stream = Stream::default();

    // Use as_strided to create sliding windows
    // New shape: (N, out_len, kernel_size, C)
    let new_shape = [batch, out_len, kernel_size, channels];
    let new_strides = [
        (length * channels) as i64,
        (stride * channels) as i64,
        channels as i64,
        1i64,
    ];

    let mut windowed = Array::new_uninit();
    let status = unsafe {
        mlx_sys::mlx_as_strided(
            windowed.as_mut_ptr(),
            input.as_raw(),
            new_shape.as_ptr(),
            new_shape.len(),
            new_strides.as_ptr(),
            new_strides.len(),
            0,
            stream.as_raw(),
        )
    };
    if status != 0 {
        return Err(Error::ArrayCreation("Failed to create strided view".into()));
    }

    // Mean over the kernel dimension (axis=2)
    let mut result = Array::new_uninit();
    let status = unsafe {
        mlx_sys::mlx_mean_axis(
            result.as_mut_ptr(),
            windowed.as_raw(),
            2,
            false,
            stream.as_raw(),
        )
    };
    if status != 0 {
        return Err(Error::ArrayCreation("Failed to compute avg_pool1d".into()));
    }

    Ok(result)
}

/// 2D Max Pooling
///
/// MLX uses **channels-last** format: Input shape is (N, H, W, C).
///
/// # Arguments
/// * `input` - Input array of shape (N, H, W, C) - batch, height, width, channels
/// * `kernel_size` - Size of the pooling window as (kH, kW)
/// * `stride` - Stride of the pooling (defaults to kernel_size if None)
///
/// # Returns
/// Output array of shape (N, H_out, W_out, C)
pub fn max_pool2d(
    input: &Array,
    kernel_size: (i32, i32),
    stride: Option<(i32, i32)>,
) -> Result<Array> {
    let stride = stride.unwrap_or(kernel_size);
    let shape = input.shape();

    if shape.len() != 4 {
        return Err(Error::InvalidShape(
            "max_pool2d requires 4D input (N, H, W, C)".into(),
        ));
    }

    let batch = shape[0];
    let height = shape[1];
    let width = shape[2];
    let channels = shape[3];

    let out_h = (height - kernel_size.0) / stride.0 + 1;
    let out_w = (width - kernel_size.1) / stride.1 + 1;

    if out_h <= 0 || out_w <= 0 {
        return Err(Error::InvalidShape(
            "Pooling kernel is larger than input".into(),
        ));
    }

    let stream = Stream::default();

    // Use as_strided to create sliding windows
    // New shape: (N, out_h, out_w, kH, kW, C)
    let new_shape = [batch, out_h, out_w, kernel_size.0, kernel_size.1, channels];
    let new_strides = [
        (height * width * channels) as i64,
        (stride.0 * width * channels) as i64,
        (stride.1 * channels) as i64,
        (width * channels) as i64,
        channels as i64,
        1i64,
    ];

    let mut windowed = Array::new_uninit();
    let status = unsafe {
        mlx_sys::mlx_as_strided(
            windowed.as_mut_ptr(),
            input.as_raw(),
            new_shape.as_ptr(),
            new_shape.len(),
            new_strides.as_ptr(),
            new_strides.len(),
            0,
            stream.as_raw(),
        )
    };
    if status != 0 {
        return Err(Error::ArrayCreation("Failed to create strided view".into()));
    }

    // Max over kW dimension (axis=4)
    let mut temp = Array::new_uninit();
    let status = unsafe {
        mlx_sys::mlx_max_axis(
            temp.as_mut_ptr(),
            windowed.as_raw(),
            4,
            false,
            stream.as_raw(),
        )
    };
    if status != 0 {
        return Err(Error::ArrayCreation("Failed to compute max_pool2d".into()));
    }

    // Max over kH dimension (axis=3)
    let mut result = Array::new_uninit();
    let status = unsafe {
        mlx_sys::mlx_max_axis(
            result.as_mut_ptr(),
            temp.as_raw(),
            3,
            false,
            stream.as_raw(),
        )
    };
    if status != 0 {
        return Err(Error::ArrayCreation("Failed to compute max_pool2d".into()));
    }

    Ok(result)
}

/// 2D Average Pooling
///
/// MLX uses **channels-last** format: Input shape is (N, H, W, C).
///
/// # Arguments
/// * `input` - Input array of shape (N, H, W, C) - batch, height, width, channels
/// * `kernel_size` - Size of the pooling window as (kH, kW)
/// * `stride` - Stride of the pooling (defaults to kernel_size if None)
///
/// # Returns
/// Output array of shape (N, H_out, W_out, C)
pub fn avg_pool2d(
    input: &Array,
    kernel_size: (i32, i32),
    stride: Option<(i32, i32)>,
) -> Result<Array> {
    let stride = stride.unwrap_or(kernel_size);
    let shape = input.shape();

    if shape.len() != 4 {
        return Err(Error::InvalidShape(
            "avg_pool2d requires 4D input (N, H, W, C)".into(),
        ));
    }

    let batch = shape[0];
    let height = shape[1];
    let width = shape[2];
    let channels = shape[3];

    let out_h = (height - kernel_size.0) / stride.0 + 1;
    let out_w = (width - kernel_size.1) / stride.1 + 1;

    if out_h <= 0 || out_w <= 0 {
        return Err(Error::InvalidShape(
            "Pooling kernel is larger than input".into(),
        ));
    }

    let stream = Stream::default();

    // Use as_strided to create sliding windows
    // New shape: (N, out_h, out_w, kH, kW, C)
    let new_shape = [batch, out_h, out_w, kernel_size.0, kernel_size.1, channels];
    let new_strides = [
        (height * width * channels) as i64,
        (stride.0 * width * channels) as i64,
        (stride.1 * channels) as i64,
        (width * channels) as i64,
        channels as i64,
        1i64,
    ];

    let mut windowed = Array::new_uninit();
    let status = unsafe {
        mlx_sys::mlx_as_strided(
            windowed.as_mut_ptr(),
            input.as_raw(),
            new_shape.as_ptr(),
            new_shape.len(),
            new_strides.as_ptr(),
            new_strides.len(),
            0,
            stream.as_raw(),
        )
    };
    if status != 0 {
        return Err(Error::ArrayCreation("Failed to create strided view".into()));
    }

    // Mean over kW dimension (axis=4)
    let mut temp = Array::new_uninit();
    let status = unsafe {
        mlx_sys::mlx_mean_axis(
            temp.as_mut_ptr(),
            windowed.as_raw(),
            4,
            false,
            stream.as_raw(),
        )
    };
    if status != 0 {
        return Err(Error::ArrayCreation("Failed to compute avg_pool2d".into()));
    }

    // Mean over kH dimension (axis=3)
    let mut result = Array::new_uninit();
    let status = unsafe {
        mlx_sys::mlx_mean_axis(
            result.as_mut_ptr(),
            temp.as_raw(),
            3,
            false,
            stream.as_raw(),
        )
    };
    if status != 0 {
        return Err(Error::ArrayCreation("Failed to compute avg_pool2d".into()));
    }

    Ok(result)
}

// ============================================================================
// Normalization Operations
// ============================================================================

/// Layer Normalization
///
/// Normalizes over the last dimension of the input.
///
/// # Arguments
/// * `x` - Input array
/// * `weight` - Scale parameter (gamma)
/// * `bias` - Shift parameter (beta)
/// * `eps` - Small constant for numerical stability (default: 1e-5)
///
/// # Returns
/// Normalized array
pub fn layer_norm(x: &Array, weight: &Array, bias: &Array, eps: f32) -> Result<Array> {
    let stream = Stream::default();
    let mut result = Array::new_uninit();

    let status = unsafe {
        mlx_sys::mlx_fast_layer_norm(
            result.as_mut_ptr(),
            x.as_raw(),
            weight.as_raw(),
            bias.as_raw(),
            eps,
            stream.as_raw(),
        )
    };

    if status != 0 {
        return Err(Error::ArrayCreation("Failed to compute layer_norm".into()));
    }

    Ok(result)
}

/// Layer Normalization without learnable parameters
///
/// # Arguments
/// * `x` - Input array
/// * `eps` - Small constant for numerical stability
///
/// # Returns
/// Normalized array
pub fn layer_norm_no_params(x: &Array, eps: f32) -> Result<Array> {
    let shape = x.shape();
    let last_dim = *shape.last().ok_or_else(|| {
        Error::InvalidShape("Input must have at least one dimension".into())
    })?;

    // Create weight (ones) and bias (zeros)
    let weight = Array::ones::<f32>(&[last_dim])?;
    let bias = Array::zeros::<f32>(&[last_dim])?;

    layer_norm(x, &weight, &bias, eps)
}

/// RMS Normalization (Root Mean Square Layer Normalization)
///
/// Normalizes using root mean square instead of mean and variance.
///
/// # Arguments
/// * `x` - Input array
/// * `weight` - Scale parameter
/// * `eps` - Small constant for numerical stability (default: 1e-5)
///
/// # Returns
/// Normalized array
pub fn rms_norm(x: &Array, weight: &Array, eps: f32) -> Result<Array> {
    let stream = Stream::default();
    let mut result = Array::new_uninit();

    let status = unsafe {
        mlx_sys::mlx_fast_rms_norm(
            result.as_mut_ptr(),
            x.as_raw(),
            weight.as_raw(),
            eps,
            stream.as_raw(),
        )
    };

    if status != 0 {
        return Err(Error::ArrayCreation("Failed to compute rms_norm".into()));
    }

    Ok(result)
}

/// RMS Normalization without learnable parameters
///
/// # Arguments
/// * `x` - Input array
/// * `eps` - Small constant for numerical stability
///
/// # Returns
/// Normalized array
pub fn rms_norm_no_params(x: &Array, eps: f32) -> Result<Array> {
    let shape = x.shape();
    let last_dim = *shape.last().ok_or_else(|| {
        Error::InvalidShape("Input must have at least one dimension".into())
    })?;

    // Create weight (ones)
    let weight = Array::ones::<f32>(&[last_dim])?;

    rms_norm(x, &weight, eps)
}

/// Batch Normalization
///
/// Normalizes over the batch dimension (axis 0) and optionally spatial dimensions.
///
/// # Arguments
/// * `x` - Input array of shape (N, ..., C) for channels-last
/// * `weight` - Scale parameter (gamma) of shape (C,)
/// * `bias` - Shift parameter (beta) of shape (C,)
/// * `running_mean` - Running mean for inference (optional)
/// * `running_var` - Running variance for inference (optional)
/// * `training` - If true, use batch statistics; if false, use running statistics
/// * `momentum` - Momentum for running statistics update (default: 0.1)
/// * `eps` - Small constant for numerical stability (default: 1e-5)
///
/// # Returns
/// Normalized array
pub fn batch_norm(
    x: &Array,
    weight: &Array,
    bias: &Array,
    running_mean: Option<&Array>,
    running_var: Option<&Array>,
    training: bool,
    eps: f32,
) -> Result<Array> {
    let stream = Stream::default();
    let shape = x.shape();
    let ndim = shape.len();

    if ndim < 2 {
        return Err(Error::InvalidShape(
            "batch_norm requires at least 2D input".into(),
        ));
    }

    // Compute axes to reduce over (all except last dimension for channels-last)
    let reduce_axes: Vec<i32> = (0..(ndim - 1) as i32).collect();

    let (mean, var) = if training {
        // Compute batch mean
        let mut batch_mean = Array::new_uninit();
        let status = unsafe {
            mlx_sys::mlx_mean_axes(
                batch_mean.as_mut_ptr(),
                x.as_raw(),
                reduce_axes.as_ptr(),
                reduce_axes.len(),
                false,
                stream.as_raw(),
            )
        };
        if status != 0 {
            return Err(Error::ArrayCreation("Failed to compute batch mean".into()));
        }

        // Compute batch variance: E[(x - mean)^2]
        // First compute x - mean
        let mut x_centered = Array::new_uninit();
        let status = unsafe {
            mlx_sys::mlx_subtract(
                x_centered.as_mut_ptr(),
                x.as_raw(),
                batch_mean.as_raw(),
                stream.as_raw(),
            )
        };
        if status != 0 {
            return Err(Error::ArrayCreation("Failed to center input".into()));
        }

        // Square
        let mut x_squared = Array::new_uninit();
        let status = unsafe {
            mlx_sys::mlx_multiply(
                x_squared.as_mut_ptr(),
                x_centered.as_raw(),
                x_centered.as_raw(),
                stream.as_raw(),
            )
        };
        if status != 0 {
            return Err(Error::ArrayCreation("Failed to square centered input".into()));
        }

        // Mean of squares = variance
        let mut batch_var = Array::new_uninit();
        let status = unsafe {
            mlx_sys::mlx_mean_axes(
                batch_var.as_mut_ptr(),
                x_squared.as_raw(),
                reduce_axes.as_ptr(),
                reduce_axes.len(),
                false,
                stream.as_raw(),
            )
        };
        if status != 0 {
            return Err(Error::ArrayCreation("Failed to compute batch variance".into()));
        }

        (batch_mean, batch_var)
    } else {
        // Use running statistics
        let mean = running_mean
            .ok_or_else(|| Error::InvalidShape("running_mean required for inference".into()))?
            .clone();
        let var = running_var
            .ok_or_else(|| Error::InvalidShape("running_var required for inference".into()))?
            .clone();
        (mean, var)
    };

    // Normalize: (x - mean) / sqrt(var + eps)
    let mut x_centered = Array::new_uninit();
    let status = unsafe {
        mlx_sys::mlx_subtract(
            x_centered.as_mut_ptr(),
            x.as_raw(),
            mean.as_raw(),
            stream.as_raw(),
        )
    };
    if status != 0 {
        return Err(Error::ArrayCreation("Failed to center input".into()));
    }

    // var + eps
    let eps_arr = Array::from_float(eps);
    let mut var_eps = Array::new_uninit();
    let status = unsafe {
        mlx_sys::mlx_add(
            var_eps.as_mut_ptr(),
            var.as_raw(),
            eps_arr.as_raw(),
            stream.as_raw(),
        )
    };
    if status != 0 {
        return Err(Error::ArrayCreation("Failed to add eps".into()));
    }

    // sqrt(var + eps)
    let mut std = Array::new_uninit();
    let status = unsafe {
        mlx_sys::mlx_sqrt(std.as_mut_ptr(), var_eps.as_raw(), stream.as_raw())
    };
    if status != 0 {
        return Err(Error::ArrayCreation("Failed to compute std".into()));
    }

    // x_normalized = x_centered / std
    let mut x_norm = Array::new_uninit();
    let status = unsafe {
        mlx_sys::mlx_divide(
            x_norm.as_mut_ptr(),
            x_centered.as_raw(),
            std.as_raw(),
            stream.as_raw(),
        )
    };
    if status != 0 {
        return Err(Error::ArrayCreation("Failed to normalize".into()));
    }

    // Apply scale and shift: x_norm * weight + bias
    let mut scaled = Array::new_uninit();
    let status = unsafe {
        mlx_sys::mlx_multiply(
            scaled.as_mut_ptr(),
            x_norm.as_raw(),
            weight.as_raw(),
            stream.as_raw(),
        )
    };
    if status != 0 {
        return Err(Error::ArrayCreation("Failed to scale".into()));
    }

    let mut result = Array::new_uninit();
    let status = unsafe {
        mlx_sys::mlx_add(
            result.as_mut_ptr(),
            scaled.as_raw(),
            bias.as_raw(),
            stream.as_raw(),
        )
    };
    if status != 0 {
        return Err(Error::ArrayCreation("Failed to add bias".into()));
    }

    Ok(result)
}

/// Instance Normalization
///
/// Normalizes each instance independently over spatial dimensions.
/// For input of shape (N, H, W, C), normalizes over H, W for each (N, C).
///
/// # Arguments
/// * `x` - Input array of shape (N, H, W, C) for 2D or (N, L, C) for 1D
/// * `weight` - Scale parameter (gamma) of shape (C,)
/// * `bias` - Shift parameter (beta) of shape (C,)
/// * `eps` - Small constant for numerical stability (default: 1e-5)
///
/// # Returns
/// Normalized array
pub fn instance_norm(x: &Array, weight: &Array, bias: &Array, eps: f32) -> Result<Array> {
    let stream = Stream::default();
    let shape = x.shape();
    let ndim = shape.len();

    if ndim < 3 {
        return Err(Error::InvalidShape(
            "instance_norm requires at least 3D input (N, spatial..., C)".into(),
        ));
    }

    // Reduce over spatial dimensions (all except first and last)
    let reduce_axes: Vec<i32> = (1..(ndim - 1) as i32).collect();

    // Compute mean over spatial dimensions
    let mut mean = Array::new_uninit();
    let status = unsafe {
        mlx_sys::mlx_mean_axes(
            mean.as_mut_ptr(),
            x.as_raw(),
            reduce_axes.as_ptr(),
            reduce_axes.len(),
            true, // keepdims
            stream.as_raw(),
        )
    };
    if status != 0 {
        return Err(Error::ArrayCreation("Failed to compute instance mean".into()));
    }

    // Compute x - mean
    let mut x_centered = Array::new_uninit();
    let status = unsafe {
        mlx_sys::mlx_subtract(
            x_centered.as_mut_ptr(),
            x.as_raw(),
            mean.as_raw(),
            stream.as_raw(),
        )
    };
    if status != 0 {
        return Err(Error::ArrayCreation("Failed to center input".into()));
    }

    // Compute variance
    let mut x_squared = Array::new_uninit();
    let status = unsafe {
        mlx_sys::mlx_multiply(
            x_squared.as_mut_ptr(),
            x_centered.as_raw(),
            x_centered.as_raw(),
            stream.as_raw(),
        )
    };
    if status != 0 {
        return Err(Error::ArrayCreation("Failed to square".into()));
    }

    let mut var = Array::new_uninit();
    let status = unsafe {
        mlx_sys::mlx_mean_axes(
            var.as_mut_ptr(),
            x_squared.as_raw(),
            reduce_axes.as_ptr(),
            reduce_axes.len(),
            true, // keepdims
            stream.as_raw(),
        )
    };
    if status != 0 {
        return Err(Error::ArrayCreation("Failed to compute variance".into()));
    }

    // var + eps
    let eps_arr = Array::from_float(eps);
    let mut var_eps = Array::new_uninit();
    let status = unsafe {
        mlx_sys::mlx_add(
            var_eps.as_mut_ptr(),
            var.as_raw(),
            eps_arr.as_raw(),
            stream.as_raw(),
        )
    };
    if status != 0 {
        return Err(Error::ArrayCreation("Failed to add eps".into()));
    }

    // sqrt(var + eps)
    let mut std = Array::new_uninit();
    let status = unsafe {
        mlx_sys::mlx_sqrt(std.as_mut_ptr(), var_eps.as_raw(), stream.as_raw())
    };
    if status != 0 {
        return Err(Error::ArrayCreation("Failed to compute std".into()));
    }

    // Normalize
    let mut x_norm = Array::new_uninit();
    let status = unsafe {
        mlx_sys::mlx_divide(
            x_norm.as_mut_ptr(),
            x_centered.as_raw(),
            std.as_raw(),
            stream.as_raw(),
        )
    };
    if status != 0 {
        return Err(Error::ArrayCreation("Failed to normalize".into()));
    }

    // Apply scale and shift
    let mut scaled = Array::new_uninit();
    let status = unsafe {
        mlx_sys::mlx_multiply(
            scaled.as_mut_ptr(),
            x_norm.as_raw(),
            weight.as_raw(),
            stream.as_raw(),
        )
    };
    if status != 0 {
        return Err(Error::ArrayCreation("Failed to scale".into()));
    }

    let mut result = Array::new_uninit();
    let status = unsafe {
        mlx_sys::mlx_add(
            result.as_mut_ptr(),
            scaled.as_raw(),
            bias.as_raw(),
            stream.as_raw(),
        )
    };
    if status != 0 {
        return Err(Error::ArrayCreation("Failed to add bias".into()));
    }

    Ok(result)
}

/// Group Normalization
///
/// Divides channels into groups and normalizes within each group.
///
/// # Arguments
/// * `x` - Input array of shape (N, ..., C)
/// * `num_groups` - Number of groups to divide channels into
/// * `weight` - Scale parameter (gamma) of shape (C,)
/// * `bias` - Shift parameter (beta) of shape (C,)
/// * `eps` - Small constant for numerical stability (default: 1e-5)
///
/// # Returns
/// Normalized array
pub fn group_norm(
    x: &Array,
    num_groups: i32,
    weight: &Array,
    bias: &Array,
    eps: f32,
) -> Result<Array> {
    let stream = Stream::default();
    let shape = x.shape();
    let ndim = shape.len();

    if ndim < 2 {
        return Err(Error::InvalidShape(
            "group_norm requires at least 2D input".into(),
        ));
    }

    let channels = shape[ndim - 1];
    if channels % num_groups != 0 {
        return Err(Error::InvalidShape(
            format!("Number of channels ({}) must be divisible by num_groups ({})", channels, num_groups),
        ));
    }

    let group_size = channels / num_groups;

    // Reshape to separate groups: (..., num_groups, group_size)
    let mut new_shape: Vec<i32> = shape[..ndim - 1].to_vec();
    new_shape.push(num_groups);
    new_shape.push(group_size);

    let mut x_reshaped = Array::new_uninit();
    let status = unsafe {
        mlx_sys::mlx_reshape(
            x_reshaped.as_mut_ptr(),
            x.as_raw(),
            new_shape.as_ptr(),
            new_shape.len(),
            stream.as_raw(),
        )
    };
    if status != 0 {
        return Err(Error::ArrayCreation("Failed to reshape for group_norm".into()));
    }

    // Reduce over all dimensions except batch (0) and groups (ndim-1 in reshaped)
    // For reshaped array of shape (N, ..., num_groups, group_size), reduce over spatial dims and group_size
    let reshaped_ndim = new_shape.len();
    let mut reduce_axes: Vec<i32> = (1..(reshaped_ndim - 2) as i32).collect();
    reduce_axes.push((reshaped_ndim - 1) as i32); // Include group_size dimension

    // Compute mean
    let mut mean = Array::new_uninit();
    let status = unsafe {
        mlx_sys::mlx_mean_axes(
            mean.as_mut_ptr(),
            x_reshaped.as_raw(),
            reduce_axes.as_ptr(),
            reduce_axes.len(),
            true, // keepdims
            stream.as_raw(),
        )
    };
    if status != 0 {
        return Err(Error::ArrayCreation("Failed to compute group mean".into()));
    }

    // Compute variance
    let mut x_centered = Array::new_uninit();
    let status = unsafe {
        mlx_sys::mlx_subtract(
            x_centered.as_mut_ptr(),
            x_reshaped.as_raw(),
            mean.as_raw(),
            stream.as_raw(),
        )
    };
    if status != 0 {
        return Err(Error::ArrayCreation("Failed to center".into()));
    }

    let mut x_squared = Array::new_uninit();
    let status = unsafe {
        mlx_sys::mlx_multiply(
            x_squared.as_mut_ptr(),
            x_centered.as_raw(),
            x_centered.as_raw(),
            stream.as_raw(),
        )
    };
    if status != 0 {
        return Err(Error::ArrayCreation("Failed to square".into()));
    }

    let mut var = Array::new_uninit();
    let status = unsafe {
        mlx_sys::mlx_mean_axes(
            var.as_mut_ptr(),
            x_squared.as_raw(),
            reduce_axes.as_ptr(),
            reduce_axes.len(),
            true,
            stream.as_raw(),
        )
    };
    if status != 0 {
        return Err(Error::ArrayCreation("Failed to compute variance".into()));
    }

    // Normalize
    let eps_arr = Array::from_float(eps);
    let mut var_eps = Array::new_uninit();
    let status = unsafe {
        mlx_sys::mlx_add(var_eps.as_mut_ptr(), var.as_raw(), eps_arr.as_raw(), stream.as_raw())
    };
    if status != 0 {
        return Err(Error::ArrayCreation("Failed to add eps".into()));
    }

    let mut std = Array::new_uninit();
    let status = unsafe {
        mlx_sys::mlx_sqrt(std.as_mut_ptr(), var_eps.as_raw(), stream.as_raw())
    };
    if status != 0 {
        return Err(Error::ArrayCreation("Failed to compute std".into()));
    }

    let mut x_norm = Array::new_uninit();
    let status = unsafe {
        mlx_sys::mlx_divide(x_norm.as_mut_ptr(), x_centered.as_raw(), std.as_raw(), stream.as_raw())
    };
    if status != 0 {
        return Err(Error::ArrayCreation("Failed to normalize".into()));
    }

    // Reshape back to original shape
    let mut x_norm_reshaped = Array::new_uninit();
    let status = unsafe {
        mlx_sys::mlx_reshape(
            x_norm_reshaped.as_mut_ptr(),
            x_norm.as_raw(),
            shape.as_ptr(),
            shape.len(),
            stream.as_raw(),
        )
    };
    if status != 0 {
        return Err(Error::ArrayCreation("Failed to reshape back".into()));
    }

    // Apply scale and shift
    let mut scaled = Array::new_uninit();
    let status = unsafe {
        mlx_sys::mlx_multiply(
            scaled.as_mut_ptr(),
            x_norm_reshaped.as_raw(),
            weight.as_raw(),
            stream.as_raw(),
        )
    };
    if status != 0 {
        return Err(Error::ArrayCreation("Failed to scale".into()));
    }

    let mut result = Array::new_uninit();
    let status = unsafe {
        mlx_sys::mlx_add(result.as_mut_ptr(), scaled.as_raw(), bias.as_raw(), stream.as_raw())
    };
    if status != 0 {
        return Err(Error::ArrayCreation("Failed to add bias".into()));
    }

    Ok(result)
}

// ============================================================================
// Dropout Operations
// ============================================================================

/// Dropout
///
/// During training, randomly zeroes some elements of the input tensor with
/// probability `p` using samples from a Bernoulli distribution. The outputs
/// are scaled by a factor of `1/(1-p)` during training.
///
/// During evaluation (training=false), returns the input unchanged.
///
/// # Arguments
/// * `x` - Input array
/// * `p` - Probability of an element to be zeroed (default: 0.5)
/// * `training` - If true, applies dropout; if false, returns input unchanged
///
/// # Returns
/// Array with dropout applied (during training) or input unchanged (during eval)
///
/// # Example
/// ```ignore
/// let x = Array::from_slice(&[1.0f32, 2.0, 3.0, 4.0], &[2, 2])?;
/// let dropped = nn::dropout(&x, 0.5, true)?;  // Training mode
/// let same = nn::dropout(&x, 0.5, false)?;    // Eval mode (returns input)
/// ```
pub fn dropout(x: &Array, p: f32, training: bool) -> Result<Array> {
    if !training || p == 0.0 {
        return Ok(x.clone());
    }

    if p < 0.0 || p >= 1.0 {
        return Err(Error::InvalidShape(
            format!("Dropout probability must be in [0, 1), got {}", p),
        ));
    }

    let stream = Stream::default();
    let shape = x.shape();

    // Generate bernoulli mask with probability (1-p) of keeping elements
    let mask = random::bernoulli(&shape, 1.0 - p, None)?;

    // Scale factor: 1 / (1 - p)
    let scale = Array::from_float(1.0 / (1.0 - p));

    // Apply mask
    let mut masked = Array::new_uninit();
    let status = unsafe {
        mlx_sys::mlx_multiply(
            masked.as_mut_ptr(),
            x.as_raw(),
            mask.as_raw(),
            stream.as_raw(),
        )
    };
    if status != 0 {
        return Err(Error::ArrayCreation("Failed to apply dropout mask".into()));
    }

    // Scale
    let mut result = Array::new_uninit();
    let status = unsafe {
        mlx_sys::mlx_multiply(
            result.as_mut_ptr(),
            masked.as_raw(),
            scale.as_raw(),
            stream.as_raw(),
        )
    };
    if status != 0 {
        return Err(Error::ArrayCreation("Failed to scale dropout".into()));
    }

    Ok(result)
}

/// Dropout 2D (Spatial Dropout)
///
/// Randomly zeroes entire channels of the input tensor. For inputs of shape
/// (N, H, W, C) in channels-last format, each channel is zeroed entirely
/// with probability `p`.
///
/// This is more effective for convolutional layers where adjacent pixels
/// are highly correlated.
///
/// # Arguments
/// * `x` - Input array of shape (N, H, W, C) or (N, L, C) for 1D
/// * `p` - Probability of a channel to be zeroed (default: 0.5)
/// * `training` - If true, applies dropout; if false, returns input unchanged
///
/// # Returns
/// Array with channel dropout applied
pub fn dropout2d(x: &Array, p: f32, training: bool) -> Result<Array> {
    if !training || p == 0.0 {
        return Ok(x.clone());
    }

    if p < 0.0 || p >= 1.0 {
        return Err(Error::InvalidShape(
            format!("Dropout probability must be in [0, 1), got {}", p),
        ));
    }

    let stream = Stream::default();
    let shape = x.shape();
    let ndim = shape.len();

    if ndim < 3 {
        return Err(Error::InvalidShape(
            "dropout2d requires at least 3D input (N, spatial..., C)".into(),
        ));
    }

    // For channels-last format (N, ..., C), we want to drop entire channels
    // Create mask of shape (N, 1, ..., 1, C)
    let batch = shape[0];
    let channels = shape[ndim - 1];

    let mut mask_shape = vec![batch];
    for _ in 1..(ndim - 1) {
        mask_shape.push(1);
    }
    mask_shape.push(channels);

    // Generate bernoulli mask with probability (1-p) of keeping channels
    let mask = random::bernoulli(&mask_shape, 1.0 - p, None)?;

    // Scale factor: 1 / (1 - p)
    let scale = Array::from_float(1.0 / (1.0 - p));

    // Apply mask (broadcasts across spatial dimensions)
    let mut masked = Array::new_uninit();
    let status = unsafe {
        mlx_sys::mlx_multiply(
            masked.as_mut_ptr(),
            x.as_raw(),
            mask.as_raw(),
            stream.as_raw(),
        )
    };
    if status != 0 {
        return Err(Error::ArrayCreation("Failed to apply dropout2d mask".into()));
    }

    // Scale
    let mut result = Array::new_uninit();
    let status = unsafe {
        mlx_sys::mlx_multiply(
            result.as_mut_ptr(),
            masked.as_raw(),
            scale.as_raw(),
            stream.as_raw(),
        )
    };
    if status != 0 {
        return Err(Error::ArrayCreation("Failed to scale dropout2d".into()));
    }

    Ok(result)
}

/// Dropout 3D (Volumetric Dropout)
///
/// Randomly zeroes entire channels of the 5D input tensor. For inputs of shape
/// (N, D, H, W, C) in channels-last format, each channel is zeroed entirely
/// with probability `p`.
///
/// # Arguments
/// * `x` - Input array of shape (N, D, H, W, C)
/// * `p` - Probability of a channel to be zeroed (default: 0.5)
/// * `training` - If true, applies dropout; if false, returns input unchanged
///
/// # Returns
/// Array with channel dropout applied
pub fn dropout3d(x: &Array, p: f32, training: bool) -> Result<Array> {
    if !training || p == 0.0 {
        return Ok(x.clone());
    }

    if p < 0.0 || p >= 1.0 {
        return Err(Error::InvalidShape(
            format!("Dropout probability must be in [0, 1), got {}", p),
        ));
    }

    let shape = x.shape();
    let ndim = shape.len();

    if ndim != 5 {
        return Err(Error::InvalidShape(
            "dropout3d requires 5D input (N, D, H, W, C)".into(),
        ));
    }

    // Use dropout2d which handles the general case
    dropout2d(x, p, training)
}

/// Alpha Dropout
///
/// Applies Alpha Dropout over the input, designed for SELU activation.
/// Alpha Dropout maintains the self-normalizing property of SELU.
///
/// # Arguments
/// * `x` - Input array
/// * `p` - Probability of an element to be dropped (default: 0.5)
/// * `training` - If true, applies dropout; if false, returns input unchanged
///
/// # Returns
/// Array with alpha dropout applied
pub fn alpha_dropout(x: &Array, p: f32, training: bool) -> Result<Array> {
    if !training || p == 0.0 {
        return Ok(x.clone());
    }

    if p < 0.0 || p >= 1.0 {
        return Err(Error::InvalidShape(
            format!("Dropout probability must be in [0, 1), got {}", p),
        ));
    }

    let stream = Stream::default();
    let shape = x.shape();

    // SELU parameters
    let alpha: f32 = 1.6732632423543772;
    let scale: f32 = 1.0507009873554805;
    let alpha_p = -alpha * scale;

    // Affine transformation parameters to maintain mean and variance
    let a = ((1.0 - p) * (1.0 + p * alpha_p * alpha_p)).sqrt();
    let b = -a * alpha_p * p;

    // Generate bernoulli mask
    let mask = random::bernoulli(&shape, 1.0 - p, None)?;

    // Where mask is 0, replace with alpha_p (the saturation value)
    let alpha_p_arr = Array::from_float(alpha_p);
    let one = Array::from_float(1.0);

    // inverted_mask = 1 - mask
    let mut inverted_mask = Array::new_uninit();
    let status = unsafe {
        mlx_sys::mlx_subtract(
            inverted_mask.as_mut_ptr(),
            one.as_raw(),
            mask.as_raw(),
            stream.as_raw(),
        )
    };
    if status != 0 {
        return Err(Error::ArrayCreation("Failed to compute inverted mask".into()));
    }

    // masked_input = x * mask
    let mut masked_input = Array::new_uninit();
    let status = unsafe {
        mlx_sys::mlx_multiply(
            masked_input.as_mut_ptr(),
            x.as_raw(),
            mask.as_raw(),
            stream.as_raw(),
        )
    };
    if status != 0 {
        return Err(Error::ArrayCreation("Failed to apply mask".into()));
    }

    // saturation = alpha_p * inverted_mask
    let mut saturation = Array::new_uninit();
    let status = unsafe {
        mlx_sys::mlx_multiply(
            saturation.as_mut_ptr(),
            alpha_p_arr.as_raw(),
            inverted_mask.as_raw(),
            stream.as_raw(),
        )
    };
    if status != 0 {
        return Err(Error::ArrayCreation("Failed to compute saturation".into()));
    }

    // combined = masked_input + saturation
    let mut combined = Array::new_uninit();
    let status = unsafe {
        mlx_sys::mlx_add(
            combined.as_mut_ptr(),
            masked_input.as_raw(),
            saturation.as_raw(),
            stream.as_raw(),
        )
    };
    if status != 0 {
        return Err(Error::ArrayCreation("Failed to combine".into()));
    }

    // Apply affine transformation: a * combined + b
    let a_arr = Array::from_float(a);
    let b_arr = Array::from_float(b);

    let mut scaled = Array::new_uninit();
    let status = unsafe {
        mlx_sys::mlx_multiply(
            scaled.as_mut_ptr(),
            a_arr.as_raw(),
            combined.as_raw(),
            stream.as_raw(),
        )
    };
    if status != 0 {
        return Err(Error::ArrayCreation("Failed to scale".into()));
    }

    let mut result = Array::new_uninit();
    let status = unsafe {
        mlx_sys::mlx_add(
            result.as_mut_ptr(),
            scaled.as_raw(),
            b_arr.as_raw(),
            stream.as_raw(),
        )
    };
    if status != 0 {
        return Err(Error::ArrayCreation("Failed to add bias".into()));
    }

    Ok(result)
}

// ============================================================================
// Attention Operations
// ============================================================================

/// Mask mode for scaled dot-product attention
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum AttentionMask {
    /// No masking
    None,
    /// Causal masking (for autoregressive models)
    Causal,
    /// Custom mask provided as an array
    Custom,
}

impl AttentionMask {
    fn as_cstr(&self) -> *const std::ffi::c_char {
        match self {
            AttentionMask::None => b"\0".as_ptr() as *const std::ffi::c_char,
            AttentionMask::Causal => b"causal\0".as_ptr() as *const std::ffi::c_char,
            AttentionMask::Custom => b"array\0".as_ptr() as *const std::ffi::c_char,
        }
    }
}

/// Scaled Dot-Product Attention
///
/// Computes attention(Q, K, V) = softmax(Q @ K^T / sqrt(d_k)) @ V
///
/// This is the core attention mechanism used in Transformers.
///
/// # Arguments
/// * `queries` - Query tensor of shape (batch, num_heads, seq_len, head_dim)
/// * `keys` - Key tensor of shape (batch, num_heads, seq_len, head_dim)
/// * `values` - Value tensor of shape (batch, num_heads, seq_len, head_dim)
/// * `scale` - Scaling factor, typically 1/sqrt(head_dim). If None, uses 1/sqrt(head_dim)
/// * `mask` - Optional attention mask
/// * `mask_array` - Optional mask array (required if mask is Custom)
///
/// # Returns
/// Output tensor of shape (batch, num_heads, seq_len, head_dim)
///
/// # Example
/// ```ignore
/// let q = Array::zeros::<f32>(&[2, 8, 10, 64])?;  // (batch, heads, seq, dim)
/// let k = Array::zeros::<f32>(&[2, 8, 10, 64])?;
/// let v = Array::zeros::<f32>(&[2, 8, 10, 64])?;
///
/// // With automatic scaling
/// let out = nn::scaled_dot_product_attention(&q, &k, &v, None, AttentionMask::None, None)?;
///
/// // With causal masking for autoregressive models
/// let out = nn::scaled_dot_product_attention(&q, &k, &v, None, AttentionMask::Causal, None)?;
/// ```
pub fn scaled_dot_product_attention(
    queries: &Array,
    keys: &Array,
    values: &Array,
    scale: Option<f32>,
    mask: AttentionMask,
    mask_array: Option<&Array>,
) -> Result<Array> {
    let stream = Stream::default();
    let mut result = Array::new_uninit();

    let q_shape = queries.shape();
    if q_shape.len() < 2 {
        return Err(Error::InvalidShape(
            "Queries must have at least 2 dimensions".into(),
        ));
    }

    // Default scale is 1/sqrt(head_dim)
    let head_dim = q_shape[q_shape.len() - 1] as f32;
    let scale = scale.unwrap_or(1.0 / head_dim.sqrt());

    // Get mask array or create dummy
    let mask_arr = match (mask, mask_array) {
        (AttentionMask::Custom, Some(m)) => m.as_raw(),
        (AttentionMask::Custom, None) => {
            return Err(Error::InvalidShape(
                "Custom mask mode requires a mask array".into(),
            ));
        }
        _ => {
            // Create a dummy array for non-custom masks
            let dummy = Array::new_uninit();
            dummy.as_raw()
        }
    };

    // Create dummy sinks array (not commonly used)
    let sinks = Array::new_uninit();

    let status = unsafe {
        mlx_sys::mlx_fast_scaled_dot_product_attention(
            result.as_mut_ptr(),
            queries.as_raw(),
            keys.as_raw(),
            values.as_raw(),
            scale,
            mask.as_cstr(),
            mask_arr,
            sinks.as_raw(),
            stream.as_raw(),
        )
    };

    if status != 0 {
        return Err(Error::ArrayCreation(
            "Failed to compute scaled_dot_product_attention".into(),
        ));
    }

    Ok(result)
}

/// Multi-Head Attention
///
/// Applies multi-head attention mechanism.
///
/// # Arguments
/// * `query` - Query tensor of shape (batch, seq_len, embed_dim)
/// * `key` - Key tensor of shape (batch, seq_len, embed_dim)
/// * `value` - Value tensor of shape (batch, seq_len, embed_dim)
/// * `num_heads` - Number of attention heads
/// * `w_q` - Query projection weight of shape (embed_dim, embed_dim)
/// * `w_k` - Key projection weight of shape (embed_dim, embed_dim)
/// * `w_v` - Value projection weight of shape (embed_dim, embed_dim)
/// * `w_o` - Output projection weight of shape (embed_dim, embed_dim)
/// * `mask` - Attention mask mode
/// * `mask_array` - Optional custom mask array
///
/// # Returns
/// Output tensor of shape (batch, seq_len, embed_dim)
pub fn multi_head_attention(
    query: &Array,
    key: &Array,
    value: &Array,
    num_heads: i32,
    w_q: &Array,
    w_k: &Array,
    w_v: &Array,
    w_o: &Array,
    mask: AttentionMask,
    mask_array: Option<&Array>,
) -> Result<Array> {
    let stream = Stream::default();
    let q_shape = query.shape();

    if q_shape.len() != 3 {
        return Err(Error::InvalidShape(
            "Query must be 3D (batch, seq_len, embed_dim)".into(),
        ));
    }

    let batch_size = q_shape[0];
    let seq_len = q_shape[1];
    let embed_dim = q_shape[2];

    if embed_dim % num_heads != 0 {
        return Err(Error::InvalidShape(format!(
            "embed_dim ({}) must be divisible by num_heads ({})",
            embed_dim, num_heads
        )));
    }

    let head_dim = embed_dim / num_heads;

    // Project Q, K, V: (batch, seq, embed) @ (embed, embed) -> (batch, seq, embed)
    let mut q_proj = Array::new_uninit();
    let status = unsafe {
        mlx_sys::mlx_matmul(
            q_proj.as_mut_ptr(),
            query.as_raw(),
            w_q.as_raw(),
            stream.as_raw(),
        )
    };
    if status != 0 {
        return Err(Error::ArrayCreation("Failed to project queries".into()));
    }

    let mut k_proj = Array::new_uninit();
    let status = unsafe {
        mlx_sys::mlx_matmul(
            k_proj.as_mut_ptr(),
            key.as_raw(),
            w_k.as_raw(),
            stream.as_raw(),
        )
    };
    if status != 0 {
        return Err(Error::ArrayCreation("Failed to project keys".into()));
    }

    let mut v_proj = Array::new_uninit();
    let status = unsafe {
        mlx_sys::mlx_matmul(
            v_proj.as_mut_ptr(),
            value.as_raw(),
            w_v.as_raw(),
            stream.as_raw(),
        )
    };
    if status != 0 {
        return Err(Error::ArrayCreation("Failed to project values".into()));
    }

    // Reshape to (batch, seq, num_heads, head_dim)
    let reshape_to = [batch_size, seq_len, num_heads, head_dim];

    let mut q_reshaped = Array::new_uninit();
    let status = unsafe {
        mlx_sys::mlx_reshape(
            q_reshaped.as_mut_ptr(),
            q_proj.as_raw(),
            reshape_to.as_ptr(),
            reshape_to.len(),
            stream.as_raw(),
        )
    };
    if status != 0 {
        return Err(Error::ArrayCreation("Failed to reshape queries".into()));
    }

    let mut k_reshaped = Array::new_uninit();
    let status = unsafe {
        mlx_sys::mlx_reshape(
            k_reshaped.as_mut_ptr(),
            k_proj.as_raw(),
            reshape_to.as_ptr(),
            reshape_to.len(),
            stream.as_raw(),
        )
    };
    if status != 0 {
        return Err(Error::ArrayCreation("Failed to reshape keys".into()));
    }

    let mut v_reshaped = Array::new_uninit();
    let status = unsafe {
        mlx_sys::mlx_reshape(
            v_reshaped.as_mut_ptr(),
            v_proj.as_raw(),
            reshape_to.as_ptr(),
            reshape_to.len(),
            stream.as_raw(),
        )
    };
    if status != 0 {
        return Err(Error::ArrayCreation("Failed to reshape values".into()));
    }

    // Transpose to (batch, num_heads, seq, head_dim)
    let transpose_axes = [0i32, 2, 1, 3];

    let mut q_transposed = Array::new_uninit();
    let status = unsafe {
        mlx_sys::mlx_transpose_axes(
            q_transposed.as_mut_ptr(),
            q_reshaped.as_raw(),
            transpose_axes.as_ptr(),
            transpose_axes.len(),
            stream.as_raw(),
        )
    };
    if status != 0 {
        return Err(Error::ArrayCreation("Failed to transpose queries".into()));
    }

    let mut k_transposed = Array::new_uninit();
    let status = unsafe {
        mlx_sys::mlx_transpose_axes(
            k_transposed.as_mut_ptr(),
            k_reshaped.as_raw(),
            transpose_axes.as_ptr(),
            transpose_axes.len(),
            stream.as_raw(),
        )
    };
    if status != 0 {
        return Err(Error::ArrayCreation("Failed to transpose keys".into()));
    }

    let mut v_transposed = Array::new_uninit();
    let status = unsafe {
        mlx_sys::mlx_transpose_axes(
            v_transposed.as_mut_ptr(),
            v_reshaped.as_raw(),
            transpose_axes.as_ptr(),
            transpose_axes.len(),
            stream.as_raw(),
        )
    };
    if status != 0 {
        return Err(Error::ArrayCreation("Failed to transpose values".into()));
    }

    // Apply scaled dot-product attention
    let attn_output = scaled_dot_product_attention(
        &q_transposed,
        &k_transposed,
        &v_transposed,
        None,
        mask,
        mask_array,
    )?;

    // Transpose back to (batch, seq, num_heads, head_dim)
    let transpose_back = [0i32, 2, 1, 3];
    let mut attn_transposed = Array::new_uninit();
    let status = unsafe {
        mlx_sys::mlx_transpose_axes(
            attn_transposed.as_mut_ptr(),
            attn_output.as_raw(),
            transpose_back.as_ptr(),
            transpose_back.len(),
            stream.as_raw(),
        )
    };
    if status != 0 {
        return Err(Error::ArrayCreation("Failed to transpose attention output".into()));
    }

    // Reshape to (batch, seq, embed_dim)
    let reshape_back = [batch_size, seq_len, embed_dim];
    let mut attn_reshaped = Array::new_uninit();
    let status = unsafe {
        mlx_sys::mlx_reshape(
            attn_reshaped.as_mut_ptr(),
            attn_transposed.as_raw(),
            reshape_back.as_ptr(),
            reshape_back.len(),
            stream.as_raw(),
        )
    };
    if status != 0 {
        return Err(Error::ArrayCreation("Failed to reshape attention output".into()));
    }

    // Final projection
    let mut result = Array::new_uninit();
    let status = unsafe {
        mlx_sys::mlx_matmul(
            result.as_mut_ptr(),
            attn_reshaped.as_raw(),
            w_o.as_raw(),
            stream.as_raw(),
        )
    };
    if status != 0 {
        return Err(Error::ArrayCreation("Failed to project output".into()));
    }

    Ok(result)
}

/// Simplified Multi-Head Attention (without projection weights)
///
/// Applies multi-head attention directly on pre-projected Q, K, V tensors.
///
/// # Arguments
/// * `query` - Query tensor of shape (batch, seq_len, num_heads * head_dim)
/// * `key` - Key tensor of shape (batch, seq_len, num_heads * head_dim)
/// * `value` - Value tensor of shape (batch, seq_len, num_heads * head_dim)
/// * `num_heads` - Number of attention heads
/// * `mask` - Attention mask mode
/// * `mask_array` - Optional custom mask array
///
/// # Returns
/// Output tensor of shape (batch, seq_len, num_heads * head_dim)
pub fn multi_head_attention_simple(
    query: &Array,
    key: &Array,
    value: &Array,
    num_heads: i32,
    mask: AttentionMask,
    mask_array: Option<&Array>,
) -> Result<Array> {
    let stream = Stream::default();
    let q_shape = query.shape();

    if q_shape.len() != 3 {
        return Err(Error::InvalidShape(
            "Query must be 3D (batch, seq_len, embed_dim)".into(),
        ));
    }

    let batch_size = q_shape[0];
    let seq_len = q_shape[1];
    let embed_dim = q_shape[2];

    if embed_dim % num_heads != 0 {
        return Err(Error::InvalidShape(format!(
            "embed_dim ({}) must be divisible by num_heads ({})",
            embed_dim, num_heads
        )));
    }

    let head_dim = embed_dim / num_heads;

    // Reshape to (batch, seq, num_heads, head_dim)
    let reshape_to = [batch_size, seq_len, num_heads, head_dim];

    let mut q_reshaped = Array::new_uninit();
    let status = unsafe {
        mlx_sys::mlx_reshape(
            q_reshaped.as_mut_ptr(),
            query.as_raw(),
            reshape_to.as_ptr(),
            reshape_to.len(),
            stream.as_raw(),
        )
    };
    if status != 0 {
        return Err(Error::ArrayCreation("Failed to reshape queries".into()));
    }

    let mut k_reshaped = Array::new_uninit();
    let status = unsafe {
        mlx_sys::mlx_reshape(
            k_reshaped.as_mut_ptr(),
            key.as_raw(),
            reshape_to.as_ptr(),
            reshape_to.len(),
            stream.as_raw(),
        )
    };
    if status != 0 {
        return Err(Error::ArrayCreation("Failed to reshape keys".into()));
    }

    let mut v_reshaped = Array::new_uninit();
    let status = unsafe {
        mlx_sys::mlx_reshape(
            v_reshaped.as_mut_ptr(),
            value.as_raw(),
            reshape_to.as_ptr(),
            reshape_to.len(),
            stream.as_raw(),
        )
    };
    if status != 0 {
        return Err(Error::ArrayCreation("Failed to reshape values".into()));
    }

    // Transpose to (batch, num_heads, seq, head_dim)
    let transpose_axes = [0i32, 2, 1, 3];

    let mut q_transposed = Array::new_uninit();
    let status = unsafe {
        mlx_sys::mlx_transpose_axes(
            q_transposed.as_mut_ptr(),
            q_reshaped.as_raw(),
            transpose_axes.as_ptr(),
            transpose_axes.len(),
            stream.as_raw(),
        )
    };
    if status != 0 {
        return Err(Error::ArrayCreation("Failed to transpose queries".into()));
    }

    let mut k_transposed = Array::new_uninit();
    let status = unsafe {
        mlx_sys::mlx_transpose_axes(
            k_transposed.as_mut_ptr(),
            k_reshaped.as_raw(),
            transpose_axes.as_ptr(),
            transpose_axes.len(),
            stream.as_raw(),
        )
    };
    if status != 0 {
        return Err(Error::ArrayCreation("Failed to transpose keys".into()));
    }

    let mut v_transposed = Array::new_uninit();
    let status = unsafe {
        mlx_sys::mlx_transpose_axes(
            v_transposed.as_mut_ptr(),
            v_reshaped.as_raw(),
            transpose_axes.as_ptr(),
            transpose_axes.len(),
            stream.as_raw(),
        )
    };
    if status != 0 {
        return Err(Error::ArrayCreation("Failed to transpose values".into()));
    }

    // Apply scaled dot-product attention
    let attn_output = scaled_dot_product_attention(
        &q_transposed,
        &k_transposed,
        &v_transposed,
        None,
        mask,
        mask_array,
    )?;

    // Transpose back to (batch, seq, num_heads, head_dim)
    let transpose_back = [0i32, 2, 1, 3];
    let mut attn_transposed = Array::new_uninit();
    let status = unsafe {
        mlx_sys::mlx_transpose_axes(
            attn_transposed.as_mut_ptr(),
            attn_output.as_raw(),
            transpose_back.as_ptr(),
            transpose_back.len(),
            stream.as_raw(),
        )
    };
    if status != 0 {
        return Err(Error::ArrayCreation("Failed to transpose attention output".into()));
    }

    // Reshape to (batch, seq, embed_dim)
    let reshape_back = [batch_size, seq_len, embed_dim];
    let mut result = Array::new_uninit();
    let status = unsafe {
        mlx_sys::mlx_reshape(
            result.as_mut_ptr(),
            attn_transposed.as_raw(),
            reshape_back.as_ptr(),
            reshape_back.len(),
            stream.as_raw(),
        )
    };
    if status != 0 {
        return Err(Error::ArrayCreation("Failed to reshape output".into()));
    }

    Ok(result)
}

// ============================================================================
// Embedding Operations
// ============================================================================

/// Embedding lookup
///
/// Looks up embeddings from a weight matrix using indices.
/// This is the core operation for word embeddings in NLP models.
///
/// # Arguments
/// * `weight` - Embedding weight matrix of shape (num_embeddings, embedding_dim)
/// * `indices` - Integer indices of shape (...) into the embedding table
///
/// # Returns
/// Embeddings of shape (..., embedding_dim)
///
/// # Example
/// ```ignore
/// // Create embedding table with 1000 words, 256-dim embeddings
/// let weight = random::normal::<f32>(&[1000, 256], None)?;
///
/// // Look up embeddings for token indices [5, 10, 15]
/// let indices = Array::from_slice(&[5i32, 10, 15], &[3])?;
/// let embeddings = nn::embedding(&weight, &indices)?;
/// // embeddings.shape() == [3, 256]
/// ```
pub fn embedding(weight: &Array, indices: &Array) -> Result<Array> {
    let stream = Stream::default();
    let mut result = Array::new_uninit();

    // Use take_axis with axis=0 to get rows from the embedding table
    let status = unsafe {
        mlx_sys::mlx_take_axis(
            result.as_mut_ptr(),
            weight.as_raw(),
            indices.as_raw(),
            0,
            stream.as_raw(),
        )
    };

    if status != 0 {
        return Err(Error::ArrayCreation("Failed to compute embedding".into()));
    }

    Ok(result)
}

/// Embedding with padding index
///
/// Same as embedding, but zeros out the embedding at the padding index.
///
/// # Arguments
/// * `weight` - Embedding weight matrix of shape (num_embeddings, embedding_dim)
/// * `indices` - Integer indices of shape (...) into the embedding table
/// * `padding_idx` - Index that should return zeros
///
/// # Returns
/// Embeddings of shape (..., embedding_dim) with zeros at padding positions
pub fn embedding_with_padding(
    weight: &Array,
    indices: &Array,
    padding_idx: i32,
) -> Result<Array> {
    let stream = Stream::default();

    // Get embeddings
    let embeddings = embedding(weight, indices)?;

    // Create mask for non-padding indices
    let padding_arr = Array::from_int(padding_idx);
    let mut mask = Array::new_uninit();
    let status = unsafe {
        mlx_sys::mlx_not_equal(
            mask.as_mut_ptr(),
            indices.as_raw(),
            padding_arr.as_raw(),
            stream.as_raw(),
        )
    };
    if status != 0 {
        return Err(Error::ArrayCreation("Failed to create padding mask".into()));
    }

    // Expand mask to match embedding dimensions
    let mut mask_expanded = Array::new_uninit();
    let status = unsafe {
        mlx_sys::mlx_expand_dims(
            mask_expanded.as_mut_ptr(),
            mask.as_raw(),
            -1,
            stream.as_raw(),
        )
    };
    if status != 0 {
        return Err(Error::ArrayCreation("Failed to expand mask".into()));
    }

    // Apply mask (zeros out padding positions)
    let mut result = Array::new_uninit();
    let status = unsafe {
        mlx_sys::mlx_multiply(
            result.as_mut_ptr(),
            embeddings.as_raw(),
            mask_expanded.as_raw(),
            stream.as_raw(),
        )
    };
    if status != 0 {
        return Err(Error::ArrayCreation("Failed to apply padding mask".into()));
    }

    Ok(result)
}

/// Sinusoidal Positional Encoding
///
/// Creates sinusoidal positional encodings as described in "Attention Is All You Need".
/// Uses sin for even indices and cos for odd indices.
///
/// PE(pos, 2i) = sin(pos / 10000^(2i/d_model))
/// PE(pos, 2i+1) = cos(pos / 10000^(2i/d_model))
///
/// # Arguments
/// * `max_len` - Maximum sequence length
/// * `embed_dim` - Embedding dimension (must be even)
///
/// # Returns
/// Positional encoding matrix of shape (max_len, embed_dim)
pub fn sinusoidal_positional_encoding(max_len: i32, embed_dim: i32) -> Result<Array> {
    if embed_dim % 2 != 0 {
        return Err(Error::InvalidShape(
            "embed_dim must be even for sinusoidal encoding".into(),
        ));
    }

    let stream = Stream::default();

    // Create position indices: [0, 1, 2, ..., max_len-1]
    let positions = Array::arange::<f32>(0.0, max_len as f64, 1.0)?;

    // Create dimension indices: [0, 2, 4, ..., embed_dim-2]
    let dim_indices = Array::arange::<f32>(0.0, embed_dim as f64, 2.0)?;

    // Compute div_term = 10000^(2i/d_model) = exp(2i * -log(10000) / d_model)
    let log_10000 = (10000.0_f32).ln();
    let scale = -log_10000 / embed_dim as f32;
    let scale_arr = Array::from_float(scale);

    let mut scaled_dims = Array::new_uninit();
    let status = unsafe {
        mlx_sys::mlx_multiply(
            scaled_dims.as_mut_ptr(),
            dim_indices.as_raw(),
            scale_arr.as_raw(),
            stream.as_raw(),
        )
    };
    if status != 0 {
        return Err(Error::ArrayCreation("Failed to scale dimensions".into()));
    }

    let mut div_term = Array::new_uninit();
    let status = unsafe {
        mlx_sys::mlx_exp(div_term.as_mut_ptr(), scaled_dims.as_raw(), stream.as_raw())
    };
    if status != 0 {
        return Err(Error::ArrayCreation("Failed to compute div_term".into()));
    }

    // Expand dimensions for broadcasting
    // positions: (max_len,) -> (max_len, 1)
    let mut pos_expanded = Array::new_uninit();
    let status = unsafe {
        mlx_sys::mlx_expand_dims(
            pos_expanded.as_mut_ptr(),
            positions.as_raw(),
            1,
            stream.as_raw(),
        )
    };
    if status != 0 {
        return Err(Error::ArrayCreation("Failed to expand positions".into()));
    }

    // div_term: (embed_dim/2,) -> (1, embed_dim/2)
    let mut div_expanded = Array::new_uninit();
    let status = unsafe {
        mlx_sys::mlx_expand_dims(
            div_expanded.as_mut_ptr(),
            div_term.as_raw(),
            0,
            stream.as_raw(),
        )
    };
    if status != 0 {
        return Err(Error::ArrayCreation("Failed to expand div_term".into()));
    }

    // Compute pos * div_term: (max_len, embed_dim/2)
    let mut angles = Array::new_uninit();
    let status = unsafe {
        mlx_sys::mlx_multiply(
            angles.as_mut_ptr(),
            pos_expanded.as_raw(),
            div_expanded.as_raw(),
            stream.as_raw(),
        )
    };
    if status != 0 {
        return Err(Error::ArrayCreation("Failed to compute angles".into()));
    }

    // Compute sin and cos
    let mut sin_enc = Array::new_uninit();
    let status = unsafe {
        mlx_sys::mlx_sin(sin_enc.as_mut_ptr(), angles.as_raw(), stream.as_raw())
    };
    if status != 0 {
        return Err(Error::ArrayCreation("Failed to compute sin".into()));
    }

    let mut cos_enc = Array::new_uninit();
    let status = unsafe {
        mlx_sys::mlx_cos(cos_enc.as_mut_ptr(), angles.as_raw(), stream.as_raw())
    };
    if status != 0 {
        return Err(Error::ArrayCreation("Failed to compute cos".into()));
    }

    // Stack sin and cos and reshape to interleave
    // sin_enc: (max_len, embed_dim/2), cos_enc: (max_len, embed_dim/2)
    // We want: (max_len, embed_dim) where even indices are sin, odd are cos

    // Expand dims for stacking
    let mut sin_expanded = Array::new_uninit();
    let status = unsafe {
        mlx_sys::mlx_expand_dims(
            sin_expanded.as_mut_ptr(),
            sin_enc.as_raw(),
            2,
            stream.as_raw(),
        )
    };
    if status != 0 {
        return Err(Error::ArrayCreation("Failed to expand sin".into()));
    }

    let mut cos_expanded = Array::new_uninit();
    let status = unsafe {
        mlx_sys::mlx_expand_dims(
            cos_expanded.as_mut_ptr(),
            cos_enc.as_raw(),
            2,
            stream.as_raw(),
        )
    };
    if status != 0 {
        return Err(Error::ArrayCreation("Failed to expand cos".into()));
    }

    // Concatenate along last dimension
    let raw_arrays = [sin_expanded.as_raw(), cos_expanded.as_raw()];
    let vec = unsafe { mlx_sys::mlx_vector_array_new_data(raw_arrays.as_ptr(), 2) };

    let mut stacked = Array::new_uninit();
    let status = unsafe {
        mlx_sys::mlx_concatenate_axis(
            stacked.as_mut_ptr(),
            vec,
            2,
            stream.as_raw(),
        )
    };
    unsafe { mlx_sys::mlx_vector_array_free(vec) };

    if status != 0 {
        return Err(Error::ArrayCreation("Failed to concatenate sin/cos".into()));
    }

    // Reshape to (max_len, embed_dim)
    let new_shape = [max_len, embed_dim];
    let mut result = Array::new_uninit();
    let status = unsafe {
        mlx_sys::mlx_reshape(
            result.as_mut_ptr(),
            stacked.as_raw(),
            new_shape.as_ptr(),
            new_shape.len(),
            stream.as_raw(),
        )
    };
    if status != 0 {
        return Err(Error::ArrayCreation("Failed to reshape positional encoding".into()));
    }

    Ok(result)
}

/// Learned Positional Embedding
///
/// Creates a learnable positional embedding table.
/// This is an alternative to sinusoidal encodings used in models like BERT and GPT.
///
/// # Arguments
/// * `max_len` - Maximum sequence length
/// * `embed_dim` - Embedding dimension
///
/// # Returns
/// Positional embedding matrix of shape (max_len, embed_dim) initialized with normal distribution
pub fn learned_positional_embedding(max_len: i32, embed_dim: i32) -> Result<Array> {
    // Initialize with normal distribution (std = 0.02 is common)
    crate::random::normal_with_params::<f32>(&[max_len, embed_dim], 0.0, 0.02, None)
}

/// Add positional encoding to input embeddings
///
/// # Arguments
/// * `embeddings` - Input embeddings of shape (batch, seq_len, embed_dim)
/// * `positional_encoding` - Positional encoding of shape (max_len, embed_dim)
///
/// # Returns
/// Embeddings with positional information added
pub fn add_positional_encoding(
    embeddings: &Array,
    positional_encoding: &Array,
) -> Result<Array> {
    let stream = Stream::default();
    let shape = embeddings.shape();

    if shape.len() != 3 {
        return Err(Error::InvalidShape(
            "Embeddings must be 3D (batch, seq_len, embed_dim)".into(),
        ));
    }

    let seq_len = shape[1];

    // Slice positional encoding to match sequence length
    let start = [0, 0];
    let stop = [seq_len, positional_encoding.shape()[1]];
    let pe_sliced = positional_encoding.slice(&start, &stop, None)?;

    // Add positional encoding (broadcasts over batch dimension)
    let mut result = Array::new_uninit();
    let status = unsafe {
        mlx_sys::mlx_add(
            result.as_mut_ptr(),
            embeddings.as_raw(),
            pe_sliced.as_raw(),
            stream.as_raw(),
        )
    };
    if status != 0 {
        return Err(Error::ArrayCreation("Failed to add positional encoding".into()));
    }

    Ok(result)
}

/// Rotary Positional Embedding (RoPE)
///
/// Applies rotary positional embeddings as described in "RoFormer: Enhanced Transformer
/// with Rotary Position Embedding". This is used in models like LLaMA.
///
/// # Arguments
/// * `x` - Input tensor of shape (batch, num_heads, seq_len, head_dim)
/// * `cos` - Precomputed cosine values of shape (seq_len, head_dim/2)
/// * `sin` - Precomputed sine values of shape (seq_len, head_dim/2)
///
/// # Returns
/// Tensor with rotary positional embeddings applied
pub fn apply_rotary_embedding(
    x: &Array,
    cos: &Array,
    sin: &Array,
) -> Result<Array> {
    let stream = Stream::default();
    let shape = x.shape();

    if shape.len() != 4 {
        return Err(Error::InvalidShape(
            "Input must be 4D (batch, num_heads, seq_len, head_dim)".into(),
        ));
    }

    let head_dim = shape[3];
    if head_dim % 2 != 0 {
        return Err(Error::InvalidShape(
            "head_dim must be even for rotary embeddings".into(),
        ));
    }

    let half_dim = head_dim / 2;

    // Split x into first half and second half along head_dim
    let start1 = [0, 0, 0, 0];
    let stop1 = [shape[0], shape[1], shape[2], half_dim];
    let x1 = x.slice(&start1, &stop1, None)?;

    let start2 = [0, 0, 0, half_dim];
    let stop2 = [shape[0], shape[1], shape[2], head_dim];
    let x2 = x.slice(&start2, &stop2, None)?;

    // Apply rotation:
    // x_rotated = [x1 * cos - x2 * sin, x1 * sin + x2 * cos]

    // x1 * cos
    let mut x1_cos = Array::new_uninit();
    let status = unsafe {
        mlx_sys::mlx_multiply(x1_cos.as_mut_ptr(), x1.as_raw(), cos.as_raw(), stream.as_raw())
    };
    if status != 0 {
        return Err(Error::ArrayCreation("Failed to compute x1*cos".into()));
    }

    // x2 * sin
    let mut x2_sin = Array::new_uninit();
    let status = unsafe {
        mlx_sys::mlx_multiply(x2_sin.as_mut_ptr(), x2.as_raw(), sin.as_raw(), stream.as_raw())
    };
    if status != 0 {
        return Err(Error::ArrayCreation("Failed to compute x2*sin".into()));
    }

    // x1 * sin
    let mut x1_sin = Array::new_uninit();
    let status = unsafe {
        mlx_sys::mlx_multiply(x1_sin.as_mut_ptr(), x1.as_raw(), sin.as_raw(), stream.as_raw())
    };
    if status != 0 {
        return Err(Error::ArrayCreation("Failed to compute x1*sin".into()));
    }

    // x2 * cos
    let mut x2_cos = Array::new_uninit();
    let status = unsafe {
        mlx_sys::mlx_multiply(x2_cos.as_mut_ptr(), x2.as_raw(), cos.as_raw(), stream.as_raw())
    };
    if status != 0 {
        return Err(Error::ArrayCreation("Failed to compute x2*cos".into()));
    }

    // First half: x1 * cos - x2 * sin
    let mut first_half = Array::new_uninit();
    let status = unsafe {
        mlx_sys::mlx_subtract(first_half.as_mut_ptr(), x1_cos.as_raw(), x2_sin.as_raw(), stream.as_raw())
    };
    if status != 0 {
        return Err(Error::ArrayCreation("Failed to compute first half".into()));
    }

    // Second half: x1 * sin + x2 * cos
    let mut second_half = Array::new_uninit();
    let status = unsafe {
        mlx_sys::mlx_add(second_half.as_mut_ptr(), x1_sin.as_raw(), x2_cos.as_raw(), stream.as_raw())
    };
    if status != 0 {
        return Err(Error::ArrayCreation("Failed to compute second half".into()));
    }

    // Concatenate halves
    let raw_arrays = [first_half.as_raw(), second_half.as_raw()];
    let vec = unsafe { mlx_sys::mlx_vector_array_new_data(raw_arrays.as_ptr(), 2) };

    let mut result = Array::new_uninit();
    let status = unsafe {
        mlx_sys::mlx_concatenate_axis(
            result.as_mut_ptr(),
            vec,
            3,  // concatenate along head_dim axis
            stream.as_raw(),
        )
    };
    unsafe { mlx_sys::mlx_vector_array_free(vec) };

    if status != 0 {
        return Err(Error::ArrayCreation("Failed to concatenate rotary result".into()));
    }

    Ok(result)
}

/// Precompute cosine and sine values for Rotary Positional Embedding
///
/// # Arguments
/// * `max_len` - Maximum sequence length
/// * `head_dim` - Head dimension (must be even)
/// * `base` - Base for computing frequencies (default: 10000.0)
///
/// # Returns
/// Tuple of (cos, sin) arrays each of shape (max_len, head_dim/2)
pub fn precompute_rope_frequencies(
    max_len: i32,
    head_dim: i32,
    base: f32,
) -> Result<(Array, Array)> {
    if head_dim % 2 != 0 {
        return Err(Error::InvalidShape(
            "head_dim must be even for rotary embeddings".into(),
        ));
    }

    let stream = Stream::default();
    let half_dim = head_dim / 2;

    // Compute frequencies: 1 / (base^(2i/dim))
    let dim_indices = Array::arange::<f32>(0.0, half_dim as f64, 1.0)?;
    let two = Array::from_float(2.0);
    let dim_float = Array::from_float(head_dim as f32);

    // 2 * i / dim
    let mut scaled = Array::new_uninit();
    let status = unsafe {
        mlx_sys::mlx_multiply(scaled.as_mut_ptr(), two.as_raw(), dim_indices.as_raw(), stream.as_raw())
    };
    if status != 0 {
        return Err(Error::ArrayCreation("Failed to scale indices".into()));
    }

    let mut exponent = Array::new_uninit();
    let status = unsafe {
        mlx_sys::mlx_divide(exponent.as_mut_ptr(), scaled.as_raw(), dim_float.as_raw(), stream.as_raw())
    };
    if status != 0 {
        return Err(Error::ArrayCreation("Failed to compute exponent".into()));
    }

    // base^exponent
    let base_arr = Array::from_float(base);
    let mut base_pow = Array::new_uninit();
    let status = unsafe {
        mlx_sys::mlx_power(base_pow.as_mut_ptr(), base_arr.as_raw(), exponent.as_raw(), stream.as_raw())
    };
    if status != 0 {
        return Err(Error::ArrayCreation("Failed to compute base^exponent".into()));
    }

    // frequencies = 1 / base^exponent
    let one = Array::from_float(1.0);
    let mut freqs = Array::new_uninit();
    let status = unsafe {
        mlx_sys::mlx_divide(freqs.as_mut_ptr(), one.as_raw(), base_pow.as_raw(), stream.as_raw())
    };
    if status != 0 {
        return Err(Error::ArrayCreation("Failed to compute frequencies".into()));
    }

    // Position indices
    let positions = Array::arange::<f32>(0.0, max_len as f64, 1.0)?;

    // Expand for broadcasting: positions (max_len, 1), freqs (1, half_dim)
    let mut pos_expanded = Array::new_uninit();
    let status = unsafe {
        mlx_sys::mlx_expand_dims(pos_expanded.as_mut_ptr(), positions.as_raw(), 1, stream.as_raw())
    };
    if status != 0 {
        return Err(Error::ArrayCreation("Failed to expand positions".into()));
    }

    let mut freqs_expanded = Array::new_uninit();
    let status = unsafe {
        mlx_sys::mlx_expand_dims(freqs_expanded.as_mut_ptr(), freqs.as_raw(), 0, stream.as_raw())
    };
    if status != 0 {
        return Err(Error::ArrayCreation("Failed to expand frequencies".into()));
    }

    // angles = positions * frequencies
    let mut angles = Array::new_uninit();
    let status = unsafe {
        mlx_sys::mlx_multiply(angles.as_mut_ptr(), pos_expanded.as_raw(), freqs_expanded.as_raw(), stream.as_raw())
    };
    if status != 0 {
        return Err(Error::ArrayCreation("Failed to compute angles".into()));
    }

    // Compute cos and sin
    let mut cos = Array::new_uninit();
    let status = unsafe {
        mlx_sys::mlx_cos(cos.as_mut_ptr(), angles.as_raw(), stream.as_raw())
    };
    if status != 0 {
        return Err(Error::ArrayCreation("Failed to compute cos".into()));
    }

    let mut sin = Array::new_uninit();
    let status = unsafe {
        mlx_sys::mlx_sin(sin.as_mut_ptr(), angles.as_raw(), stream.as_raw())
    };
    if status != 0 {
        return Err(Error::ArrayCreation("Failed to compute sin".into()));
    }

    Ok((cos, sin))
}

// ============================================================================
// Additional Loss Functions
// ============================================================================

/// Huber loss (Smooth L1 loss)
///
/// Combines MSE and MAE for robustness to outliers:
/// - For |error| <= delta: 0.5 * error^2
/// - For |error| > delta: delta * (|error| - 0.5 * delta)
///
/// # Arguments
/// * `predictions` - Model predictions
/// * `targets` - Ground truth values
/// * `delta` - Threshold for switching between L1 and L2 (default 1.0)
/// * `reduction` - How to reduce the loss ("mean", "sum", "none")
pub fn huber_loss(predictions: &Array, targets: &Array, delta: f32, reduction: &str) -> Result<Array> {
    let _stream = Stream::default();

    // diff = predictions - targets
    let diff = predictions - targets;

    // abs_diff = |diff|
    let abs_diff = diff.abs()?;

    // Create constants
    let delta_arr = Array::from_float(delta);
    let half = Array::from_float(0.5);
    let half_delta = Array::from_float(0.5 * delta);

    // squared_loss = 0.5 * diff^2
    let squared = &diff * &diff;
    let squared_loss = &half * &squared;

    // linear_loss = delta * (|diff| - 0.5 * delta)
    let diff_minus_half_delta = &abs_diff - &half_delta;
    let linear_loss = &delta_arr * &diff_minus_half_delta;

    // condition = abs_diff <= delta
    let condition = abs_diff.le(&delta_arr)?;

    // loss = where(condition, squared_loss, linear_loss)
    let loss = crate::ops::where_cond(&condition, &squared_loss, &linear_loss)?;

    match reduction {
        "mean" => loss.mean_all(false),
        "sum" => loss.sum_all(false),
        "none" => Ok(loss),
        _ => Err(Error::NotSupported(format!("Unknown reduction: {}", reduction))),
    }
}

/// Smooth L1 loss (same as Huber with delta=1.0)
///
/// # Arguments
/// * `predictions` - Model predictions
/// * `targets` - Ground truth values
/// * `beta` - Threshold (same as delta in Huber loss)
/// * `reduction` - How to reduce the loss ("mean", "sum", "none")
pub fn smooth_l1_loss(predictions: &Array, targets: &Array, beta: f32, reduction: &str) -> Result<Array> {
    huber_loss(predictions, targets, beta, reduction)
}

/// Binary Cross-Entropy with logits
///
/// More numerically stable version that takes logits (pre-sigmoid values):
/// BCE = max(x, 0) - x * targets + log(1 + exp(-|x|))
///
/// # Arguments
/// * `logits` - Model outputs (logits, not probabilities)
/// * `targets` - Ground truth labels (0 or 1)
/// * `reduction` - How to reduce the loss ("mean", "sum", "none")
pub fn binary_cross_entropy_with_logits(logits: &Array, targets: &Array, reduction: &str) -> Result<Array> {
    let stream = Stream::default();
    let zero = Array::from_float(0.0);
    let one = Array::from_float(1.0);

    // max_val = max(logits, 0)
    let mut max_val = Array::new_uninit();
    let status = unsafe {
        mlx_sys::mlx_maximum(max_val.as_mut_ptr(), logits.as_raw(), zero.as_raw(), stream.as_raw())
    };
    if status != 0 {
        return Err(Error::ArrayCreation("Failed to compute maximum".into()));
    }

    // neg_abs_logits = -|logits|
    let abs_logits = logits.abs()?;
    let neg_abs_logits = -&abs_logits;

    // exp_neg_abs = exp(-|logits|)
    let exp_neg_abs = neg_abs_logits.exp()?;

    // log_term = log(1 + exp(-|logits|))
    let one_plus_exp = &one + &exp_neg_abs;
    let log_term = one_plus_exp.log()?;

    // x_times_targets = logits * targets
    let x_times_targets = logits * targets;

    // loss = max_val - x * targets + log_term
    let loss = &(&max_val - &x_times_targets) + &log_term;

    match reduction {
        "mean" => loss.mean_all(false),
        "sum" => loss.sum_all(false),
        "none" => Ok(loss),
        _ => Err(Error::NotSupported(format!("Unknown reduction: {}", reduction))),
    }
}

/// Cross-Entropy loss for multi-class classification
///
/// Computes cross-entropy between predictions and one-hot targets:
/// CE = -sum(targets * log(predictions))
///
/// # Arguments
/// * `predictions` - Model predictions (softmax probabilities)
/// * `targets` - One-hot encoded ground truth
/// * `reduction` - How to reduce the loss ("mean", "sum", "none")
///
/// # Note
/// For sparse targets (class indices), use `nll_loss` with log_softmax outputs
pub fn cross_entropy_loss(predictions: &Array, targets: &Array, reduction: &str) -> Result<Array> {
    let stream = Stream::default();
    let eps = Array::from_float(1e-7);
    let one = Array::from_float(1.0);

    // Clamp predictions for numerical stability using max/min
    let mut pred_max = Array::new_uninit();
    let status = unsafe {
        mlx_sys::mlx_maximum(pred_max.as_mut_ptr(), predictions.as_raw(), eps.as_raw(), stream.as_raw())
    };
    if status != 0 {
        return Err(Error::ArrayCreation("Failed to compute maximum".into()));
    }

    let mut pred_clamped = Array::new_uninit();
    let status = unsafe {
        mlx_sys::mlx_minimum(pred_clamped.as_mut_ptr(), pred_max.as_raw(), one.as_raw(), stream.as_raw())
    };
    if status != 0 {
        return Err(Error::ArrayCreation("Failed to compute minimum".into()));
    }

    // log_pred = log(pred_clamped)
    let log_pred = pred_clamped.log()?;

    // ce = -sum(targets * log(pred), axis=-1)
    let targets_times_log = targets * &log_pred;

    // Sum along class dimension (axis -1) and negate
    let ndim = targets_times_log.ndim();
    let axis = (ndim as i32) - 1;
    let summed = targets_times_log.sum_axes(&[axis], false)?;
    let loss = -&summed;

    match reduction {
        "mean" => loss.mean_all(false),
        "sum" => loss.sum_all(false),
        "none" => Ok(loss),
        _ => Err(Error::NotSupported(format!("Unknown reduction: {}", reduction))),
    }
}

/// Hinge loss for binary classification (SVM-style)
///
/// Computes: max(0, margin - y * predictions) where y in {-1, +1}
///
/// # Arguments
/// * `predictions` - Model predictions
/// * `targets` - Ground truth labels (-1 or +1)
/// * `margin` - Margin value (default 1.0)
/// * `reduction` - How to reduce the loss ("mean", "sum", "none")
pub fn hinge_loss(predictions: &Array, targets: &Array, margin: f32, reduction: &str) -> Result<Array> {
    let stream = Stream::default();
    let margin_arr = Array::from_float(margin);
    let zero = Array::from_float(0.0);

    // y_times_pred = targets * predictions
    let y_times_pred = targets * predictions;

    // margin_minus = margin - y * pred
    let margin_minus = &margin_arr - &y_times_pred;

    // loss = max(0, margin - y * pred)
    let mut loss = Array::new_uninit();
    let status = unsafe {
        mlx_sys::mlx_maximum(loss.as_mut_ptr(), margin_minus.as_raw(), zero.as_raw(), stream.as_raw())
    };
    if status != 0 {
        return Err(Error::ArrayCreation("Failed to compute maximum".into()));
    }

    match reduction {
        "mean" => loss.mean_all(false),
        "sum" => loss.sum_all(false),
        "none" => Ok(loss),
        _ => Err(Error::NotSupported(format!("Unknown reduction: {}", reduction))),
    }
}

/// Triplet Margin Loss
///
/// For learning embeddings where similar items are closer:
/// loss = max(0, d(anchor, positive) - d(anchor, negative) + margin)
///
/// # Arguments
/// * `anchor` - Anchor embeddings
/// * `positive` - Positive (similar) embeddings
/// * `negative` - Negative (dissimilar) embeddings
/// * `margin` - Margin value (default 1.0)
/// * `reduction` - How to reduce the loss ("mean", "sum", "none")
pub fn triplet_margin_loss(anchor: &Array, positive: &Array, negative: &Array, margin: f32, reduction: &str) -> Result<Array> {
    let stream = Stream::default();
    let margin_arr = Array::from_float(margin);
    let zero = Array::from_float(0.0);

    // Compute squared L2 distances
    // d(a, p)^2 = sum((a - p)^2, axis=-1)
    let diff_pos = anchor - positive;
    let diff_pos_sq = &diff_pos * &diff_pos;
    let ndim = diff_pos_sq.ndim();
    let axis = (ndim as i32) - 1;
    let dist_pos = diff_pos_sq.sum_axes(&[axis], false)?;

    // d(a, n)^2 = sum((a - n)^2, axis=-1)
    let diff_neg = anchor - negative;
    let diff_neg_sq = &diff_neg * &diff_neg;
    let dist_neg = diff_neg_sq.sum_axes(&[axis], false)?;

    // loss = max(0, d_pos - d_neg + margin)
    let diff_dist = &dist_pos - &dist_neg;
    let diff_plus_margin = &diff_dist + &margin_arr;

    let mut loss = Array::new_uninit();
    let status = unsafe {
        mlx_sys::mlx_maximum(loss.as_mut_ptr(), diff_plus_margin.as_raw(), zero.as_raw(), stream.as_raw())
    };
    if status != 0 {
        return Err(Error::ArrayCreation("Failed to compute maximum".into()));
    }

    match reduction {
        "mean" => loss.mean_all(false),
        "sum" => loss.sum_all(false),
        "none" => Ok(loss),
        _ => Err(Error::NotSupported(format!("Unknown reduction: {}", reduction))),
    }
}

// ============================================================================
// Optimizers
// ============================================================================

/// SGD (Stochastic Gradient Descent) optimizer step
///
/// Updates parameters using: param = param - lr * grad
///
/// # Arguments
/// * `param` - Parameter array to update
/// * `grad` - Gradient array
/// * `lr` - Learning rate
///
/// # Returns
/// Updated parameter array
pub fn sgd_step(param: &Array, grad: &Array, lr: f32) -> Result<Array> {
    let lr_arr = Array::from_float(lr);
    let update = &lr_arr * grad;
    Ok(param - &update)
}

/// SGD with momentum optimizer step
///
/// Updates parameters using:
/// - velocity = momentum * velocity + grad
/// - param = param - lr * velocity
///
/// # Arguments
/// * `param` - Parameter array to update
/// * `grad` - Gradient array
/// * `velocity` - Velocity (momentum) state
/// * `lr` - Learning rate
/// * `momentum` - Momentum coefficient (typically 0.9)
///
/// # Returns
/// Tuple of (updated_param, updated_velocity)
pub fn sgd_momentum_step(
    param: &Array,
    grad: &Array,
    velocity: &Array,
    lr: f32,
    momentum: f32,
) -> Result<(Array, Array)> {
    let lr_arr = Array::from_float(lr);
    let momentum_arr = Array::from_float(momentum);

    // velocity = momentum * velocity + grad
    let new_velocity = &(&momentum_arr * velocity) + grad;

    // param = param - lr * velocity
    let update = &lr_arr * &new_velocity;
    let new_param = param - &update;

    Ok((new_param, new_velocity))
}

/// SGD with momentum and weight decay (L2 regularization)
///
/// Updates parameters using:
/// - grad_with_decay = grad + weight_decay * param
/// - velocity = momentum * velocity + grad_with_decay
/// - param = param - lr * velocity
///
/// # Arguments
/// * `param` - Parameter array to update
/// * `grad` - Gradient array
/// * `velocity` - Velocity (momentum) state
/// * `lr` - Learning rate
/// * `momentum` - Momentum coefficient (typically 0.9)
/// * `weight_decay` - Weight decay coefficient (L2 penalty)
///
/// # Returns
/// Tuple of (updated_param, updated_velocity)
pub fn sgd_weight_decay_step(
    param: &Array,
    grad: &Array,
    velocity: &Array,
    lr: f32,
    momentum: f32,
    weight_decay: f32,
) -> Result<(Array, Array)> {
    let lr_arr = Array::from_float(lr);
    let momentum_arr = Array::from_float(momentum);
    let wd_arr = Array::from_float(weight_decay);

    // Apply weight decay to gradient
    let grad_with_decay = grad + &(&wd_arr * param);

    // velocity = momentum * velocity + grad_with_decay
    let new_velocity = &(&momentum_arr * velocity) + &grad_with_decay;

    // param = param - lr * velocity
    let update = &lr_arr * &new_velocity;
    let new_param = param - &update;

    Ok((new_param, new_velocity))
}

/// Adam optimizer step
///
/// Updates parameters using the Adam algorithm:
/// - m = beta1 * m + (1 - beta1) * grad
/// - v = beta2 * v + (1 - beta2) * grad^2
/// - m_hat = m / (1 - beta1^t)
/// - v_hat = v / (1 - beta2^t)
/// - param = param - lr * m_hat / (sqrt(v_hat) + eps)
///
/// # Arguments
/// * `param` - Parameter array to update
/// * `grad` - Gradient array
/// * `m` - First moment estimate (mean of gradients)
/// * `v` - Second moment estimate (variance of gradients)
/// * `lr` - Learning rate
/// * `beta1` - Exponential decay rate for first moment (typically 0.9)
/// * `beta2` - Exponential decay rate for second moment (typically 0.999)
/// * `eps` - Small constant for numerical stability (typically 1e-8)
/// * `t` - Current timestep (starting from 1)
///
/// # Returns
/// Tuple of (updated_param, updated_m, updated_v)
pub fn adam_step(
    param: &Array,
    grad: &Array,
    m: &Array,
    v: &Array,
    lr: f32,
    beta1: f32,
    beta2: f32,
    eps: f32,
    t: i32,
) -> Result<(Array, Array, Array)> {
    let stream = Stream::default();

    let lr_arr = Array::from_float(lr);
    let beta1_arr = Array::from_float(beta1);
    let beta2_arr = Array::from_float(beta2);
    let one_minus_beta1 = Array::from_float(1.0 - beta1);
    let one_minus_beta2 = Array::from_float(1.0 - beta2);
    let eps_arr = Array::from_float(eps);

    // Bias correction terms
    let beta1_t = beta1.powi(t);
    let beta2_t = beta2.powi(t);
    let bias_correction1 = Array::from_float(1.0 / (1.0 - beta1_t));
    let bias_correction2 = Array::from_float(1.0 / (1.0 - beta2_t));

    // m = beta1 * m + (1 - beta1) * grad
    let new_m = &(&beta1_arr * m) + &(&one_minus_beta1 * grad);

    // v = beta2 * v + (1 - beta2) * grad^2
    let grad_sq = grad * grad;
    let new_v = &(&beta2_arr * v) + &(&one_minus_beta2 * &grad_sq);

    // Bias-corrected estimates
    let m_hat = &new_m * &bias_correction1;
    let v_hat = &new_v * &bias_correction2;

    // param = param - lr * m_hat / (sqrt(v_hat) + eps)
    let v_hat_sqrt = v_hat.sqrt()?;
    let denom = &v_hat_sqrt + &eps_arr;

    let mut update = Array::new_uninit();
    let status = unsafe {
        mlx_sys::mlx_divide(update.as_mut_ptr(), m_hat.as_raw(), denom.as_raw(), stream.as_raw())
    };
    if status != 0 {
        return Err(Error::ArrayCreation("Failed to compute Adam update".into()));
    }

    let scaled_update = &lr_arr * &update;
    let new_param = param - &scaled_update;

    Ok((new_param, new_m, new_v))
}

/// Adam optimizer step with weight decay (AdamW)
///
/// Like Adam but applies weight decay directly to parameters:
/// param = param - lr * weight_decay * param - lr * m_hat / (sqrt(v_hat) + eps)
///
/// # Arguments
/// * `param` - Parameter array to update
/// * `grad` - Gradient array
/// * `m` - First moment estimate
/// * `v` - Second moment estimate
/// * `lr` - Learning rate
/// * `beta1` - Exponential decay rate for first moment (typically 0.9)
/// * `beta2` - Exponential decay rate for second moment (typically 0.999)
/// * `eps` - Small constant for numerical stability (typically 1e-8)
/// * `weight_decay` - Weight decay coefficient
/// * `t` - Current timestep (starting from 1)
///
/// # Returns
/// Tuple of (updated_param, updated_m, updated_v)
pub fn adamw_step(
    param: &Array,
    grad: &Array,
    m: &Array,
    v: &Array,
    lr: f32,
    beta1: f32,
    beta2: f32,
    eps: f32,
    weight_decay: f32,
    t: i32,
) -> Result<(Array, Array, Array)> {
    let stream = Stream::default();

    let lr_arr = Array::from_float(lr);
    let beta1_arr = Array::from_float(beta1);
    let beta2_arr = Array::from_float(beta2);
    let one_minus_beta1 = Array::from_float(1.0 - beta1);
    let one_minus_beta2 = Array::from_float(1.0 - beta2);
    let eps_arr = Array::from_float(eps);
    let wd_arr = Array::from_float(weight_decay);

    // Bias correction terms
    let beta1_t = beta1.powi(t);
    let beta2_t = beta2.powi(t);
    let bias_correction1 = Array::from_float(1.0 / (1.0 - beta1_t));
    let bias_correction2 = Array::from_float(1.0 / (1.0 - beta2_t));

    // m = beta1 * m + (1 - beta1) * grad
    let new_m = &(&beta1_arr * m) + &(&one_minus_beta1 * grad);

    // v = beta2 * v + (1 - beta2) * grad^2
    let grad_sq = grad * grad;
    let new_v = &(&beta2_arr * v) + &(&one_minus_beta2 * &grad_sq);

    // Bias-corrected estimates
    let m_hat = &new_m * &bias_correction1;
    let v_hat = &new_v * &bias_correction2;

    // Weight decay term
    let weight_decay_update = &(&lr_arr * &wd_arr) * param;

    // Adam update: lr * m_hat / (sqrt(v_hat) + eps)
    let v_hat_sqrt = v_hat.sqrt()?;
    let denom = &v_hat_sqrt + &eps_arr;

    let mut adam_update = Array::new_uninit();
    let status = unsafe {
        mlx_sys::mlx_divide(adam_update.as_mut_ptr(), m_hat.as_raw(), denom.as_raw(), stream.as_raw())
    };
    if status != 0 {
        return Err(Error::ArrayCreation("Failed to compute AdamW update".into()));
    }

    let scaled_adam = &lr_arr * &adam_update;

    // param = param - weight_decay_update - adam_update
    let new_param = &(param - &weight_decay_update) - &scaled_adam;

    Ok((new_param, new_m, new_v))
}

/// RMSprop optimizer step
///
/// Updates parameters using:
/// - v = decay * v + (1 - decay) * grad^2
/// - param = param - lr * grad / (sqrt(v) + eps)
///
/// # Arguments
/// * `param` - Parameter array to update
/// * `grad` - Gradient array
/// * `v` - Running average of squared gradients
/// * `lr` - Learning rate
/// * `decay` - Decay rate (typically 0.99)
/// * `eps` - Small constant for numerical stability (typically 1e-8)
///
/// # Returns
/// Tuple of (updated_param, updated_v)
pub fn rmsprop_step(
    param: &Array,
    grad: &Array,
    v: &Array,
    lr: f32,
    decay: f32,
    eps: f32,
) -> Result<(Array, Array)> {
    let stream = Stream::default();

    let lr_arr = Array::from_float(lr);
    let decay_arr = Array::from_float(decay);
    let one_minus_decay = Array::from_float(1.0 - decay);
    let eps_arr = Array::from_float(eps);

    // v = decay * v + (1 - decay) * grad^2
    let grad_sq = grad * grad;
    let new_v = &(&decay_arr * v) + &(&one_minus_decay * &grad_sq);

    // param = param - lr * grad / (sqrt(v) + eps)
    let v_sqrt = new_v.sqrt()?;
    let denom = &v_sqrt + &eps_arr;

    let mut update = Array::new_uninit();
    let status = unsafe {
        mlx_sys::mlx_divide(update.as_mut_ptr(), grad.as_raw(), denom.as_raw(), stream.as_raw())
    };
    if status != 0 {
        return Err(Error::ArrayCreation("Failed to compute RMSprop update".into()));
    }

    let scaled_update = &lr_arr * &update;
    let new_param = param - &scaled_update;

    Ok((new_param, new_v))
}

/// AdaGrad optimizer step
///
/// Updates parameters using:
/// - accumulated = accumulated + grad^2
/// - param = param - lr * grad / (sqrt(accumulated) + eps)
///
/// # Arguments
/// * `param` - Parameter array to update
/// * `grad` - Gradient array
/// * `accumulated` - Sum of squared gradients
/// * `lr` - Learning rate
/// * `eps` - Small constant for numerical stability (typically 1e-8)
///
/// # Returns
/// Tuple of (updated_param, updated_accumulated)
pub fn adagrad_step(
    param: &Array,
    grad: &Array,
    accumulated: &Array,
    lr: f32,
    eps: f32,
) -> Result<(Array, Array)> {
    let stream = Stream::default();

    let lr_arr = Array::from_float(lr);
    let eps_arr = Array::from_float(eps);

    // accumulated = accumulated + grad^2
    let grad_sq = grad * grad;
    let new_accumulated = accumulated + &grad_sq;

    // param = param - lr * grad / (sqrt(accumulated) + eps)
    let acc_sqrt = new_accumulated.sqrt()?;
    let denom = &acc_sqrt + &eps_arr;

    let mut update = Array::new_uninit();
    let status = unsafe {
        mlx_sys::mlx_divide(update.as_mut_ptr(), grad.as_raw(), denom.as_raw(), stream.as_raw())
    };
    if status != 0 {
        return Err(Error::ArrayCreation("Failed to compute AdaGrad update".into()));
    }

    let scaled_update = &lr_arr * &update;
    let new_param = param - &scaled_update;

    Ok((new_param, new_accumulated))
}

/// Initialize optimizer state with zeros matching parameter shape
///
/// Creates a zero-initialized array with the same shape as the parameter,
/// useful for initializing momentum, velocity, or accumulator states.
///
/// # Arguments
/// * `param` - Parameter array to match shape
///
/// # Returns
/// Zero-initialized array with same shape and dtype
pub fn init_optimizer_state(param: &Array) -> Result<Array> {
    let shape = param.shape();
    let shape_i32: Vec<i32> = shape.iter().map(|&x| x as i32).collect();
    Array::zeros::<f32>(&shape_i32)
}

// ============================================================================
// Stateful Optimizers
// ============================================================================

/// Trait for optimizers that can update parameters
pub trait Optimizer {
    /// Perform one optimization step
    ///
    /// Updates the internal state and returns the updated parameters.
    fn step(&mut self, params: &[Array], grads: &[Array]) -> Result<Vec<Array>>;

    /// Get the current learning rate
    fn learning_rate(&self) -> f32;

    /// Set the learning rate
    fn set_learning_rate(&mut self, lr: f32);

    /// Reset the optimizer state
    fn reset(&mut self);

    /// Get the current step count
    fn step_count(&self) -> i32;
}

/// SGD optimizer with optional momentum and weight decay
///
/// # Example
/// ```ignore
/// use mlx_rs::nn::SGD;
///
/// let mut optimizer = SGD::new(0.01)
///     .momentum(0.9)
///     .weight_decay(0.0001);
///
/// // Training loop
/// for _ in 0..epochs {
///     let grads = compute_gradients(&params);
///     params = optimizer.step(&params, &grads)?;
/// }
/// ```
pub struct SGD {
    lr: f32,
    momentum: f32,
    weight_decay: f32,
    dampening: f32,
    nesterov: bool,
    velocities: Vec<Array>,
    step_count: i32,
}

impl SGD {
    /// Create a new SGD optimizer
    pub fn new(lr: f32) -> Self {
        Self {
            lr,
            momentum: 0.0,
            weight_decay: 0.0,
            dampening: 0.0,
            nesterov: false,
            velocities: Vec::new(),
            step_count: 0,
        }
    }

    /// Set momentum factor (default: 0.0)
    pub fn momentum(mut self, momentum: f32) -> Self {
        self.momentum = momentum;
        self
    }

    /// Set weight decay (L2 regularization) (default: 0.0)
    pub fn weight_decay(mut self, weight_decay: f32) -> Self {
        self.weight_decay = weight_decay;
        self
    }

    /// Set dampening for momentum (default: 0.0)
    pub fn dampening(mut self, dampening: f32) -> Self {
        self.dampening = dampening;
        self
    }

    /// Enable Nesterov momentum (default: false)
    pub fn nesterov(mut self, nesterov: bool) -> Self {
        self.nesterov = nesterov;
        self
    }

    /// Initialize velocities for the given parameters
    fn init_state(&mut self, params: &[Array]) -> Result<()> {
        if self.velocities.is_empty() && self.momentum != 0.0 {
            for param in params {
                self.velocities.push(init_optimizer_state(param)?);
            }
        }
        Ok(())
    }
}

impl Optimizer for SGD {
    fn step(&mut self, params: &[Array], grads: &[Array]) -> Result<Vec<Array>> {
        if params.len() != grads.len() {
            return Err(Error::InvalidShape(
                "Number of parameters must match number of gradients".into()
            ));
        }

        self.init_state(params)?;
        self.step_count += 1;

        let mut new_params = Vec::with_capacity(params.len());

        for (i, (param, grad)) in params.iter().zip(grads.iter()).enumerate() {
            // Apply weight decay
            let grad = if self.weight_decay != 0.0 {
                let wd = Array::from_float(self.weight_decay);
                grad + &(&wd * param)
            } else {
                grad.clone()
            };

            // Apply momentum
            let new_param = if self.momentum != 0.0 {
                let momentum_arr = Array::from_float(self.momentum);
                let dampening_arr = Array::from_float(1.0 - self.dampening);

                // velocity = momentum * velocity + (1 - dampening) * grad
                let new_velocity = &(&momentum_arr * &self.velocities[i]) + &(&dampening_arr * &grad);
                self.velocities[i] = new_velocity.clone();
                self.velocities[i].eval();

                // For Nesterov: param = param - lr * (grad + momentum * velocity)
                // For standard: param = param - lr * velocity
                let lr_arr = Array::from_float(self.lr);
                if self.nesterov {
                    let update = &grad + &(&momentum_arr * &new_velocity);
                    param - &(&lr_arr * &update)
                } else {
                    param - &(&lr_arr * &new_velocity)
                }
            } else {
                // Simple SGD without momentum
                let lr_arr = Array::from_float(self.lr);
                param - &(&lr_arr * &grad)
            };

            new_params.push(new_param);
        }

        Ok(new_params)
    }

    fn learning_rate(&self) -> f32 {
        self.lr
    }

    fn set_learning_rate(&mut self, lr: f32) {
        self.lr = lr;
    }

    fn reset(&mut self) {
        self.velocities.clear();
        self.step_count = 0;
    }

    fn step_count(&self) -> i32 {
        self.step_count
    }
}

/// Adam optimizer with bias correction
///
/// Implements the Adam optimization algorithm from "Adam: A Method for
/// Stochastic Optimization" (Kingma & Ba, 2014).
///
/// # Example
/// ```ignore
/// use mlx_rs::nn::Adam;
///
/// let mut optimizer = Adam::new(0.001)
///     .betas(0.9, 0.999)
///     .eps(1e-8)
///     .weight_decay(0.01);
///
/// // Training loop
/// for _ in 0..epochs {
///     let grads = compute_gradients(&params);
///     params = optimizer.step(&params, &grads)?;
/// }
/// ```
pub struct Adam {
    lr: f32,
    beta1: f32,
    beta2: f32,
    eps: f32,
    weight_decay: f32,
    m: Vec<Array>,  // First moment
    v: Vec<Array>,  // Second moment
    step_count: i32,
}

impl Adam {
    /// Create a new Adam optimizer
    pub fn new(lr: f32) -> Self {
        Self {
            lr,
            beta1: 0.9,
            beta2: 0.999,
            eps: 1e-8,
            weight_decay: 0.0,
            m: Vec::new(),
            v: Vec::new(),
            step_count: 0,
        }
    }

    /// Set beta parameters (default: 0.9, 0.999)
    pub fn betas(mut self, beta1: f32, beta2: f32) -> Self {
        self.beta1 = beta1;
        self.beta2 = beta2;
        self
    }

    /// Set epsilon for numerical stability (default: 1e-8)
    pub fn eps(mut self, eps: f32) -> Self {
        self.eps = eps;
        self
    }

    /// Set weight decay (L2 regularization) (default: 0.0)
    pub fn weight_decay(mut self, weight_decay: f32) -> Self {
        self.weight_decay = weight_decay;
        self
    }

    /// Initialize state for the given parameters
    fn init_state(&mut self, params: &[Array]) -> Result<()> {
        if self.m.is_empty() {
            for param in params {
                self.m.push(init_optimizer_state(param)?);
                self.v.push(init_optimizer_state(param)?);
            }
        }
        Ok(())
    }
}

impl Optimizer for Adam {
    fn step(&mut self, params: &[Array], grads: &[Array]) -> Result<Vec<Array>> {
        if params.len() != grads.len() {
            return Err(Error::InvalidShape(
                "Number of parameters must match number of gradients".into()
            ));
        }

        self.init_state(params)?;
        self.step_count += 1;

        let beta1 = self.beta1;
        let beta2 = self.beta2;
        let t = self.step_count;

        // Bias correction factors
        let bias_correction1 = 1.0 - beta1.powi(t);
        let bias_correction2 = 1.0 - beta2.powi(t);

        let mut new_params = Vec::with_capacity(params.len());

        for (i, (param, grad)) in params.iter().zip(grads.iter()).enumerate() {
            // Apply weight decay (standard Adam, not decoupled)
            let grad = if self.weight_decay != 0.0 {
                let wd = Array::from_float(self.weight_decay);
                grad + &(&wd * param)
            } else {
                grad.clone()
            };

            // Update first moment: m = beta1 * m + (1 - beta1) * grad
            let beta1_arr = Array::from_float(beta1);
            let one_minus_beta1 = Array::from_float(1.0 - beta1);
            let new_m = &(&beta1_arr * &self.m[i]) + &(&one_minus_beta1 * &grad);

            // Update second moment: v = beta2 * v + (1 - beta2) * grad^2
            let beta2_arr = Array::from_float(beta2);
            let one_minus_beta2 = Array::from_float(1.0 - beta2);
            let grad_sq = &grad * &grad;
            let new_v = &(&beta2_arr * &self.v[i]) + &(&one_minus_beta2 * &grad_sq);

            // Bias-corrected estimates
            let m_hat = &new_m / &Array::from_float(bias_correction1);
            let v_hat = &new_v / &Array::from_float(bias_correction2);

            // Update parameters: param = param - lr * m_hat / (sqrt(v_hat) + eps)
            let lr_arr = Array::from_float(self.lr);
            let eps_arr = Array::from_float(self.eps);
            let v_sqrt = v_hat.sqrt()?;
            let denom = &v_sqrt + &eps_arr;

            let stream = Stream::default();
            let mut update = Array::new_uninit();
            let status = unsafe {
                mlx_sys::mlx_divide(update.as_mut_ptr(), m_hat.as_raw(), denom.as_raw(), stream.as_raw())
            };
            if status != 0 {
                return Err(Error::ArrayCreation("Failed to compute Adam update".into()));
            }

            let new_param = param - &(&lr_arr * &update);

            // Update state
            self.m[i] = new_m;
            self.v[i] = new_v;
            self.m[i].eval();
            self.v[i].eval();

            new_params.push(new_param);
        }

        Ok(new_params)
    }

    fn learning_rate(&self) -> f32 {
        self.lr
    }

    fn set_learning_rate(&mut self, lr: f32) {
        self.lr = lr;
    }

    fn reset(&mut self) {
        self.m.clear();
        self.v.clear();
        self.step_count = 0;
    }

    fn step_count(&self) -> i32 {
        self.step_count
    }
}

/// AdamW optimizer with decoupled weight decay
///
/// Implements the AdamW optimization algorithm from "Decoupled Weight Decay
/// Regularization" (Loshchilov & Hutter, 2017).
///
/// The key difference from Adam is that weight decay is applied directly to
/// the parameters rather than to the gradients.
///
/// # Example
/// ```ignore
/// use mlx_rs::nn::AdamW;
///
/// let mut optimizer = AdamW::new(0.001)
///     .betas(0.9, 0.999)
///     .weight_decay(0.01);
///
/// // Training loop
/// for _ in 0..epochs {
///     let grads = compute_gradients(&params);
///     params = optimizer.step(&params, &grads)?;
/// }
/// ```
pub struct AdamW {
    lr: f32,
    beta1: f32,
    beta2: f32,
    eps: f32,
    weight_decay: f32,
    m: Vec<Array>,
    v: Vec<Array>,
    step_count: i32,
}

impl AdamW {
    /// Create a new AdamW optimizer
    pub fn new(lr: f32) -> Self {
        Self {
            lr,
            beta1: 0.9,
            beta2: 0.999,
            eps: 1e-8,
            weight_decay: 0.01,  // Default non-zero for AdamW
            m: Vec::new(),
            v: Vec::new(),
            step_count: 0,
        }
    }

    /// Set beta parameters (default: 0.9, 0.999)
    pub fn betas(mut self, beta1: f32, beta2: f32) -> Self {
        self.beta1 = beta1;
        self.beta2 = beta2;
        self
    }

    /// Set epsilon for numerical stability (default: 1e-8)
    pub fn eps(mut self, eps: f32) -> Self {
        self.eps = eps;
        self
    }

    /// Set weight decay (default: 0.01)
    pub fn weight_decay(mut self, weight_decay: f32) -> Self {
        self.weight_decay = weight_decay;
        self
    }

    /// Initialize state for the given parameters
    fn init_state(&mut self, params: &[Array]) -> Result<()> {
        if self.m.is_empty() {
            for param in params {
                self.m.push(init_optimizer_state(param)?);
                self.v.push(init_optimizer_state(param)?);
            }
        }
        Ok(())
    }
}

impl Optimizer for AdamW {
    fn step(&mut self, params: &[Array], grads: &[Array]) -> Result<Vec<Array>> {
        if params.len() != grads.len() {
            return Err(Error::InvalidShape(
                "Number of parameters must match number of gradients".into()
            ));
        }

        self.init_state(params)?;
        self.step_count += 1;

        let beta1 = self.beta1;
        let beta2 = self.beta2;
        let t = self.step_count;

        // Bias correction factors
        let bias_correction1 = 1.0 - beta1.powi(t);
        let bias_correction2 = 1.0 - beta2.powi(t);

        let mut new_params = Vec::with_capacity(params.len());

        for (i, (param, grad)) in params.iter().zip(grads.iter()).enumerate() {
            // Decoupled weight decay: apply directly to parameter
            let decayed_param = if self.weight_decay != 0.0 {
                let decay = Array::from_float(1.0 - self.lr * self.weight_decay);
                param * &decay
            } else {
                param.clone()
            };

            // Update first moment: m = beta1 * m + (1 - beta1) * grad
            let beta1_arr = Array::from_float(beta1);
            let one_minus_beta1 = Array::from_float(1.0 - beta1);
            let new_m = &(&beta1_arr * &self.m[i]) + &(&one_minus_beta1 * grad);

            // Update second moment: v = beta2 * v + (1 - beta2) * grad^2
            let beta2_arr = Array::from_float(beta2);
            let one_minus_beta2 = Array::from_float(1.0 - beta2);
            let grad_sq = grad * grad;
            let new_v = &(&beta2_arr * &self.v[i]) + &(&one_minus_beta2 * &grad_sq);

            // Bias-corrected estimates
            let m_hat = &new_m / &Array::from_float(bias_correction1);
            let v_hat = &new_v / &Array::from_float(bias_correction2);

            // Update parameters: param = decayed_param - lr * m_hat / (sqrt(v_hat) + eps)
            let lr_arr = Array::from_float(self.lr);
            let eps_arr = Array::from_float(self.eps);
            let v_sqrt = v_hat.sqrt()?;
            let denom = &v_sqrt + &eps_arr;

            let stream = Stream::default();
            let mut update = Array::new_uninit();
            let status = unsafe {
                mlx_sys::mlx_divide(update.as_mut_ptr(), m_hat.as_raw(), denom.as_raw(), stream.as_raw())
            };
            if status != 0 {
                return Err(Error::ArrayCreation("Failed to compute AdamW update".into()));
            }

            let new_param = &decayed_param - &(&lr_arr * &update);

            // Update state
            self.m[i] = new_m;
            self.v[i] = new_v;
            self.m[i].eval();
            self.v[i].eval();

            new_params.push(new_param);
        }

        Ok(new_params)
    }

    fn learning_rate(&self) -> f32 {
        self.lr
    }

    fn set_learning_rate(&mut self, lr: f32) {
        self.lr = lr;
    }

    fn reset(&mut self) {
        self.m.clear();
        self.v.clear();
        self.step_count = 0;
    }

    fn step_count(&self) -> i32 {
        self.step_count
    }
}

/// RMSprop optimizer
///
/// # Example
/// ```ignore
/// use mlx_rs::nn::RMSprop;
///
/// let mut optimizer = RMSprop::new(0.01)
///     .alpha(0.99)
///     .eps(1e-8);
///
/// params = optimizer.step(&params, &grads)?;
/// ```
pub struct RMSprop {
    lr: f32,
    alpha: f32,
    eps: f32,
    weight_decay: f32,
    momentum: f32,
    centered: bool,
    v: Vec<Array>,         // Running average of squared gradients
    buf: Vec<Array>,       // Momentum buffer
    grad_avg: Vec<Array>,  // For centered RMSprop
    step_count: i32,
}

impl RMSprop {
    /// Create a new RMSprop optimizer
    pub fn new(lr: f32) -> Self {
        Self {
            lr,
            alpha: 0.99,
            eps: 1e-8,
            weight_decay: 0.0,
            momentum: 0.0,
            centered: false,
            v: Vec::new(),
            buf: Vec::new(),
            grad_avg: Vec::new(),
            step_count: 0,
        }
    }

    /// Set smoothing constant alpha (default: 0.99)
    pub fn alpha(mut self, alpha: f32) -> Self {
        self.alpha = alpha;
        self
    }

    /// Set epsilon (default: 1e-8)
    pub fn eps(mut self, eps: f32) -> Self {
        self.eps = eps;
        self
    }

    /// Set weight decay (default: 0.0)
    pub fn weight_decay(mut self, weight_decay: f32) -> Self {
        self.weight_decay = weight_decay;
        self
    }

    /// Set momentum (default: 0.0)
    pub fn momentum(mut self, momentum: f32) -> Self {
        self.momentum = momentum;
        self
    }

    /// Enable centered RMSprop (default: false)
    pub fn centered(mut self, centered: bool) -> Self {
        self.centered = centered;
        self
    }

    fn init_state(&mut self, params: &[Array]) -> Result<()> {
        if self.v.is_empty() {
            for param in params {
                self.v.push(init_optimizer_state(param)?);
                if self.momentum != 0.0 {
                    self.buf.push(init_optimizer_state(param)?);
                }
                if self.centered {
                    self.grad_avg.push(init_optimizer_state(param)?);
                }
            }
        }
        Ok(())
    }
}

impl Optimizer for RMSprop {
    fn step(&mut self, params: &[Array], grads: &[Array]) -> Result<Vec<Array>> {
        if params.len() != grads.len() {
            return Err(Error::InvalidShape(
                "Number of parameters must match number of gradients".into()
            ));
        }

        self.init_state(params)?;
        self.step_count += 1;

        let mut new_params = Vec::with_capacity(params.len());

        for (i, (param, grad)) in params.iter().zip(grads.iter()).enumerate() {
            // Apply weight decay
            let grad = if self.weight_decay != 0.0 {
                let wd = Array::from_float(self.weight_decay);
                grad + &(&wd * param)
            } else {
                grad.clone()
            };

            // v = alpha * v + (1 - alpha) * grad^2
            let alpha_arr = Array::from_float(self.alpha);
            let one_minus_alpha = Array::from_float(1.0 - self.alpha);
            let grad_sq = &grad * &grad;
            let new_v = &(&alpha_arr * &self.v[i]) + &(&one_minus_alpha * &grad_sq);
            self.v[i] = new_v.clone();
            self.v[i].eval();

            // Compute denominator
            let avg = if self.centered {
                let new_grad_avg = &(&alpha_arr * &self.grad_avg[i]) + &(&one_minus_alpha * &grad);
                self.grad_avg[i] = new_grad_avg.clone();
                self.grad_avg[i].eval();
                let grad_avg_sq = &new_grad_avg * &new_grad_avg;
                &new_v - &grad_avg_sq
            } else {
                new_v.clone()
            };

            let eps_arr = Array::from_float(self.eps);
            let avg_sqrt = avg.sqrt()?;
            let denom = &avg_sqrt + &eps_arr;

            // Compute update
            let stream = Stream::default();
            let mut update = Array::new_uninit();
            let status = unsafe {
                mlx_sys::mlx_divide(update.as_mut_ptr(), grad.as_raw(), denom.as_raw(), stream.as_raw())
            };
            if status != 0 {
                return Err(Error::ArrayCreation("Failed to compute RMSprop update".into()));
            }

            // Apply momentum if specified
            let final_update = if self.momentum != 0.0 {
                let momentum_arr = Array::from_float(self.momentum);
                let new_buf = &(&momentum_arr * &self.buf[i]) + &update;
                self.buf[i] = new_buf.clone();
                self.buf[i].eval();
                new_buf
            } else {
                update
            };

            let lr_arr = Array::from_float(self.lr);
            let new_param = param - &(&lr_arr * &final_update);
            new_params.push(new_param);
        }

        Ok(new_params)
    }

    fn learning_rate(&self) -> f32 {
        self.lr
    }

    fn set_learning_rate(&mut self, lr: f32) {
        self.lr = lr;
    }

    fn reset(&mut self) {
        self.v.clear();
        self.buf.clear();
        self.grad_avg.clear();
        self.step_count = 0;
    }

    fn step_count(&self) -> i32 {
        self.step_count
    }
}

// ============================================================================
// Llama Model Architecture
// ============================================================================

/// Configuration for Llama model
#[derive(Debug, Clone)]
pub struct LlamaConfig {
    /// Vocabulary size
    pub vocab_size: i32,
    /// Hidden dimension (model dimension)
    pub hidden_size: i32,
    /// Intermediate dimension for FFN (usually 4 * hidden_size * 2/3 for SwiGLU)
    pub intermediate_size: i32,
    /// Number of attention heads
    pub num_attention_heads: i32,
    /// Number of key-value heads (for GQA, grouped query attention)
    pub num_key_value_heads: i32,
    /// Number of transformer layers
    pub num_hidden_layers: i32,
    /// RMS norm epsilon
    pub rms_norm_eps: f32,
    /// Maximum sequence length
    pub max_position_embeddings: i32,
    /// RoPE theta (base frequency)
    pub rope_theta: f32,
    /// Whether to use scaled RoPE
    pub rope_scaling: Option<f32>,
}

impl Default for LlamaConfig {
    fn default() -> Self {
        // Default Llama 7B configuration
        Self {
            vocab_size: 32000,
            hidden_size: 4096,
            intermediate_size: 11008,  // 4 * 4096 * 2/3 ≈ 10922, rounded
            num_attention_heads: 32,
            num_key_value_heads: 32,   // No GQA in original Llama 1
            num_hidden_layers: 32,
            rms_norm_eps: 1e-6,
            max_position_embeddings: 2048,
            rope_theta: 10000.0,
            rope_scaling: None,
        }
    }
}

impl LlamaConfig {
    /// Create a new Llama configuration
    pub fn new() -> Self {
        Self::default()
    }

    /// Llama 2 7B configuration
    pub fn llama2_7b() -> Self {
        Self {
            vocab_size: 32000,
            hidden_size: 4096,
            intermediate_size: 11008,
            num_attention_heads: 32,
            num_key_value_heads: 32,
            num_hidden_layers: 32,
            rms_norm_eps: 1e-5,
            max_position_embeddings: 4096,
            rope_theta: 10000.0,
            rope_scaling: None,
        }
    }

    /// Llama 2 13B configuration
    pub fn llama2_13b() -> Self {
        Self {
            vocab_size: 32000,
            hidden_size: 5120,
            intermediate_size: 13824,
            num_attention_heads: 40,
            num_key_value_heads: 40,
            num_hidden_layers: 40,
            rms_norm_eps: 1e-5,
            max_position_embeddings: 4096,
            rope_theta: 10000.0,
            rope_scaling: None,
        }
    }

    /// Llama 3 8B configuration
    pub fn llama3_8b() -> Self {
        Self {
            vocab_size: 128256,
            hidden_size: 4096,
            intermediate_size: 14336,
            num_attention_heads: 32,
            num_key_value_heads: 8,  // GQA with 8 KV heads
            num_hidden_layers: 32,
            rms_norm_eps: 1e-5,
            max_position_embeddings: 8192,
            rope_theta: 500000.0,
            rope_scaling: None,
        }
    }

    /// Set vocabulary size
    pub fn vocab_size(mut self, vocab_size: i32) -> Self {
        self.vocab_size = vocab_size;
        self
    }

    /// Set hidden size
    pub fn hidden_size(mut self, hidden_size: i32) -> Self {
        self.hidden_size = hidden_size;
        self
    }

    /// Set number of layers
    pub fn num_hidden_layers(mut self, num_layers: i32) -> Self {
        self.num_hidden_layers = num_layers;
        self
    }

    /// Set number of attention heads
    pub fn num_attention_heads(mut self, num_heads: i32) -> Self {
        self.num_attention_heads = num_heads;
        self
    }

    /// Head dimension
    pub fn head_dim(&self) -> i32 {
        self.hidden_size / self.num_attention_heads
    }
}

/// SwiGLU activation function
///
/// SwiGLU(x, W, V) = Swish(x @ W) * (x @ V)
/// where Swish(x) = x * sigmoid(x)
pub fn swiglu(x: &Array, gate_proj: &Array, up_proj: &Array) -> Result<Array> {
    // gate = x @ gate_proj
    let gate = x.matmul(gate_proj)?;

    // up = x @ up_proj
    let up = x.matmul(up_proj)?;

    // swish = gate * sigmoid(gate)
    let gate_sigmoid = sigmoid(&gate)?;
    let swish = &gate * &gate_sigmoid;

    // output = swish * up
    Ok(&swish * &up)
}

/// Llama FeedForward network (MLP with SwiGLU)
///
/// FFN(x) = (Swish(x @ W_gate) * (x @ W_up)) @ W_down
///
/// # Arguments
/// * `x` - Input tensor of shape (batch, seq_len, hidden_size)
/// * `gate_proj` - Gate projection weights (hidden_size, intermediate_size)
/// * `up_proj` - Up projection weights (hidden_size, intermediate_size)
/// * `down_proj` - Down projection weights (intermediate_size, hidden_size)
pub fn llama_feedforward(
    x: &Array,
    gate_proj: &Array,
    up_proj: &Array,
    down_proj: &Array,
) -> Result<Array> {
    let intermediate = swiglu(x, gate_proj, up_proj)?;
    intermediate.matmul(down_proj)
}

/// Llama attention with RoPE (Rotary Position Embedding)
///
/// # Arguments
/// * `x` - Input tensor of shape (batch, seq_len, hidden_size)
/// * `q_proj` - Query projection weights (hidden_size, hidden_size)
/// * `k_proj` - Key projection weights (hidden_size, kv_hidden_size)
/// * `v_proj` - Value projection weights (hidden_size, kv_hidden_size)
/// * `o_proj` - Output projection weights (hidden_size, hidden_size)
/// * `cos` - Cosine frequencies for RoPE (seq_len, head_dim)
/// * `sin` - Sine frequencies for RoPE (seq_len, head_dim)
/// * `num_heads` - Number of attention heads
/// * `num_kv_heads` - Number of key-value heads (for GQA)
/// * `mask` - Optional attention mask
pub fn llama_attention(
    x: &Array,
    q_proj: &Array,
    k_proj: &Array,
    v_proj: &Array,
    o_proj: &Array,
    cos: &Array,
    sin: &Array,
    num_heads: i32,
    num_kv_heads: i32,
    mask: Option<&Array>,
) -> Result<Array> {
    let shape = x.shape();
    if shape.len() < 3 {
        return Err(Error::InvalidShape("Expected 3D input (batch, seq_len, hidden)".into()));
    }

    let batch = shape[0] as i32;
    let seq_len = shape[1] as i32;
    let hidden_size = shape[2] as i32;
    let head_dim = hidden_size / num_heads;
    let _kv_hidden_size = head_dim * num_kv_heads;

    // Project to Q, K, V
    let q = x.matmul(q_proj)?;
    let k = x.matmul(k_proj)?;
    let v = x.matmul(v_proj)?;

    // Reshape for multi-head attention
    // Q: (batch, seq_len, num_heads, head_dim)
    let q = q.reshape(&[batch, seq_len, num_heads, head_dim])?;
    // K, V: (batch, seq_len, num_kv_heads, head_dim)
    let k = k.reshape(&[batch, seq_len, num_kv_heads, head_dim])?;
    let v = v.reshape(&[batch, seq_len, num_kv_heads, head_dim])?;

    // Transpose to (batch, num_heads, seq_len, head_dim)
    let q = q.transpose_axes(&[0, 2, 1, 3])?;
    let k = k.transpose_axes(&[0, 2, 1, 3])?;
    let v = v.transpose_axes(&[0, 2, 1, 3])?;

    // Apply RoPE to Q and K
    let q = apply_rotary_embedding(&q, cos, sin)?;
    let k = apply_rotary_embedding(&k, cos, sin)?;

    // Handle GQA: repeat K, V heads if needed
    let (k, v) = if num_kv_heads < num_heads {
        let num_groups = num_heads / num_kv_heads;
        // Expand K and V by repeating
        let k_expanded = repeat_kv(&k, num_groups)?;
        let v_expanded = repeat_kv(&v, num_groups)?;
        (k_expanded, v_expanded)
    } else {
        (k, v)
    };

    // Compute scaled dot-product attention
    let scale = 1.0 / (head_dim as f32).sqrt();
    let attn_mask = if mask.is_some() { AttentionMask::Custom } else { AttentionMask::None };
    let attn_output = scaled_dot_product_attention(&q, &k, &v, Some(scale), attn_mask, mask)?;

    // Transpose back: (batch, seq_len, num_heads, head_dim)
    let attn_output = attn_output.transpose_axes(&[0, 2, 1, 3])?;

    // Reshape to (batch, seq_len, hidden_size)
    let attn_output = attn_output.reshape(&[batch, seq_len, hidden_size])?;

    // Output projection
    attn_output.matmul(o_proj)
}

/// Repeat key-value heads for grouped query attention (GQA)
fn repeat_kv(x: &Array, num_groups: i32) -> Result<Array> {
    let shape = x.shape();
    if shape.len() != 4 {
        return Err(Error::InvalidShape("Expected 4D input for repeat_kv".into()));
    }

    let batch = shape[0] as i32;
    let num_kv_heads = shape[1] as i32;
    let seq_len = shape[2] as i32;
    let head_dim = shape[3] as i32;

    // Expand and repeat
    // (batch, num_kv_heads, seq_len, head_dim) -> (batch, num_kv_heads, 1, seq_len, head_dim)
    let x_expanded = x.expand_dims(2)?;

    // Broadcast to (batch, num_kv_heads, num_groups, seq_len, head_dim)
    let stream = Stream::default();

    // Create shape for broadcasting
    let target_shape = [batch, num_kv_heads, num_groups, seq_len, head_dim];

    // Use broadcast_to
    let mut result = Array::new_uninit();
    let status = unsafe {
        mlx_sys::mlx_broadcast_to(
            result.as_mut_ptr(),
            x_expanded.as_raw(),
            target_shape.as_ptr(),
            target_shape.len(),
            stream.as_raw(),
        )
    };

    if status != 0 {
        return Err(Error::ArrayCreation("Failed to broadcast for repeat_kv".into()));
    }

    // Reshape to (batch, num_kv_heads * num_groups, seq_len, head_dim)
    result.reshape(&[batch, num_kv_heads * num_groups, seq_len, head_dim])
}

/// Llama Transformer Block
///
/// A single transformer block consisting of:
/// 1. Input -> RMSNorm -> Self-Attention -> Residual
/// 2. -> RMSNorm -> FeedForward -> Residual
///
/// # Arguments
/// * `x` - Input tensor (batch, seq_len, hidden_size)
/// * `attention_norm` - RMSNorm weight for attention
/// * `ffn_norm` - RMSNorm weight for feedforward
/// * `q_proj`, `k_proj`, `v_proj`, `o_proj` - Attention projections
/// * `gate_proj`, `up_proj`, `down_proj` - FFN projections
/// * `cos`, `sin` - RoPE frequencies
/// * `config` - Llama configuration
/// * `mask` - Optional attention mask
pub fn llama_transformer_block(
    x: &Array,
    attention_norm: &Array,
    ffn_norm: &Array,
    q_proj: &Array,
    k_proj: &Array,
    v_proj: &Array,
    o_proj: &Array,
    gate_proj: &Array,
    up_proj: &Array,
    down_proj: &Array,
    cos: &Array,
    sin: &Array,
    config: &LlamaConfig,
    mask: Option<&Array>,
) -> Result<Array> {
    // Self-attention with pre-norm
    let normed = rms_norm(x, attention_norm, config.rms_norm_eps)?;
    let attn_output = llama_attention(
        &normed,
        q_proj, k_proj, v_proj, o_proj,
        cos, sin,
        config.num_attention_heads,
        config.num_key_value_heads,
        mask,
    )?;

    // Residual connection
    let x = x + &attn_output;

    // FFN with pre-norm
    let normed = rms_norm(&x, ffn_norm, config.rms_norm_eps)?;
    let ffn_output = llama_feedforward(&normed, gate_proj, up_proj, down_proj)?;

    // Residual connection
    Ok(&x + &ffn_output)
}

/// Weights for a single Llama transformer layer
#[derive(Debug)]
pub struct LlamaLayerWeights {
    /// RMSNorm weight for attention
    pub attention_norm: Array,
    /// RMSNorm weight for feedforward
    pub ffn_norm: Array,
    /// Query projection
    pub q_proj: Array,
    /// Key projection
    pub k_proj: Array,
    /// Value projection
    pub v_proj: Array,
    /// Output projection
    pub o_proj: Array,
    /// Gate projection for SwiGLU
    pub gate_proj: Array,
    /// Up projection for SwiGLU
    pub up_proj: Array,
    /// Down projection
    pub down_proj: Array,
}

impl LlamaLayerWeights {
    /// Initialize random weights for testing
    pub fn random(config: &LlamaConfig) -> Result<Self> {
        let hidden = config.hidden_size;
        let intermediate = config.intermediate_size;
        let kv_hidden = config.head_dim() * config.num_key_value_heads;

        Ok(Self {
            attention_norm: crate::random::normal_with_params::<f32>(&[hidden], 0.0, 0.02, None)?,
            ffn_norm: crate::random::normal_with_params::<f32>(&[hidden], 0.0, 0.02, None)?,
            q_proj: crate::random::normal_with_params::<f32>(&[hidden, hidden], 0.0, 0.02, None)?,
            k_proj: crate::random::normal_with_params::<f32>(&[hidden, kv_hidden], 0.0, 0.02, None)?,
            v_proj: crate::random::normal_with_params::<f32>(&[hidden, kv_hidden], 0.0, 0.02, None)?,
            o_proj: crate::random::normal_with_params::<f32>(&[hidden, hidden], 0.0, 0.02, None)?,
            gate_proj: crate::random::normal_with_params::<f32>(&[hidden, intermediate], 0.0, 0.02, None)?,
            up_proj: crate::random::normal_with_params::<f32>(&[hidden, intermediate], 0.0, 0.02, None)?,
            down_proj: crate::random::normal_with_params::<f32>(&[intermediate, hidden], 0.0, 0.02, None)?,
        })
    }
}

/// Full Llama model weights
#[derive(Debug)]
pub struct LlamaWeights {
    /// Token embedding
    pub embed_tokens: Array,
    /// Layer weights
    pub layers: Vec<LlamaLayerWeights>,
    /// Final RMSNorm
    pub norm: Array,
    /// Output projection (lm_head), often tied to embed_tokens
    pub lm_head: Option<Array>,
}

impl LlamaWeights {
    /// Initialize random weights for testing
    pub fn random(config: &LlamaConfig) -> Result<Self> {
        let hidden = config.hidden_size;
        let vocab = config.vocab_size;

        let mut layers = Vec::with_capacity(config.num_hidden_layers as usize);
        for _ in 0..config.num_hidden_layers {
            layers.push(LlamaLayerWeights::random(config)?);
        }

        Ok(Self {
            embed_tokens: crate::random::normal_with_params::<f32>(&[vocab, hidden], 0.0, 0.02, None)?,
            layers,
            norm: crate::random::normal_with_params::<f32>(&[hidden], 0.0, 0.02, None)?,
            lm_head: Some(crate::random::normal_with_params::<f32>(&[hidden, vocab], 0.0, 0.02, None)?),
        })
    }
}

/// Llama Model
///
/// Complete Llama model implementation supporting inference.
///
/// # Example
/// ```ignore
/// use mlx_rs::nn::{LlamaConfig, LlamaModel, LlamaWeights};
///
/// let config = LlamaConfig::llama2_7b();
/// let weights = LlamaWeights::random(&config)?;
/// let model = LlamaModel::new(config);
///
/// let input_ids = Array::from_slice(&[1i32, 2, 3, 4], &[1, 4])?;
/// let logits = model.forward(&input_ids, &weights)?;
/// ```
pub struct LlamaModel {
    pub config: LlamaConfig,
}

impl LlamaModel {
    /// Create a new Llama model
    pub fn new(config: LlamaConfig) -> Self {
        Self { config }
    }

    /// Forward pass
    ///
    /// # Arguments
    /// * `input_ids` - Token IDs of shape (batch, seq_len)
    /// * `weights` - Model weights
    ///
    /// # Returns
    /// Logits of shape (batch, seq_len, vocab_size)
    pub fn forward(&self, input_ids: &Array, weights: &LlamaWeights) -> Result<Array> {
        let shape = input_ids.shape();
        if shape.len() != 2 {
            return Err(Error::InvalidShape("Expected 2D input_ids (batch, seq_len)".into()));
        }

        let _batch = shape[0] as i32;
        let seq_len = shape[1] as i32;

        // Token embedding
        let hidden_states = embedding(&weights.embed_tokens, input_ids)?;

        // Precompute RoPE frequencies
        let (cos, sin) = precompute_rope_frequencies(
            self.config.head_dim(),
            seq_len,
            self.config.rope_theta,
        )?;

        // Create causal mask
        let mask = create_causal_mask(seq_len)?;

        // Apply transformer layers
        let mut hidden_states = hidden_states;
        for layer in &weights.layers {
            hidden_states = llama_transformer_block(
                &hidden_states,
                &layer.attention_norm,
                &layer.ffn_norm,
                &layer.q_proj,
                &layer.k_proj,
                &layer.v_proj,
                &layer.o_proj,
                &layer.gate_proj,
                &layer.up_proj,
                &layer.down_proj,
                &cos,
                &sin,
                &self.config,
                Some(&mask),
            )?;
        }

        // Final normalization
        let hidden_states = rms_norm(&hidden_states, &weights.norm, self.config.rms_norm_eps)?;

        // Language model head
        if let Some(ref lm_head) = weights.lm_head {
            hidden_states.matmul(lm_head)
        } else {
            // Tie weights with embeddings (transpose embedding matrix)
            let lm_head = weights.embed_tokens.transpose()?;
            hidden_states.matmul(&lm_head)
        }
    }

    /// Generate tokens autoregressively
    ///
    /// # Arguments
    /// * `input_ids` - Initial token IDs (batch, seq_len)
    /// * `weights` - Model weights
    /// * `max_new_tokens` - Maximum number of tokens to generate
    /// * `temperature` - Sampling temperature (1.0 = no change, <1.0 = more deterministic)
    ///
    /// # Returns
    /// Generated token IDs including the input
    pub fn generate(
        &self,
        input_ids: &Array,
        weights: &LlamaWeights,
        max_new_tokens: i32,
        temperature: f32,
    ) -> Result<Array> {
        let mut current_ids = input_ids.clone();

        for _ in 0..max_new_tokens {
            // Get logits for last position
            let logits = self.forward(&current_ids, weights)?;

            // Get last token logits
            let last_logits = get_last_token_logits(&logits)?;

            // Apply temperature
            let scaled_logits = if (temperature - 1.0).abs() > 1e-6 {
                let temp = Array::from_float(temperature);
                &last_logits / &temp
            } else {
                last_logits
            };

            // Sample next token (greedy for now - take argmax)
            let next_token = scaled_logits.argmax(false)?;
            next_token.eval();

            // Reshape next_token to (batch, 1)
            let shape = current_ids.shape();
            let batch = shape[0] as i32;
            let next_token = next_token.reshape(&[batch, 1])?;

            // Concatenate
            current_ids = crate::ops::concatenate(&[&current_ids, &next_token], 1)?;
        }

        Ok(current_ids)
    }
}

/// Helper to get logits for the last token position
fn get_last_token_logits(logits: &Array) -> Result<Array> {
    let shape = logits.shape();
    if shape.len() != 3 {
        return Err(Error::InvalidShape("Expected 3D logits".into()));
    }

    let seq_len = shape[1] as i32;

    // Slice to get last position: [:, -1, :]
    logits.slice(&[0, seq_len - 1, 0], &[shape[0] as i32, seq_len, shape[2] as i32], None)
}

/// Create a causal attention mask
///
/// Returns a mask where mask[i, j] = -inf if j > i, else 0
pub fn create_causal_mask(seq_len: i32) -> Result<Array> {
    // Create upper triangular matrix of -inf (for causal attention)
    let stream = Stream::default();

    // First create a matrix of ones
    let ones = Array::ones::<f32>(&[seq_len, seq_len])?;

    // Create upper triangular (triu with k=1)
    let mut triu = Array::new_uninit();
    let status = unsafe {
        mlx_sys::mlx_triu(
            triu.as_mut_ptr(),
            ones.as_raw(),
            1,  // k=1 to exclude diagonal
            stream.as_raw(),
        )
    };

    if status != 0 {
        return Err(Error::ArrayCreation("Failed to create triu mask".into()));
    }

    // Create boolean condition: triu > 0 (true for upper triangle)
    let zeros = Array::zeros::<f32>(&[seq_len, seq_len])?;
    let condition = triu.gt(&zeros)?;

    // Create -inf array by using a very negative number
    let neg_inf_val = Array::from_float(f32::NEG_INFINITY);
    let zeros_scalar = Array::from_float(0.0f32);

    // Where condition is true (upper triangle), use -inf; otherwise use 0
    crate::ops::where_cond(&condition, &neg_inf_val, &zeros_scalar)
}

// ============================================================================
// BERT Model Implementation
// ============================================================================

/// Configuration for BERT models.
///
/// BERT (Bidirectional Encoder Representations from Transformers) is an
/// encoder-only transformer model for embeddings, classification, and NLU tasks.
///
/// # Example
/// ```ignore
/// use mlx_rs::nn::BertConfig;
///
/// // Use a preset
/// let config = BertConfig::bert_base_uncased();
///
/// // Or customize
/// let config = BertConfig::new()
///     .hidden_size(768)
///     .num_hidden_layers(12)
///     .num_attention_heads(12);
/// ```
#[derive(Debug, Clone)]
pub struct BertConfig {
    /// Vocabulary size
    pub vocab_size: i32,
    /// Hidden size (embedding dimension)
    pub hidden_size: i32,
    /// Number of transformer layers
    pub num_hidden_layers: i32,
    /// Number of attention heads
    pub num_attention_heads: i32,
    /// Intermediate size in feed-forward layers
    pub intermediate_size: i32,
    /// Hidden activation function (gelu or relu)
    pub hidden_act: String,
    /// Dropout probability
    pub hidden_dropout_prob: f32,
    /// Attention dropout probability
    pub attention_probs_dropout_prob: f32,
    /// Maximum sequence length
    pub max_position_embeddings: i32,
    /// Number of token types (segment IDs)
    pub type_vocab_size: i32,
    /// Layer normalization epsilon
    pub layer_norm_eps: f32,
    /// Padding token ID
    pub pad_token_id: i32,
}

impl Default for BertConfig {
    fn default() -> Self {
        Self::new()
    }
}

impl BertConfig {
    /// Create a new BERT configuration with default values.
    pub fn new() -> Self {
        Self {
            vocab_size: 30522,
            hidden_size: 768,
            num_hidden_layers: 12,
            num_attention_heads: 12,
            intermediate_size: 3072,
            hidden_act: "gelu".to_string(),
            hidden_dropout_prob: 0.1,
            attention_probs_dropout_prob: 0.1,
            max_position_embeddings: 512,
            type_vocab_size: 2,
            layer_norm_eps: 1e-12,
            pad_token_id: 0,
        }
    }

    /// BERT Base Uncased configuration (110M parameters).
    pub fn bert_base_uncased() -> Self {
        Self::new()
    }

    /// BERT Base Cased configuration.
    pub fn bert_base_cased() -> Self {
        Self {
            vocab_size: 28996,
            ..Self::new()
        }
    }

    /// BERT Large Uncased configuration (340M parameters).
    pub fn bert_large_uncased() -> Self {
        Self {
            vocab_size: 30522,
            hidden_size: 1024,
            num_hidden_layers: 24,
            num_attention_heads: 16,
            intermediate_size: 4096,
            ..Self::new()
        }
    }

    /// BERT Large Cased configuration.
    pub fn bert_large_cased() -> Self {
        Self {
            vocab_size: 28996,
            hidden_size: 1024,
            num_hidden_layers: 24,
            num_attention_heads: 16,
            intermediate_size: 4096,
            ..Self::new()
        }
    }

    // Builder methods
    pub fn vocab_size(mut self, vocab_size: i32) -> Self {
        self.vocab_size = vocab_size;
        self
    }

    pub fn hidden_size(mut self, hidden_size: i32) -> Self {
        self.hidden_size = hidden_size;
        self
    }

    pub fn num_hidden_layers(mut self, num_hidden_layers: i32) -> Self {
        self.num_hidden_layers = num_hidden_layers;
        self
    }

    pub fn num_attention_heads(mut self, num_attention_heads: i32) -> Self {
        self.num_attention_heads = num_attention_heads;
        self
    }

    pub fn intermediate_size(mut self, intermediate_size: i32) -> Self {
        self.intermediate_size = intermediate_size;
        self
    }

    pub fn max_position_embeddings(mut self, max_position_embeddings: i32) -> Self {
        self.max_position_embeddings = max_position_embeddings;
        self
    }

    pub fn type_vocab_size(mut self, type_vocab_size: i32) -> Self {
        self.type_vocab_size = type_vocab_size;
        self
    }

    pub fn layer_norm_eps(mut self, layer_norm_eps: f32) -> Self {
        self.layer_norm_eps = layer_norm_eps;
        self
    }

    /// Get the head dimension.
    pub fn head_dim(&self) -> i32 {
        self.hidden_size / self.num_attention_heads
    }
}

/// BERT embeddings layer.
///
/// Combines word embeddings, position embeddings, and token type embeddings,
/// followed by layer normalization.
pub fn bert_embeddings(
    input_ids: &Array,
    token_type_ids: Option<&Array>,
    position_ids: Option<&Array>,
    word_embeddings: &Array,
    position_embeddings: &Array,
    token_type_embeddings: &Array,
    layer_norm_weight: &Array,
    layer_norm_bias: &Array,
    config: &BertConfig,
) -> Result<Array> {
    let shape = input_ids.shape();
    if shape.len() != 2 {
        return Err(Error::InvalidShape("Expected 2D input_ids (batch, seq_len)".into()));
    }

    let batch_size = shape[0] as i32;
    let seq_len = shape[1] as i32;

    // Word embeddings
    let words_embeds = embedding(word_embeddings, input_ids)?;

    // Position embeddings
    let pos_embeds = if let Some(pos_ids) = position_ids {
        embedding(position_embeddings, pos_ids)?
    } else {
        // Create position IDs: [0, 1, 2, ..., seq_len-1]
        let pos_ids = Array::arange::<i32>(0.0, seq_len as f64, 1.0)?;
        let pos_ids = pos_ids.reshape(&[1, seq_len])?;
        // Broadcast to batch
        let pos_embeds = embedding(position_embeddings, &pos_ids)?;
        // Broadcast to batch size
        broadcast_to(&pos_embeds, &[batch_size, seq_len, config.hidden_size])?
    };

    // Token type embeddings
    let token_embeds = if let Some(tt_ids) = token_type_ids {
        embedding(token_type_embeddings, tt_ids)?
    } else {
        // Default to zeros (all segment A)
        let tt_ids = Array::zeros::<i32>(&[batch_size, seq_len])?;
        embedding(token_type_embeddings, &tt_ids)?
    };

    // Sum embeddings
    let embeddings = &(&words_embeds + &pos_embeds) + &token_embeds;

    // Layer normalization
    layer_norm(&embeddings, layer_norm_weight, layer_norm_bias, config.layer_norm_eps)
}

/// Broadcast an array to target shape.
fn broadcast_to(x: &Array, target_shape: &[i32]) -> Result<Array> {
    let stream = Stream::default();
    let mut result = Array::new_uninit();

    let status = unsafe {
        mlx_sys::mlx_broadcast_to(
            result.as_mut_ptr(),
            x.as_raw(),
            target_shape.as_ptr(),
            target_shape.len(),
            stream.as_raw(),
        )
    };

    if status != 0 {
        return Err(Error::ArrayCreation("Failed to broadcast array".into()));
    }

    Ok(result)
}

/// BERT self-attention mechanism.
///
/// Multi-head self-attention without masking (bidirectional).
pub fn bert_self_attention(
    hidden_states: &Array,
    query_weight: &Array,
    query_bias: &Array,
    key_weight: &Array,
    key_bias: &Array,
    value_weight: &Array,
    value_bias: &Array,
    attention_mask: Option<&Array>,
    config: &BertConfig,
) -> Result<Array> {
    let shape = hidden_states.shape();
    let batch_size = shape[0] as i32;
    let seq_len = shape[1] as i32;
    let hidden_size = config.hidden_size;
    let num_heads = config.num_attention_heads;
    let head_dim = config.head_dim();

    // Linear projections: (batch, seq, hidden) @ (hidden, hidden) -> (batch, seq, hidden)
    let query = hidden_states.matmul(query_weight)?;
    let query = &query + query_bias;

    let key = hidden_states.matmul(key_weight)?;
    let key = &key + key_bias;

    let value = hidden_states.matmul(value_weight)?;
    let value = &value + value_bias;

    // Reshape for multi-head attention: (batch, seq, hidden) -> (batch, num_heads, seq, head_dim)
    let query = query.reshape(&[batch_size, seq_len, num_heads, head_dim])?;
    let query = query.transpose_axes(&[0, 2, 1, 3])?;

    let key = key.reshape(&[batch_size, seq_len, num_heads, head_dim])?;
    let key = key.transpose_axes(&[0, 2, 1, 3])?;

    let value = value.reshape(&[batch_size, seq_len, num_heads, head_dim])?;
    let value = value.transpose_axes(&[0, 2, 1, 3])?;

    // Attention scores: (batch, heads, seq, head_dim) @ (batch, heads, head_dim, seq)
    let key_t = key.transpose_axes(&[0, 1, 3, 2])?;
    let mut attention_scores = query.matmul(&key_t)?;

    // Scale
    let scale = Array::from_float((head_dim as f32).sqrt());
    attention_scores = &attention_scores / &scale;

    // Apply attention mask if provided
    if let Some(mask) = attention_mask {
        // Mask shape should be (batch, 1, 1, seq) or (batch, 1, seq, seq)
        attention_scores = &attention_scores + mask;
    }

    // Softmax
    let attention_probs = softmax(&attention_scores, -1)?;

    // Context: (batch, heads, seq, head_dim)
    let context = attention_probs.matmul(&value)?;

    // Reshape back: (batch, heads, seq, head_dim) -> (batch, seq, hidden)
    let context = context.transpose_axes(&[0, 2, 1, 3])?;
    context.reshape(&[batch_size, seq_len, hidden_size])
}

/// BERT attention output (projection + residual + layer norm).
pub fn bert_attention_output(
    hidden_states: &Array,
    input_tensor: &Array,
    dense_weight: &Array,
    dense_bias: &Array,
    layer_norm_weight: &Array,
    layer_norm_bias: &Array,
    config: &BertConfig,
) -> Result<Array> {
    // Dense projection
    let hidden_states = hidden_states.matmul(dense_weight)?;
    let hidden_states = &hidden_states + dense_bias;

    // Residual connection and layer norm
    let hidden_states = &hidden_states + input_tensor;
    layer_norm(&hidden_states, layer_norm_weight, layer_norm_bias, config.layer_norm_eps)
}

/// BERT intermediate layer (first part of feed-forward).
pub fn bert_intermediate(
    hidden_states: &Array,
    dense_weight: &Array,
    dense_bias: &Array,
) -> Result<Array> {
    let hidden_states = hidden_states.matmul(dense_weight)?;
    let hidden_states = &hidden_states + dense_bias;
    // GELU activation
    gelu(&hidden_states)
}

/// BERT output layer (second part of feed-forward with residual).
pub fn bert_output(
    hidden_states: &Array,
    input_tensor: &Array,
    dense_weight: &Array,
    dense_bias: &Array,
    layer_norm_weight: &Array,
    layer_norm_bias: &Array,
    config: &BertConfig,
) -> Result<Array> {
    let hidden_states = hidden_states.matmul(dense_weight)?;
    let hidden_states = &hidden_states + dense_bias;

    // Residual and layer norm
    let hidden_states = &hidden_states + input_tensor;
    layer_norm(&hidden_states, layer_norm_weight, layer_norm_bias, config.layer_norm_eps)
}

/// BERT transformer layer.
///
/// A single BERT layer consisting of:
/// 1. Self-attention
/// 2. Attention output (projection + residual + layer norm)
/// 3. Intermediate (feed-forward part 1)
/// 4. Output (feed-forward part 2 + residual + layer norm)
pub fn bert_layer(
    hidden_states: &Array,
    layer_weights: &BertLayerWeights,
    attention_mask: Option<&Array>,
    config: &BertConfig,
) -> Result<Array> {
    // Self-attention
    let attention_output = bert_self_attention(
        hidden_states,
        &layer_weights.attention_query_weight,
        &layer_weights.attention_query_bias,
        &layer_weights.attention_key_weight,
        &layer_weights.attention_key_bias,
        &layer_weights.attention_value_weight,
        &layer_weights.attention_value_bias,
        attention_mask,
        config,
    )?;

    // Attention output projection
    let attention_output = bert_attention_output(
        &attention_output,
        hidden_states,
        &layer_weights.attention_output_dense_weight,
        &layer_weights.attention_output_dense_bias,
        &layer_weights.attention_output_layer_norm_weight,
        &layer_weights.attention_output_layer_norm_bias,
        config,
    )?;

    // Intermediate
    let intermediate_output = bert_intermediate(
        &attention_output,
        &layer_weights.intermediate_dense_weight,
        &layer_weights.intermediate_dense_bias,
    )?;

    // Output
    bert_output(
        &intermediate_output,
        &attention_output,
        &layer_weights.output_dense_weight,
        &layer_weights.output_dense_bias,
        &layer_weights.output_layer_norm_weight,
        &layer_weights.output_layer_norm_bias,
        config,
    )
}

/// BERT pooler - extracts the [CLS] token representation.
pub fn bert_pooler(
    hidden_states: &Array,
    pooler_dense_weight: &Array,
    pooler_dense_bias: &Array,
) -> Result<Array> {
    // Take the first token ([CLS]) representation
    let shape = hidden_states.shape();
    let batch_size = shape[0] as i32;
    let hidden_size = shape[2] as i32;

    // Slice to get first token: hidden_states[:, 0, :]
    let first_token = hidden_states.slice(&[0, 0, 0], &[batch_size, 1, hidden_size], None)?;
    let first_token = first_token.reshape(&[batch_size, hidden_size])?;

    // Dense + tanh
    let pooled = first_token.matmul(pooler_dense_weight)?;
    let pooled = &pooled + pooler_dense_bias;
    tanh(&pooled)
}

/// Create an attention mask for BERT.
///
/// Converts a padding mask (1 for real tokens, 0 for padding) to an attention mask
/// with 0 for real tokens and -inf for padding.
pub fn create_bert_attention_mask(attention_mask: &Array) -> Result<Array> {
    // attention_mask: (batch, seq_len) with 1 for real tokens, 0 for padding
    // output: (batch, 1, 1, seq_len) with 0 for real tokens, -inf for padding

    let shape = attention_mask.shape();
    let batch_size = shape[0] as i32;
    let seq_len = shape[1] as i32;

    // Reshape to (batch, 1, 1, seq_len)
    let mask = attention_mask.reshape(&[batch_size, 1, 1, seq_len])?;

    // Convert: (1 - mask) * -10000.0
    // Where mask is 1, result is 0; where mask is 0, result is -10000
    let ones = Array::ones::<f32>(&[1])?;
    let inverted = &ones - &mask;
    let neg_inf = Array::from_float(-10000.0f32);
    Ok(&inverted * &neg_inf)
}

/// Weights for a single BERT transformer layer.
#[derive(Debug)]
pub struct BertLayerWeights {
    // Self-attention
    pub attention_query_weight: Array,
    pub attention_query_bias: Array,
    pub attention_key_weight: Array,
    pub attention_key_bias: Array,
    pub attention_value_weight: Array,
    pub attention_value_bias: Array,
    // Attention output
    pub attention_output_dense_weight: Array,
    pub attention_output_dense_bias: Array,
    pub attention_output_layer_norm_weight: Array,
    pub attention_output_layer_norm_bias: Array,
    // Intermediate (FFN part 1)
    pub intermediate_dense_weight: Array,
    pub intermediate_dense_bias: Array,
    // Output (FFN part 2)
    pub output_dense_weight: Array,
    pub output_dense_bias: Array,
    pub output_layer_norm_weight: Array,
    pub output_layer_norm_bias: Array,
}

impl BertLayerWeights {
    /// Initialize random weights for testing.
    pub fn random(config: &BertConfig) -> Result<Self> {
        let hidden = config.hidden_size;
        let intermediate = config.intermediate_size;

        Ok(Self {
            attention_query_weight: crate::random::normal_with_params::<f32>(&[hidden, hidden], 0.0, 0.02, None)?,
            attention_query_bias: Array::zeros::<f32>(&[hidden])?,
            attention_key_weight: crate::random::normal_with_params::<f32>(&[hidden, hidden], 0.0, 0.02, None)?,
            attention_key_bias: Array::zeros::<f32>(&[hidden])?,
            attention_value_weight: crate::random::normal_with_params::<f32>(&[hidden, hidden], 0.0, 0.02, None)?,
            attention_value_bias: Array::zeros::<f32>(&[hidden])?,
            attention_output_dense_weight: crate::random::normal_with_params::<f32>(&[hidden, hidden], 0.0, 0.02, None)?,
            attention_output_dense_bias: Array::zeros::<f32>(&[hidden])?,
            attention_output_layer_norm_weight: Array::ones::<f32>(&[hidden])?,
            attention_output_layer_norm_bias: Array::zeros::<f32>(&[hidden])?,
            intermediate_dense_weight: crate::random::normal_with_params::<f32>(&[hidden, intermediate], 0.0, 0.02, None)?,
            intermediate_dense_bias: Array::zeros::<f32>(&[intermediate])?,
            output_dense_weight: crate::random::normal_with_params::<f32>(&[intermediate, hidden], 0.0, 0.02, None)?,
            output_dense_bias: Array::zeros::<f32>(&[hidden])?,
            output_layer_norm_weight: Array::ones::<f32>(&[hidden])?,
            output_layer_norm_bias: Array::zeros::<f32>(&[hidden])?,
        })
    }
}

/// Full BERT model weights.
#[derive(Debug)]
pub struct BertWeights {
    // Embeddings
    pub word_embeddings: Array,
    pub position_embeddings: Array,
    pub token_type_embeddings: Array,
    pub embeddings_layer_norm_weight: Array,
    pub embeddings_layer_norm_bias: Array,
    // Transformer layers
    pub layers: Vec<BertLayerWeights>,
    // Pooler
    pub pooler_dense_weight: Array,
    pub pooler_dense_bias: Array,
}

impl BertWeights {
    /// Initialize random weights for testing.
    pub fn random(config: &BertConfig) -> Result<Self> {
        let vocab = config.vocab_size;
        let hidden = config.hidden_size;
        let max_pos = config.max_position_embeddings;
        let type_vocab = config.type_vocab_size;

        let mut layers = Vec::with_capacity(config.num_hidden_layers as usize);
        for _ in 0..config.num_hidden_layers {
            layers.push(BertLayerWeights::random(config)?);
        }

        Ok(Self {
            word_embeddings: crate::random::normal_with_params::<f32>(&[vocab, hidden], 0.0, 0.02, None)?,
            position_embeddings: crate::random::normal_with_params::<f32>(&[max_pos, hidden], 0.0, 0.02, None)?,
            token_type_embeddings: crate::random::normal_with_params::<f32>(&[type_vocab, hidden], 0.0, 0.02, None)?,
            embeddings_layer_norm_weight: Array::ones::<f32>(&[hidden])?,
            embeddings_layer_norm_bias: Array::zeros::<f32>(&[hidden])?,
            layers,
            pooler_dense_weight: crate::random::normal_with_params::<f32>(&[hidden, hidden], 0.0, 0.02, None)?,
            pooler_dense_bias: Array::zeros::<f32>(&[hidden])?,
        })
    }
}

/// BERT Model
///
/// Complete BERT model implementation for embeddings, classification, and NLU tasks.
///
/// # Example
/// ```ignore
/// use mlx_rs::nn::{BertConfig, BertModel, BertWeights};
///
/// let config = BertConfig::bert_base_uncased();
/// let weights = BertWeights::random(&config)?;
/// let model = BertModel::new(config);
///
/// let input_ids = Array::from_slice(&[101i32, 2054, 2003, 2023, 102], &[1, 5])?;
/// let (last_hidden, pooled) = model.forward(&input_ids, None, None, &weights)?;
/// ```
pub struct BertModel {
    pub config: BertConfig,
}

impl BertModel {
    /// Create a new BERT model.
    pub fn new(config: BertConfig) -> Self {
        Self { config }
    }

    /// Forward pass.
    ///
    /// # Arguments
    /// * `input_ids` - Token IDs of shape (batch, seq_len)
    /// * `token_type_ids` - Optional segment IDs of shape (batch, seq_len)
    /// * `attention_mask` - Optional attention mask (1 for real tokens, 0 for padding)
    /// * `weights` - Model weights
    ///
    /// # Returns
    /// A tuple of (last_hidden_state, pooled_output)
    /// - last_hidden_state: (batch, seq_len, hidden_size)
    /// - pooled_output: (batch, hidden_size) - [CLS] representation
    pub fn forward(
        &self,
        input_ids: &Array,
        token_type_ids: Option<&Array>,
        attention_mask: Option<&Array>,
        weights: &BertWeights,
    ) -> Result<(Array, Array)> {
        let shape = input_ids.shape();
        if shape.len() != 2 {
            return Err(Error::InvalidShape("Expected 2D input_ids (batch, seq_len)".into()));
        }

        // Embeddings
        let hidden_states = bert_embeddings(
            input_ids,
            token_type_ids,
            None,
            &weights.word_embeddings,
            &weights.position_embeddings,
            &weights.token_type_embeddings,
            &weights.embeddings_layer_norm_weight,
            &weights.embeddings_layer_norm_bias,
            &self.config,
        )?;

        // Create extended attention mask if provided
        let extended_attention_mask = attention_mask
            .map(|mask| create_bert_attention_mask(mask))
            .transpose()?;

        // Apply transformer layers
        let mut hidden_states = hidden_states;
        for layer_weights in &weights.layers {
            hidden_states = bert_layer(
                &hidden_states,
                layer_weights,
                extended_attention_mask.as_ref(),
                &self.config,
            )?;
        }

        // Pooler
        let pooled_output = bert_pooler(
            &hidden_states,
            &weights.pooler_dense_weight,
            &weights.pooler_dense_bias,
        )?;

        Ok((hidden_states, pooled_output))
    }

    /// Get embeddings for the input (last hidden state).
    ///
    /// # Returns
    /// last_hidden_state: (batch, seq_len, hidden_size)
    pub fn encode(
        &self,
        input_ids: &Array,
        token_type_ids: Option<&Array>,
        attention_mask: Option<&Array>,
        weights: &BertWeights,
    ) -> Result<Array> {
        let (last_hidden_state, _) = self.forward(input_ids, token_type_ids, attention_mask, weights)?;
        Ok(last_hidden_state)
    }

    /// Get the [CLS] token embedding (pooled output).
    ///
    /// This is commonly used for classification tasks.
    ///
    /// # Returns
    /// pooled_output: (batch, hidden_size)
    pub fn get_pooled_output(
        &self,
        input_ids: &Array,
        token_type_ids: Option<&Array>,
        attention_mask: Option<&Array>,
        weights: &BertWeights,
    ) -> Result<Array> {
        let (_, pooled_output) = self.forward(input_ids, token_type_ids, attention_mask, weights)?;
        Ok(pooled_output)
    }

    /// Get mean-pooled embeddings (average of all token embeddings).
    ///
    /// This is often used for sentence embeddings.
    ///
    /// # Returns
    /// mean_pooled: (batch, hidden_size)
    pub fn get_mean_pooled(
        &self,
        input_ids: &Array,
        token_type_ids: Option<&Array>,
        attention_mask: Option<&Array>,
        weights: &BertWeights,
    ) -> Result<Array> {
        let (last_hidden_state, _) = self.forward(input_ids, token_type_ids, attention_mask, weights)?;

        // Mean pooling over sequence dimension
        if let Some(mask) = attention_mask {
            // Masked mean pooling
            let shape = last_hidden_state.shape();
            let hidden_size = shape[2] as i32;

            // Expand mask to hidden dim
            let mask_expanded = mask.reshape(&[shape[0] as i32, shape[1] as i32, 1])?;
            let mask_expanded = broadcast_to(&mask_expanded, &[shape[0] as i32, shape[1] as i32, hidden_size])?;

            // Mask and sum
            let masked = &last_hidden_state * &mask_expanded;
            let sum = masked.sum_axes(&[1], false)?;

            // Count non-padding tokens per batch
            let counts = mask.sum_axes(&[1], true)?;
            let counts = counts.reshape(&[shape[0] as i32, 1])?;

            // Clamp counts to at least 1 to avoid division by zero
            // Use where_cond: if counts < 1, use 1, else use counts
            let one = Array::from_float(1.0f32);
            let condition = counts.lt(&one)?;
            let counts = crate::ops::where_cond(&condition, &one, &counts)?;

            Ok(&sum / &counts)
        } else {
            // Simple mean over sequence: sum / seq_len
            let shape = last_hidden_state.shape();
            let seq_len = shape[1] as f32;
            let sum = last_hidden_state.sum_axes(&[1], false)?;
            let seq_len_arr = Array::from_float(seq_len);
            Ok(&sum / &seq_len_arr)
        }
    }
}

// ============================================================================
// Vision Transformer (ViT) Implementation
// ============================================================================

/// Configuration for Vision Transformer models.
///
/// ViT (Vision Transformer) applies the transformer architecture to image
/// classification by treating images as sequences of patches.
///
/// # Example
/// ```ignore
/// use mlx_rs::nn::ViTConfig;
///
/// // Use a preset
/// let config = ViTConfig::vit_base_patch16_224();
///
/// // Or customize
/// let config = ViTConfig::new()
///     .image_size(224)
///     .patch_size(16)
///     .hidden_size(768)
///     .num_hidden_layers(12);
/// ```
#[derive(Debug, Clone)]
pub struct ViTConfig {
    /// Input image size (assumes square images)
    pub image_size: i32,
    /// Size of each patch (assumes square patches)
    pub patch_size: i32,
    /// Number of input channels (3 for RGB)
    pub num_channels: i32,
    /// Hidden size (embedding dimension)
    pub hidden_size: i32,
    /// Number of transformer layers
    pub num_hidden_layers: i32,
    /// Number of attention heads
    pub num_attention_heads: i32,
    /// Intermediate size in feed-forward layers
    pub intermediate_size: i32,
    /// Hidden activation function
    pub hidden_act: String,
    /// Dropout probability
    pub hidden_dropout_prob: f32,
    /// Attention dropout probability
    pub attention_probs_dropout_prob: f32,
    /// Layer normalization epsilon
    pub layer_norm_eps: f32,
    /// Number of classes for classification
    pub num_classes: i32,
}

impl Default for ViTConfig {
    fn default() -> Self {
        Self::new()
    }
}

impl ViTConfig {
    /// Create a new ViT configuration with default values.
    pub fn new() -> Self {
        Self {
            image_size: 224,
            patch_size: 16,
            num_channels: 3,
            hidden_size: 768,
            num_hidden_layers: 12,
            num_attention_heads: 12,
            intermediate_size: 3072,
            hidden_act: "gelu".to_string(),
            hidden_dropout_prob: 0.0,
            attention_probs_dropout_prob: 0.0,
            layer_norm_eps: 1e-6,
            num_classes: 1000,
        }
    }

    /// ViT-Base with 16x16 patches for 224x224 images (86M parameters).
    pub fn vit_base_patch16_224() -> Self {
        Self::new()
    }

    /// ViT-Base with 32x32 patches for 224x224 images.
    pub fn vit_base_patch32_224() -> Self {
        Self {
            patch_size: 32,
            ..Self::new()
        }
    }

    /// ViT-Large with 16x16 patches for 224x224 images (307M parameters).
    pub fn vit_large_patch16_224() -> Self {
        Self {
            hidden_size: 1024,
            num_hidden_layers: 24,
            num_attention_heads: 16,
            intermediate_size: 4096,
            ..Self::new()
        }
    }

    /// ViT-Huge with 14x14 patches for 224x224 images (632M parameters).
    pub fn vit_huge_patch14_224() -> Self {
        Self {
            image_size: 224,
            patch_size: 14,
            hidden_size: 1280,
            num_hidden_layers: 32,
            num_attention_heads: 16,
            intermediate_size: 5120,
            ..Self::new()
        }
    }

    /// ViT-Small (DeiT-Small) configuration.
    pub fn vit_small_patch16_224() -> Self {
        Self {
            hidden_size: 384,
            num_hidden_layers: 12,
            num_attention_heads: 6,
            intermediate_size: 1536,
            ..Self::new()
        }
    }

    /// ViT-Tiny configuration.
    pub fn vit_tiny_patch16_224() -> Self {
        Self {
            hidden_size: 192,
            num_hidden_layers: 12,
            num_attention_heads: 3,
            intermediate_size: 768,
            ..Self::new()
        }
    }

    // Builder methods
    pub fn image_size(mut self, image_size: i32) -> Self {
        self.image_size = image_size;
        self
    }

    pub fn patch_size(mut self, patch_size: i32) -> Self {
        self.patch_size = patch_size;
        self
    }

    pub fn num_channels(mut self, num_channels: i32) -> Self {
        self.num_channels = num_channels;
        self
    }

    pub fn hidden_size(mut self, hidden_size: i32) -> Self {
        self.hidden_size = hidden_size;
        self
    }

    pub fn num_hidden_layers(mut self, num_hidden_layers: i32) -> Self {
        self.num_hidden_layers = num_hidden_layers;
        self
    }

    pub fn num_attention_heads(mut self, num_attention_heads: i32) -> Self {
        self.num_attention_heads = num_attention_heads;
        self
    }

    pub fn intermediate_size(mut self, intermediate_size: i32) -> Self {
        self.intermediate_size = intermediate_size;
        self
    }

    pub fn num_classes(mut self, num_classes: i32) -> Self {
        self.num_classes = num_classes;
        self
    }

    pub fn layer_norm_eps(mut self, layer_norm_eps: f32) -> Self {
        self.layer_norm_eps = layer_norm_eps;
        self
    }

    /// Get the number of patches.
    pub fn num_patches(&self) -> i32 {
        (self.image_size / self.patch_size) * (self.image_size / self.patch_size)
    }

    /// Get the head dimension.
    pub fn head_dim(&self) -> i32 {
        self.hidden_size / self.num_attention_heads
    }
}

/// Patch embedding layer for ViT.
///
/// Converts an image into a sequence of patch embeddings using a convolution.
///
/// Note: MLX uses channels-last (NHWC) format for conv2d.
/// Input should be (batch, height, width, channels).
pub fn vit_patch_embedding(
    pixel_values: &Array,
    patch_embedding_weight: &Array,
    patch_embedding_bias: &Array,
    config: &ViTConfig,
) -> Result<Array> {
    let shape = pixel_values.shape();
    if shape.len() != 4 {
        return Err(Error::InvalidShape("Expected 4D input (batch, height, width, channels)".into()));
    }

    let batch_size = shape[0] as i32;
    let patch_size = config.patch_size;
    let hidden_size = config.hidden_size;

    // Use conv2d with kernel_size=patch_size and stride=patch_size
    // MLX conv2d expects: input (N, H, W, C), weight (O, kH, kW, C)
    // This extracts non-overlapping patches and projects them
    let patches = conv2d(
        pixel_values,
        patch_embedding_weight,
        (patch_size, patch_size),  // stride = patch_size
        (0, 0),                     // no padding
        (1, 1),                     // dilation
        1,                          // groups
    )?;

    // patches shape: (batch, num_patches_h, num_patches_w, hidden_size)
    let patches_shape = patches.shape();
    let num_patches_h = patches_shape[1] as i32;
    let num_patches_w = patches_shape[2] as i32;
    let num_patches = num_patches_h * num_patches_w;

    // Add bias (broadcasts to hidden_size dimension)
    let patches = &patches + patch_embedding_bias;

    // Reshape to (batch, num_patches, hidden_size)
    patches.reshape(&[batch_size, num_patches, hidden_size])
}

/// Add CLS token and position embeddings.
pub fn vit_embeddings(
    patch_embeddings: &Array,
    cls_token: &Array,
    position_embeddings: &Array,
) -> Result<Array> {
    let shape = patch_embeddings.shape();
    let batch_size = shape[0] as i32;
    let hidden_size = shape[2] as i32;

    // Expand CLS token to batch size: (1, 1, hidden) -> (batch, 1, hidden)
    let cls_tokens = broadcast_to(cls_token, &[batch_size, 1, hidden_size])?;

    // Concatenate CLS token with patch embeddings: (batch, 1+num_patches, hidden)
    let embeddings = crate::ops::concatenate(&[&cls_tokens, patch_embeddings], 1)?;

    // Add position embeddings
    Ok(&embeddings + position_embeddings)
}

/// ViT self-attention layer.
///
/// Same as BERT attention but without segment embeddings.
pub fn vit_self_attention(
    hidden_states: &Array,
    query_weight: &Array,
    query_bias: &Array,
    key_weight: &Array,
    key_bias: &Array,
    value_weight: &Array,
    value_bias: &Array,
    config: &ViTConfig,
) -> Result<Array> {
    let shape = hidden_states.shape();
    let batch_size = shape[0] as i32;
    let seq_len = shape[1] as i32;
    let hidden_size = config.hidden_size;
    let num_heads = config.num_attention_heads;
    let head_dim = config.head_dim();

    // Linear projections
    let query = hidden_states.matmul(query_weight)?;
    let query = &query + query_bias;

    let key = hidden_states.matmul(key_weight)?;
    let key = &key + key_bias;

    let value = hidden_states.matmul(value_weight)?;
    let value = &value + value_bias;

    // Reshape for multi-head attention
    let query = query.reshape(&[batch_size, seq_len, num_heads, head_dim])?;
    let query = query.transpose_axes(&[0, 2, 1, 3])?;

    let key = key.reshape(&[batch_size, seq_len, num_heads, head_dim])?;
    let key = key.transpose_axes(&[0, 2, 1, 3])?;

    let value = value.reshape(&[batch_size, seq_len, num_heads, head_dim])?;
    let value = value.transpose_axes(&[0, 2, 1, 3])?;

    // Attention scores
    let key_t = key.transpose_axes(&[0, 1, 3, 2])?;
    let attention_scores = query.matmul(&key_t)?;

    // Scale
    let scale = Array::from_float((head_dim as f32).sqrt());
    let attention_scores = &attention_scores / &scale;

    // Softmax
    let attention_probs = softmax(&attention_scores, -1)?;

    // Context
    let context = attention_probs.matmul(&value)?;

    // Reshape back
    let context = context.transpose_axes(&[0, 2, 1, 3])?;
    context.reshape(&[batch_size, seq_len, hidden_size])
}

/// ViT attention output (projection + residual + layer norm).
pub fn vit_attention_output(
    hidden_states: &Array,
    input_tensor: &Array,
    dense_weight: &Array,
    dense_bias: &Array,
    layer_norm_weight: &Array,
    layer_norm_bias: &Array,
    config: &ViTConfig,
) -> Result<Array> {
    let hidden_states = hidden_states.matmul(dense_weight)?;
    let hidden_states = &hidden_states + dense_bias;
    let hidden_states = &hidden_states + input_tensor;
    layer_norm(&hidden_states, layer_norm_weight, layer_norm_bias, config.layer_norm_eps)
}

/// ViT MLP (feed-forward) layer.
pub fn vit_mlp(
    hidden_states: &Array,
    fc1_weight: &Array,
    fc1_bias: &Array,
    fc2_weight: &Array,
    fc2_bias: &Array,
) -> Result<Array> {
    let hidden = hidden_states.matmul(fc1_weight)?;
    let hidden = &hidden + fc1_bias;
    let hidden = gelu(&hidden)?;
    let hidden = hidden.matmul(fc2_weight)?;
    Ok(&hidden + fc2_bias)
}

/// ViT transformer layer.
pub fn vit_layer(
    hidden_states: &Array,
    layer_weights: &ViTLayerWeights,
    config: &ViTConfig,
) -> Result<Array> {
    // Pre-norm for attention
    let normed = layer_norm(
        hidden_states,
        &layer_weights.layernorm_before_weight,
        &layer_weights.layernorm_before_bias,
        config.layer_norm_eps,
    )?;

    // Self-attention
    let attention_output = vit_self_attention(
        &normed,
        &layer_weights.attention_query_weight,
        &layer_weights.attention_query_bias,
        &layer_weights.attention_key_weight,
        &layer_weights.attention_key_bias,
        &layer_weights.attention_value_weight,
        &layer_weights.attention_value_bias,
        config,
    )?;

    // Attention output projection
    let attention_output = attention_output.matmul(&layer_weights.attention_output_dense_weight)?;
    let attention_output = &attention_output + &layer_weights.attention_output_dense_bias;

    // Residual connection
    let hidden_states = hidden_states + &attention_output;

    // Pre-norm for MLP
    let normed = layer_norm(
        &hidden_states,
        &layer_weights.layernorm_after_weight,
        &layer_weights.layernorm_after_bias,
        config.layer_norm_eps,
    )?;

    // MLP
    let mlp_output = vit_mlp(
        &normed,
        &layer_weights.mlp_fc1_weight,
        &layer_weights.mlp_fc1_bias,
        &layer_weights.mlp_fc2_weight,
        &layer_weights.mlp_fc2_bias,
    )?;

    // Residual connection
    Ok(&hidden_states + &mlp_output)
}

/// Weights for a single ViT transformer layer.
#[derive(Debug)]
pub struct ViTLayerWeights {
    // LayerNorm before attention
    pub layernorm_before_weight: Array,
    pub layernorm_before_bias: Array,
    // Self-attention
    pub attention_query_weight: Array,
    pub attention_query_bias: Array,
    pub attention_key_weight: Array,
    pub attention_key_bias: Array,
    pub attention_value_weight: Array,
    pub attention_value_bias: Array,
    pub attention_output_dense_weight: Array,
    pub attention_output_dense_bias: Array,
    // LayerNorm after attention (before MLP)
    pub layernorm_after_weight: Array,
    pub layernorm_after_bias: Array,
    // MLP
    pub mlp_fc1_weight: Array,
    pub mlp_fc1_bias: Array,
    pub mlp_fc2_weight: Array,
    pub mlp_fc2_bias: Array,
}

impl ViTLayerWeights {
    /// Initialize random weights for testing.
    pub fn random(config: &ViTConfig) -> Result<Self> {
        let hidden = config.hidden_size;
        let intermediate = config.intermediate_size;

        Ok(Self {
            layernorm_before_weight: Array::ones::<f32>(&[hidden])?,
            layernorm_before_bias: Array::zeros::<f32>(&[hidden])?,
            attention_query_weight: crate::random::normal_with_params::<f32>(&[hidden, hidden], 0.0, 0.02, None)?,
            attention_query_bias: Array::zeros::<f32>(&[hidden])?,
            attention_key_weight: crate::random::normal_with_params::<f32>(&[hidden, hidden], 0.0, 0.02, None)?,
            attention_key_bias: Array::zeros::<f32>(&[hidden])?,
            attention_value_weight: crate::random::normal_with_params::<f32>(&[hidden, hidden], 0.0, 0.02, None)?,
            attention_value_bias: Array::zeros::<f32>(&[hidden])?,
            attention_output_dense_weight: crate::random::normal_with_params::<f32>(&[hidden, hidden], 0.0, 0.02, None)?,
            attention_output_dense_bias: Array::zeros::<f32>(&[hidden])?,
            layernorm_after_weight: Array::ones::<f32>(&[hidden])?,
            layernorm_after_bias: Array::zeros::<f32>(&[hidden])?,
            mlp_fc1_weight: crate::random::normal_with_params::<f32>(&[hidden, intermediate], 0.0, 0.02, None)?,
            mlp_fc1_bias: Array::zeros::<f32>(&[intermediate])?,
            mlp_fc2_weight: crate::random::normal_with_params::<f32>(&[intermediate, hidden], 0.0, 0.02, None)?,
            mlp_fc2_bias: Array::zeros::<f32>(&[hidden])?,
        })
    }
}

/// Full ViT model weights.
#[derive(Debug)]
pub struct ViTWeights {
    // Patch embedding (conv projection)
    pub patch_embedding_weight: Array,
    pub patch_embedding_bias: Array,
    // CLS token
    pub cls_token: Array,
    // Position embeddings
    pub position_embeddings: Array,
    // Transformer layers
    pub layers: Vec<ViTLayerWeights>,
    // Final layer norm
    pub layernorm_weight: Array,
    pub layernorm_bias: Array,
    // Classification head
    pub classifier_weight: Array,
    pub classifier_bias: Array,
}

impl ViTWeights {
    /// Initialize random weights for testing.
    pub fn random(config: &ViTConfig) -> Result<Self> {
        let hidden = config.hidden_size;
        let num_patches = config.num_patches();
        let num_classes = config.num_classes;
        let patch_size = config.patch_size;
        let num_channels = config.num_channels;

        let mut layers = Vec::with_capacity(config.num_hidden_layers as usize);
        for _ in 0..config.num_hidden_layers {
            layers.push(ViTLayerWeights::random(config)?);
        }

        Ok(Self {
            // Conv weight shape for MLX (NHWC): (out_channels, kernel_h, kernel_w, in_channels)
            patch_embedding_weight: crate::random::normal_with_params::<f32>(
                &[hidden, patch_size, patch_size, num_channels], 0.0, 0.02, None
            )?,
            patch_embedding_bias: Array::zeros::<f32>(&[hidden])?,
            cls_token: crate::random::normal_with_params::<f32>(&[1, 1, hidden], 0.0, 0.02, None)?,
            // Position embeddings for CLS + patches
            position_embeddings: crate::random::normal_with_params::<f32>(
                &[1, num_patches + 1, hidden], 0.0, 0.02, None
            )?,
            layers,
            layernorm_weight: Array::ones::<f32>(&[hidden])?,
            layernorm_bias: Array::zeros::<f32>(&[hidden])?,
            classifier_weight: crate::random::normal_with_params::<f32>(&[hidden, num_classes], 0.0, 0.02, None)?,
            classifier_bias: Array::zeros::<f32>(&[num_classes])?,
        })
    }
}

/// Vision Transformer Model
///
/// Complete ViT model implementation for image classification.
///
/// Note: MLX uses channels-last (NHWC) format for images.
///
/// # Example
/// ```ignore
/// use mlx_rs::nn::{ViTConfig, ViTModel, ViTWeights};
///
/// let config = ViTConfig::vit_base_patch16_224();
/// let weights = ViTWeights::random(&config)?;
/// let model = ViTModel::new(config);
///
/// // Input: (batch, height, width, channels) - NHWC format
/// let images = Array::zeros::<f32>(&[1, 224, 224, 3])?;
/// let logits = model.forward(&images, &weights)?;
/// ```
pub struct ViTModel {
    pub config: ViTConfig,
}

impl ViTModel {
    /// Create a new ViT model.
    pub fn new(config: ViTConfig) -> Self {
        Self { config }
    }

    /// Forward pass for classification.
    ///
    /// # Arguments
    /// * `pixel_values` - Input images of shape (batch, height, width, channels) - NHWC format
    /// * `weights` - Model weights
    ///
    /// # Returns
    /// Classification logits of shape (batch, num_classes)
    pub fn forward(&self, pixel_values: &Array, weights: &ViTWeights) -> Result<Array> {
        let shape = pixel_values.shape();
        if shape.len() != 4 {
            return Err(Error::InvalidShape("Expected 4D input (batch, height, width, channels)".into()));
        }

        // Patch embedding
        let patch_embeddings = vit_patch_embedding(
            pixel_values,
            &weights.patch_embedding_weight,
            &weights.patch_embedding_bias,
            &self.config,
        )?;

        // Add CLS token and position embeddings
        let hidden_states = vit_embeddings(
            &patch_embeddings,
            &weights.cls_token,
            &weights.position_embeddings,
        )?;

        // Apply transformer layers
        let mut hidden_states = hidden_states;
        for layer_weights in &weights.layers {
            hidden_states = vit_layer(&hidden_states, layer_weights, &self.config)?;
        }

        // Final layer norm
        let hidden_states = layer_norm(
            &hidden_states,
            &weights.layernorm_weight,
            &weights.layernorm_bias,
            self.config.layer_norm_eps,
        )?;

        // Extract CLS token (first token)
        let cls_output = self.get_cls_token(&hidden_states)?;

        // Classification head
        let logits = cls_output.matmul(&weights.classifier_weight)?;
        Ok(&logits + &weights.classifier_bias)
    }

    /// Get the CLS token output (first token).
    fn get_cls_token(&self, hidden_states: &Array) -> Result<Array> {
        let shape = hidden_states.shape();
        let batch_size = shape[0] as i32;
        let hidden_size = shape[2] as i32;

        // Slice to get first token: hidden_states[:, 0, :]
        let cls_token = hidden_states.slice(&[0, 0, 0], &[batch_size, 1, hidden_size], None)?;
        cls_token.reshape(&[batch_size, hidden_size])
    }

    /// Get features (last hidden state) without classification head.
    ///
    /// # Arguments
    /// * `pixel_values` - Input images of shape (batch, height, width, channels) - NHWC format
    /// * `weights` - Model weights
    ///
    /// # Returns
    /// (cls_output, sequence_output)
    /// - cls_output: (batch, hidden_size) - CLS token representation
    /// - sequence_output: (batch, num_patches+1, hidden_size) - All token representations
    pub fn get_features(
        &self,
        pixel_values: &Array,
        weights: &ViTWeights,
    ) -> Result<(Array, Array)> {
        let shape = pixel_values.shape();
        if shape.len() != 4 {
            return Err(Error::InvalidShape("Expected 4D input (batch, height, width, channels)".into()));
        }

        // Patch embedding
        let patch_embeddings = vit_patch_embedding(
            pixel_values,
            &weights.patch_embedding_weight,
            &weights.patch_embedding_bias,
            &self.config,
        )?;

        // Add CLS token and position embeddings
        let hidden_states = vit_embeddings(
            &patch_embeddings,
            &weights.cls_token,
            &weights.position_embeddings,
        )?;

        // Apply transformer layers
        let mut hidden_states = hidden_states;
        for layer_weights in &weights.layers {
            hidden_states = vit_layer(&hidden_states, layer_weights, &self.config)?;
        }

        // Final layer norm
        let hidden_states = layer_norm(
            &hidden_states,
            &weights.layernorm_weight,
            &weights.layernorm_bias,
            self.config.layer_norm_eps,
        )?;

        // Extract CLS token
        let cls_output = self.get_cls_token(&hidden_states)?;

        Ok((cls_output, hidden_states))
    }

    /// Get patch embeddings (before transformer layers).
    ///
    /// Useful for feature extraction or visualization.
    pub fn get_patch_embeddings(
        &self,
        pixel_values: &Array,
        weights: &ViTWeights,
    ) -> Result<Array> {
        vit_patch_embedding(
            pixel_values,
            &weights.patch_embedding_weight,
            &weights.patch_embedding_bias,
            &self.config,
        )
    }
}

// =============================================================================
// Whisper Model (Speech Recognition)
// =============================================================================

/// Configuration for Whisper model.
///
/// Whisper is an encoder-decoder transformer for automatic speech recognition.
/// It takes mel spectrogram audio features as input and outputs text tokens.
///
/// # Example
///
/// ```ignore
/// use mlx_rs::nn::WhisperConfig;
///
/// // Use a preset configuration
/// let config = WhisperConfig::whisper_base();
///
/// // Or customize
/// let config = WhisperConfig::new()
///     .n_mels(80)
///     .n_audio_ctx(1500)
///     .n_audio_state(512)
///     .n_audio_head(8)
///     .n_audio_layer(6)
///     .n_vocab(51865)
///     .n_text_ctx(448)
///     .n_text_state(512)
///     .n_text_head(8)
///     .n_text_layer(6);
/// ```
#[derive(Debug, Clone)]
pub struct WhisperConfig {
    /// Number of mel frequency bins (default: 80)
    pub n_mels: i32,
    /// Audio context length (number of frames, default: 1500 for 30s)
    pub n_audio_ctx: i32,
    /// Audio encoder hidden size
    pub n_audio_state: i32,
    /// Number of audio encoder attention heads
    pub n_audio_head: i32,
    /// Number of audio encoder layers
    pub n_audio_layer: i32,
    /// Vocabulary size
    pub n_vocab: i32,
    /// Text context length (max tokens)
    pub n_text_ctx: i32,
    /// Text decoder hidden size
    pub n_text_state: i32,
    /// Number of text decoder attention heads
    pub n_text_head: i32,
    /// Number of text decoder layers
    pub n_text_layer: i32,
}

impl WhisperConfig {
    /// Create a new WhisperConfig with default values (base size).
    pub fn new() -> Self {
        Self {
            n_mels: 80,
            n_audio_ctx: 1500,
            n_audio_state: 512,
            n_audio_head: 8,
            n_audio_layer: 6,
            n_vocab: 51865,
            n_text_ctx: 448,
            n_text_state: 512,
            n_text_head: 8,
            n_text_layer: 6,
        }
    }

    /// Whisper tiny configuration (39M parameters).
    pub fn whisper_tiny() -> Self {
        Self {
            n_mels: 80,
            n_audio_ctx: 1500,
            n_audio_state: 384,
            n_audio_head: 6,
            n_audio_layer: 4,
            n_vocab: 51865,
            n_text_ctx: 448,
            n_text_state: 384,
            n_text_head: 6,
            n_text_layer: 4,
        }
    }

    /// Whisper base configuration (74M parameters).
    pub fn whisper_base() -> Self {
        Self::new()
    }

    /// Whisper small configuration (244M parameters).
    pub fn whisper_small() -> Self {
        Self {
            n_mels: 80,
            n_audio_ctx: 1500,
            n_audio_state: 768,
            n_audio_head: 12,
            n_audio_layer: 12,
            n_vocab: 51865,
            n_text_ctx: 448,
            n_text_state: 768,
            n_text_head: 12,
            n_text_layer: 12,
        }
    }

    /// Whisper medium configuration (769M parameters).
    pub fn whisper_medium() -> Self {
        Self {
            n_mels: 80,
            n_audio_ctx: 1500,
            n_audio_state: 1024,
            n_audio_head: 16,
            n_audio_layer: 24,
            n_vocab: 51865,
            n_text_ctx: 448,
            n_text_state: 1024,
            n_text_head: 16,
            n_text_layer: 24,
        }
    }

    /// Whisper large configuration (1550M parameters).
    pub fn whisper_large() -> Self {
        Self {
            n_mels: 80,
            n_audio_ctx: 1500,
            n_audio_state: 1280,
            n_audio_head: 20,
            n_audio_layer: 32,
            n_vocab: 51865,
            n_text_ctx: 448,
            n_text_state: 1280,
            n_text_head: 20,
            n_text_layer: 32,
        }
    }

    /// Whisper large-v2 configuration (same architecture as large).
    pub fn whisper_large_v2() -> Self {
        Self::whisper_large()
    }

    /// Whisper large-v3 configuration (128 mel bins).
    pub fn whisper_large_v3() -> Self {
        Self {
            n_mels: 128,
            n_audio_ctx: 1500,
            n_audio_state: 1280,
            n_audio_head: 20,
            n_audio_layer: 32,
            n_vocab: 51866,
            n_text_ctx: 448,
            n_text_state: 1280,
            n_text_head: 20,
            n_text_layer: 32,
        }
    }

    // Builder methods
    pub fn n_mels(mut self, n_mels: i32) -> Self {
        self.n_mels = n_mels;
        self
    }

    pub fn n_audio_ctx(mut self, n_audio_ctx: i32) -> Self {
        self.n_audio_ctx = n_audio_ctx;
        self
    }

    pub fn n_audio_state(mut self, n_audio_state: i32) -> Self {
        self.n_audio_state = n_audio_state;
        self
    }

    pub fn n_audio_head(mut self, n_audio_head: i32) -> Self {
        self.n_audio_head = n_audio_head;
        self
    }

    pub fn n_audio_layer(mut self, n_audio_layer: i32) -> Self {
        self.n_audio_layer = n_audio_layer;
        self
    }

    pub fn n_vocab(mut self, n_vocab: i32) -> Self {
        self.n_vocab = n_vocab;
        self
    }

    pub fn n_text_ctx(mut self, n_text_ctx: i32) -> Self {
        self.n_text_ctx = n_text_ctx;
        self
    }

    pub fn n_text_state(mut self, n_text_state: i32) -> Self {
        self.n_text_state = n_text_state;
        self
    }

    pub fn n_text_head(mut self, n_text_head: i32) -> Self {
        self.n_text_head = n_text_head;
        self
    }

    pub fn n_text_layer(mut self, n_text_layer: i32) -> Self {
        self.n_text_layer = n_text_layer;
        self
    }

    /// Get the head dimension for audio encoder.
    pub fn audio_head_dim(&self) -> i32 {
        self.n_audio_state / self.n_audio_head
    }

    /// Get the head dimension for text decoder.
    pub fn text_head_dim(&self) -> i32 {
        self.n_text_state / self.n_text_head
    }
}

impl Default for WhisperConfig {
    fn default() -> Self {
        Self::new()
    }
}

/// Weights for a single Whisper encoder layer.
#[derive(Debug, Clone)]
pub struct WhisperEncoderLayerWeights {
    /// Self-attention layer norm weight
    pub attn_ln_weight: Array,
    /// Self-attention layer norm bias
    pub attn_ln_bias: Array,
    /// Query projection weight
    pub attn_q_weight: Array,
    /// Key projection weight
    pub attn_k_weight: Array,
    /// Value projection weight
    pub attn_v_weight: Array,
    /// Output projection weight
    pub attn_out_weight: Array,
    /// Output projection bias
    pub attn_out_bias: Array,
    /// MLP layer norm weight
    pub mlp_ln_weight: Array,
    /// MLP layer norm bias
    pub mlp_ln_bias: Array,
    /// MLP first linear weight
    pub mlp_fc1_weight: Array,
    /// MLP first linear bias
    pub mlp_fc1_bias: Array,
    /// MLP second linear weight
    pub mlp_fc2_weight: Array,
    /// MLP second linear bias
    pub mlp_fc2_bias: Array,
}

/// Weights for a single Whisper decoder layer.
#[derive(Debug, Clone)]
pub struct WhisperDecoderLayerWeights {
    /// Self-attention layer norm weight
    pub attn_ln_weight: Array,
    /// Self-attention layer norm bias
    pub attn_ln_bias: Array,
    /// Self-attention query projection weight
    pub attn_q_weight: Array,
    /// Self-attention key projection weight
    pub attn_k_weight: Array,
    /// Self-attention value projection weight
    pub attn_v_weight: Array,
    /// Self-attention output projection weight
    pub attn_out_weight: Array,
    /// Self-attention output projection bias
    pub attn_out_bias: Array,
    /// Cross-attention layer norm weight
    pub cross_attn_ln_weight: Array,
    /// Cross-attention layer norm bias
    pub cross_attn_ln_bias: Array,
    /// Cross-attention query projection weight
    pub cross_attn_q_weight: Array,
    /// Cross-attention key projection weight
    pub cross_attn_k_weight: Array,
    /// Cross-attention value projection weight
    pub cross_attn_v_weight: Array,
    /// Cross-attention output projection weight
    pub cross_attn_out_weight: Array,
    /// Cross-attention output projection bias
    pub cross_attn_out_bias: Array,
    /// MLP layer norm weight
    pub mlp_ln_weight: Array,
    /// MLP layer norm bias
    pub mlp_ln_bias: Array,
    /// MLP first linear weight
    pub mlp_fc1_weight: Array,
    /// MLP first linear bias
    pub mlp_fc1_bias: Array,
    /// MLP second linear weight
    pub mlp_fc2_weight: Array,
    /// MLP second linear bias
    pub mlp_fc2_bias: Array,
}

/// All weights for the Whisper model.
#[derive(Debug, Clone)]
pub struct WhisperWeights {
    /// First conv layer weight (audio encoder)
    pub encoder_conv1_weight: Array,
    /// First conv layer bias (audio encoder)
    pub encoder_conv1_bias: Array,
    /// Second conv layer weight (audio encoder)
    pub encoder_conv2_weight: Array,
    /// Second conv layer bias (audio encoder)
    pub encoder_conv2_bias: Array,
    /// Audio encoder positional embeddings
    pub encoder_positional_embedding: Array,
    /// Audio encoder layer weights
    pub encoder_layers: Vec<WhisperEncoderLayerWeights>,
    /// Audio encoder final layer norm weight
    pub encoder_ln_weight: Array,
    /// Audio encoder final layer norm bias
    pub encoder_ln_bias: Array,
    /// Text decoder token embeddings
    pub decoder_token_embedding: Array,
    /// Text decoder positional embeddings
    pub decoder_positional_embedding: Array,
    /// Text decoder layer weights
    pub decoder_layers: Vec<WhisperDecoderLayerWeights>,
    /// Text decoder final layer norm weight
    pub decoder_ln_weight: Array,
    /// Text decoder final layer norm bias
    pub decoder_ln_bias: Array,
}

impl WhisperWeights {
    /// Create random weights for testing.
    pub fn random(config: &WhisperConfig) -> Result<Self> {
        use rand::Rng;
        let mut rng = rand::thread_rng();
        let scale = 0.02f32;

        // Helper to create random array
        let mut random_array = |shape: &[i32]| -> Result<Array> {
            let size: i32 = shape.iter().product();
            let data: Vec<f32> = (0..size).map(|_| rng.gen::<f32>() * scale - scale / 2.0).collect();
            Array::from_slice(&data, shape)
        };

        // Encoder conv layers
        // Conv1d weight shape: (out_channels, kernel_size, in_channels) for MLX
        let encoder_conv1_weight = random_array(&[config.n_audio_state, 3, config.n_mels])?;
        let encoder_conv1_bias = random_array(&[config.n_audio_state])?;
        let encoder_conv2_weight = random_array(&[config.n_audio_state, 3, config.n_audio_state])?;
        let encoder_conv2_bias = random_array(&[config.n_audio_state])?;

        // Encoder positional embedding
        let encoder_positional_embedding = random_array(&[config.n_audio_ctx, config.n_audio_state])?;

        // Encoder layers
        let mut encoder_layers = Vec::new();
        for _ in 0..config.n_audio_layer {
            encoder_layers.push(WhisperEncoderLayerWeights {
                attn_ln_weight: random_array(&[config.n_audio_state])?,
                attn_ln_bias: random_array(&[config.n_audio_state])?,
                attn_q_weight: random_array(&[config.n_audio_state, config.n_audio_state])?,
                attn_k_weight: random_array(&[config.n_audio_state, config.n_audio_state])?,
                attn_v_weight: random_array(&[config.n_audio_state, config.n_audio_state])?,
                attn_out_weight: random_array(&[config.n_audio_state, config.n_audio_state])?,
                attn_out_bias: random_array(&[config.n_audio_state])?,
                mlp_ln_weight: random_array(&[config.n_audio_state])?,
                mlp_ln_bias: random_array(&[config.n_audio_state])?,
                mlp_fc1_weight: random_array(&[config.n_audio_state * 4, config.n_audio_state])?,
                mlp_fc1_bias: random_array(&[config.n_audio_state * 4])?,
                mlp_fc2_weight: random_array(&[config.n_audio_state, config.n_audio_state * 4])?,
                mlp_fc2_bias: random_array(&[config.n_audio_state])?,
            });
        }

        let encoder_ln_weight = random_array(&[config.n_audio_state])?;
        let encoder_ln_bias = random_array(&[config.n_audio_state])?;

        // Decoder embeddings
        let decoder_token_embedding = random_array(&[config.n_vocab, config.n_text_state])?;
        let decoder_positional_embedding = random_array(&[config.n_text_ctx, config.n_text_state])?;

        // Decoder layers
        let mut decoder_layers = Vec::new();
        for _ in 0..config.n_text_layer {
            decoder_layers.push(WhisperDecoderLayerWeights {
                attn_ln_weight: random_array(&[config.n_text_state])?,
                attn_ln_bias: random_array(&[config.n_text_state])?,
                attn_q_weight: random_array(&[config.n_text_state, config.n_text_state])?,
                attn_k_weight: random_array(&[config.n_text_state, config.n_text_state])?,
                attn_v_weight: random_array(&[config.n_text_state, config.n_text_state])?,
                attn_out_weight: random_array(&[config.n_text_state, config.n_text_state])?,
                attn_out_bias: random_array(&[config.n_text_state])?,
                cross_attn_ln_weight: random_array(&[config.n_text_state])?,
                cross_attn_ln_bias: random_array(&[config.n_text_state])?,
                cross_attn_q_weight: random_array(&[config.n_text_state, config.n_text_state])?,
                cross_attn_k_weight: random_array(&[config.n_audio_state, config.n_text_state])?,
                cross_attn_v_weight: random_array(&[config.n_audio_state, config.n_text_state])?,
                cross_attn_out_weight: random_array(&[config.n_text_state, config.n_text_state])?,
                cross_attn_out_bias: random_array(&[config.n_text_state])?,
                mlp_ln_weight: random_array(&[config.n_text_state])?,
                mlp_ln_bias: random_array(&[config.n_text_state])?,
                mlp_fc1_weight: random_array(&[config.n_text_state * 4, config.n_text_state])?,
                mlp_fc1_bias: random_array(&[config.n_text_state * 4])?,
                mlp_fc2_weight: random_array(&[config.n_text_state, config.n_text_state * 4])?,
                mlp_fc2_bias: random_array(&[config.n_text_state])?,
            });
        }

        let decoder_ln_weight = random_array(&[config.n_text_state])?;
        let decoder_ln_bias = random_array(&[config.n_text_state])?;

        Ok(Self {
            encoder_conv1_weight,
            encoder_conv1_bias,
            encoder_conv2_weight,
            encoder_conv2_bias,
            encoder_positional_embedding,
            encoder_layers,
            encoder_ln_weight,
            encoder_ln_bias,
            decoder_token_embedding,
            decoder_positional_embedding,
            decoder_layers,
            decoder_ln_weight,
            decoder_ln_bias,
        })
    }
}

/// Sinusoidal positional embedding for Whisper.
#[allow(dead_code)]
fn whisper_sinusoidal_embedding(length: i32, dim: i32) -> Result<Array> {
    let half_dim = dim / 2;

    // Create position indices
    let positions = Array::arange::<f32>(0.0, length as f64, 1.0)?;

    // Create dimension indices and compute frequencies
    let dim_indices = Array::arange::<f32>(0.0, half_dim as f64, 1.0)?;
    let log_timescale = (10000.0f32).ln() / (half_dim as f32 - 1.0);
    let scale_factor = Array::from_float(-log_timescale);
    let inv_timescales = (&dim_indices * &scale_factor).exp()?;

    // Compute angles: positions * inv_timescales
    let positions_expanded = positions.reshape(&[length, 1])?;
    let inv_timescales_expanded = inv_timescales.reshape(&[1, half_dim])?;
    let angles = positions_expanded.matmul(&inv_timescales_expanded)?;

    // Concatenate sin and cos
    let sin_emb = angles.sin()?;
    let cos_emb = angles.cos()?;

    crate::ops::concatenate(&[&sin_emb, &cos_emb], -1)
}

/// Whisper encoder self-attention.
fn whisper_encoder_attention(
    x: &Array,
    weights: &WhisperEncoderLayerWeights,
    config: &WhisperConfig,
) -> Result<Array> {
    let shape = x.shape();
    let batch_size = shape[0];
    let seq_len = shape[1];
    let n_head = config.n_audio_head;
    let head_dim = config.audio_head_dim();

    // Layer norm
    let x_norm = layer_norm(x, &weights.attn_ln_weight, &weights.attn_ln_bias, 1e-5)?;

    // QKV projections
    let q = x_norm.matmul(&weights.attn_q_weight.transpose()?)?;
    let k = x_norm.matmul(&weights.attn_k_weight.transpose()?)?;
    let v = x_norm.matmul(&weights.attn_v_weight.transpose()?)?;

    // Reshape for multi-head attention: (batch, seq, n_head, head_dim) -> (batch, n_head, seq, head_dim)
    let q = q.reshape(&[batch_size, seq_len, n_head, head_dim])?.transpose_axes(&[0, 2, 1, 3])?;
    let k = k.reshape(&[batch_size, seq_len, n_head, head_dim])?.transpose_axes(&[0, 2, 1, 3])?;
    let v = v.reshape(&[batch_size, seq_len, n_head, head_dim])?.transpose_axes(&[0, 2, 1, 3])?;

    // Scaled dot-product attention
    let scale = Array::from_float((head_dim as f32).sqrt());
    let attn_weights = q.matmul(&k.transpose_axes(&[0, 1, 3, 2])?)?;
    let attn_weights = &attn_weights / &scale;
    let attn_weights = softmax(&attn_weights, -1)?;
    let attn_output = attn_weights.matmul(&v)?;

    // Reshape back: (batch, n_head, seq, head_dim) -> (batch, seq, hidden)
    let attn_output = attn_output.transpose_axes(&[0, 2, 1, 3])?;
    let attn_output = attn_output.reshape(&[batch_size, seq_len, config.n_audio_state])?;

    // Output projection
    let attn_output = attn_output.matmul(&weights.attn_out_weight.transpose()?)?;
    let attn_output = &attn_output + &weights.attn_out_bias;

    // Residual connection
    Ok(x + &attn_output)
}

/// Whisper encoder MLP.
fn whisper_encoder_mlp(
    x: &Array,
    weights: &WhisperEncoderLayerWeights,
) -> Result<Array> {
    // Layer norm
    let x_norm = layer_norm(x, &weights.mlp_ln_weight, &weights.mlp_ln_bias, 1e-5)?;

    // MLP: fc1 -> gelu -> fc2
    let hidden = x_norm.matmul(&weights.mlp_fc1_weight.transpose()?)?;
    let hidden = &hidden + &weights.mlp_fc1_bias;
    let hidden = gelu(&hidden)?;
    let output = hidden.matmul(&weights.mlp_fc2_weight.transpose()?)?;
    let output = &output + &weights.mlp_fc2_bias;

    // Residual connection
    Ok(x + &output)
}

/// Single Whisper encoder layer.
fn whisper_encoder_layer(
    x: &Array,
    weights: &WhisperEncoderLayerWeights,
    config: &WhisperConfig,
) -> Result<Array> {
    let x = whisper_encoder_attention(x, weights, config)?;
    whisper_encoder_mlp(&x, weights)
}

/// Whisper decoder self-attention (causal).
fn whisper_decoder_self_attention(
    x: &Array,
    weights: &WhisperDecoderLayerWeights,
    config: &WhisperConfig,
    mask: Option<&Array>,
) -> Result<Array> {
    let shape = x.shape();
    let batch_size = shape[0];
    let seq_len = shape[1];
    let n_head = config.n_text_head;
    let head_dim = config.text_head_dim();

    // Layer norm
    let x_norm = layer_norm(x, &weights.attn_ln_weight, &weights.attn_ln_bias, 1e-5)?;

    // QKV projections
    let q = x_norm.matmul(&weights.attn_q_weight.transpose()?)?;
    let k = x_norm.matmul(&weights.attn_k_weight.transpose()?)?;
    let v = x_norm.matmul(&weights.attn_v_weight.transpose()?)?;

    // Reshape for multi-head attention
    let q = q.reshape(&[batch_size, seq_len, n_head, head_dim])?.transpose_axes(&[0, 2, 1, 3])?;
    let k = k.reshape(&[batch_size, seq_len, n_head, head_dim])?.transpose_axes(&[0, 2, 1, 3])?;
    let v = v.reshape(&[batch_size, seq_len, n_head, head_dim])?.transpose_axes(&[0, 2, 1, 3])?;

    // Scaled dot-product attention with causal mask
    let scale = Array::from_float((head_dim as f32).sqrt());
    let attn_weights = q.matmul(&k.transpose_axes(&[0, 1, 3, 2])?)?;
    let mut attn_weights = &attn_weights / &scale;

    // Apply causal mask
    if let Some(m) = mask {
        attn_weights = &attn_weights + m;
    }

    let attn_weights = softmax(&attn_weights, -1)?;
    let attn_output = attn_weights.matmul(&v)?;

    // Reshape back
    let attn_output = attn_output.transpose_axes(&[0, 2, 1, 3])?;
    let attn_output = attn_output.reshape(&[batch_size, seq_len, config.n_text_state])?;

    // Output projection
    let attn_output = attn_output.matmul(&weights.attn_out_weight.transpose()?)?;
    let attn_output = &attn_output + &weights.attn_out_bias;

    // Residual connection
    Ok(x + &attn_output)
}

/// Whisper decoder cross-attention.
fn whisper_decoder_cross_attention(
    x: &Array,
    encoder_output: &Array,
    weights: &WhisperDecoderLayerWeights,
    config: &WhisperConfig,
) -> Result<Array> {
    let shape = x.shape();
    let batch_size = shape[0];
    let tgt_len = shape[1];
    let encoder_shape = encoder_output.shape();
    let src_len = encoder_shape[1];
    let n_head = config.n_text_head;
    let head_dim = config.text_head_dim();

    // Layer norm
    let x_norm = layer_norm(x, &weights.cross_attn_ln_weight, &weights.cross_attn_ln_bias, 1e-5)?;

    // Q from decoder, K/V from encoder
    let q = x_norm.matmul(&weights.cross_attn_q_weight.transpose()?)?;
    let k = encoder_output.matmul(&weights.cross_attn_k_weight.transpose()?)?;
    let v = encoder_output.matmul(&weights.cross_attn_v_weight.transpose()?)?;

    // Reshape for multi-head attention
    let q = q.reshape(&[batch_size, tgt_len, n_head, head_dim])?.transpose_axes(&[0, 2, 1, 3])?;
    let k = k.reshape(&[batch_size, src_len, n_head, head_dim])?.transpose_axes(&[0, 2, 1, 3])?;
    let v = v.reshape(&[batch_size, src_len, n_head, head_dim])?.transpose_axes(&[0, 2, 1, 3])?;

    // Scaled dot-product attention (no mask for cross-attention)
    let scale = Array::from_float((head_dim as f32).sqrt());
    let attn_weights = q.matmul(&k.transpose_axes(&[0, 1, 3, 2])?)?;
    let attn_weights = &attn_weights / &scale;
    let attn_weights = softmax(&attn_weights, -1)?;
    let attn_output = attn_weights.matmul(&v)?;

    // Reshape back
    let attn_output = attn_output.transpose_axes(&[0, 2, 1, 3])?;
    let attn_output = attn_output.reshape(&[batch_size, tgt_len, config.n_text_state])?;

    // Output projection
    let attn_output = attn_output.matmul(&weights.cross_attn_out_weight.transpose()?)?;
    let attn_output = &attn_output + &weights.cross_attn_out_bias;

    // Residual connection
    Ok(x + &attn_output)
}

/// Whisper decoder MLP.
fn whisper_decoder_mlp(
    x: &Array,
    weights: &WhisperDecoderLayerWeights,
) -> Result<Array> {
    // Layer norm
    let x_norm = layer_norm(x, &weights.mlp_ln_weight, &weights.mlp_ln_bias, 1e-5)?;

    // MLP: fc1 -> gelu -> fc2
    let hidden = x_norm.matmul(&weights.mlp_fc1_weight.transpose()?)?;
    let hidden = &hidden + &weights.mlp_fc1_bias;
    let hidden = gelu(&hidden)?;
    let output = hidden.matmul(&weights.mlp_fc2_weight.transpose()?)?;
    let output = &output + &weights.mlp_fc2_bias;

    // Residual connection
    Ok(x + &output)
}

/// Single Whisper decoder layer.
fn whisper_decoder_layer(
    x: &Array,
    encoder_output: &Array,
    weights: &WhisperDecoderLayerWeights,
    config: &WhisperConfig,
    mask: Option<&Array>,
) -> Result<Array> {
    let x = whisper_decoder_self_attention(x, weights, config, mask)?;
    let x = whisper_decoder_cross_attention(&x, encoder_output, weights, config)?;
    whisper_decoder_mlp(&x, weights)
}

/// Whisper model for automatic speech recognition.
///
/// # Example
///
/// ```ignore
/// use mlx_rs::nn::{WhisperConfig, WhisperModel, WhisperWeights};
/// use mlx_rs::Array;
///
/// let config = WhisperConfig::whisper_base();
/// let weights = WhisperWeights::random(&config).unwrap();
/// let model = WhisperModel::new(config);
///
/// // Mel spectrogram input: (batch, n_mels, n_frames)
/// let mel = Array::zeros::<f32>(&[1, 80, 3000]).unwrap();
///
/// // Encode audio
/// let audio_features = model.encode(&mel, &weights).unwrap();
///
/// // Decode with token IDs
/// let tokens = Array::from_slice(&[50258i32, 50259, 50359], &[1, 3]).unwrap();
/// let logits = model.decode(&tokens, &audio_features, &weights).unwrap();
/// ```
#[derive(Debug, Clone)]
pub struct WhisperModel {
    config: WhisperConfig,
}

impl WhisperModel {
    /// Create a new Whisper model with the given configuration.
    pub fn new(config: WhisperConfig) -> Self {
        Self { config }
    }

    /// Encode audio mel spectrogram to audio features.
    ///
    /// # Arguments
    /// * `mel` - Mel spectrogram of shape (batch, n_mels, n_frames)
    /// * `weights` - Model weights
    ///
    /// # Returns
    /// Audio features of shape (batch, n_audio_ctx, n_audio_state)
    pub fn encode(&self, mel: &Array, weights: &WhisperWeights) -> Result<Array> {
        let shape = mel.shape();
        if shape.len() != 3 {
            return Err(Error::InvalidShape("Expected 3D mel spectrogram (batch, n_mels, n_frames)".into()));
        }

        // Transpose to (batch, n_frames, n_mels) for processing
        let x = mel.transpose_axes(&[0, 2, 1])?;

        // Conv1d layers - MLX conv1d expects (batch, seq, channels)
        // First conv: kernel_size=3, stride=1, padding=1, dilation=1, groups=1
        let x = conv1d(&x, &weights.encoder_conv1_weight, 1, 1, 1, 1)?;
        let x = &x + &weights.encoder_conv1_bias;
        let x = gelu(&x)?;

        // Second conv: kernel_size=3, stride=2, padding=1, dilation=1, groups=1
        let x = conv1d(&x, &weights.encoder_conv2_weight, 2, 1, 1, 1)?;
        let x = &x + &weights.encoder_conv2_bias;
        let x = gelu(&x)?;

        // Add positional embeddings
        let seq_len = x.shape()[1];
        let pos_emb = if seq_len <= self.config.n_audio_ctx {
            // Slice positional embeddings: [0:seq_len, :]
            weights.encoder_positional_embedding.slice(
                &[0, 0],
                &[seq_len, self.config.n_audio_state],
                None,
            )?
        } else {
            return Err(Error::InvalidShape(format!(
                "Audio sequence length {} exceeds max context {}",
                seq_len, self.config.n_audio_ctx
            )));
        };
        let x = &x + &pos_emb;

        // Encoder layers
        let mut x = x;
        for layer_weights in &weights.encoder_layers {
            x = whisper_encoder_layer(&x, layer_weights, &self.config)?;
        }

        // Final layer norm
        layer_norm(&x, &weights.encoder_ln_weight, &weights.encoder_ln_bias, 1e-5)
    }

    /// Decode token IDs given audio features.
    ///
    /// # Arguments
    /// * `tokens` - Token IDs of shape (batch, seq_len)
    /// * `audio_features` - Encoded audio features from encode()
    /// * `weights` - Model weights
    ///
    /// # Returns
    /// Logits of shape (batch, seq_len, vocab_size)
    pub fn decode(
        &self,
        tokens: &Array,
        audio_features: &Array,
        weights: &WhisperWeights,
    ) -> Result<Array> {
        let shape = tokens.shape();
        if shape.len() != 2 {
            return Err(Error::InvalidShape("Expected 2D token IDs (batch, seq_len)".into()));
        }
        let seq_len = shape[1];

        // Token embeddings
        let x = embedding(&weights.decoder_token_embedding, tokens)?;

        // Add positional embeddings
        let pos_emb = if seq_len <= self.config.n_text_ctx {
            // Slice positional embeddings: [0:seq_len, :]
            weights.decoder_positional_embedding.slice(
                &[0, 0],
                &[seq_len, self.config.n_text_state],
                None,
            )?
        } else {
            return Err(Error::InvalidShape(format!(
                "Token sequence length {} exceeds max context {}",
                seq_len, self.config.n_text_ctx
            )));
        };
        let x = &x + &pos_emb;

        // Create causal mask
        let mask = self.create_causal_mask(seq_len)?;

        // Decoder layers
        let mut x = x;
        for layer_weights in &weights.decoder_layers {
            x = whisper_decoder_layer(&x, audio_features, layer_weights, &self.config, Some(&mask))?;
        }

        // Final layer norm
        let x = layer_norm(&x, &weights.decoder_ln_weight, &weights.decoder_ln_bias, 1e-5)?;

        // Project to vocabulary (weight tying with embedding)
        x.matmul(&weights.decoder_token_embedding.transpose()?)
    }

    /// Create a causal attention mask.
    fn create_causal_mask(&self, seq_len: i32) -> Result<Array> {
        let neg_inf = f32::NEG_INFINITY;
        let mut mask_data = vec![0.0f32; (seq_len * seq_len) as usize];

        for i in 0..seq_len {
            for j in 0..seq_len {
                if j > i {
                    mask_data[(i * seq_len + j) as usize] = neg_inf;
                }
            }
        }

        Array::from_slice(&mask_data, &[seq_len, seq_len])
    }

    /// Full forward pass: encode audio and decode to get logits for given tokens.
    ///
    /// # Arguments
    /// * `mel` - Mel spectrogram of shape (batch, n_mels, n_frames)
    /// * `tokens` - Token IDs of shape (batch, seq_len)
    /// * `weights` - Model weights
    ///
    /// # Returns
    /// Logits of shape (batch, seq_len, vocab_size)
    pub fn forward(
        &self,
        mel: &Array,
        tokens: &Array,
        weights: &WhisperWeights,
    ) -> Result<Array> {
        let audio_features = self.encode(mel, weights)?;
        self.decode(tokens, &audio_features, weights)
    }

    /// Get the configuration.
    pub fn config(&self) -> &WhisperConfig {
        &self.config
    }
}

// =============================================================================
// GPT-2 Model (Text Generation)
// =============================================================================

/// Configuration for GPT-2 model.
///
/// GPT-2 is a decoder-only transformer for autoregressive text generation.
///
/// # Example
///
/// ```ignore
/// use mlx_rs::nn::GPT2Config;
///
/// // Use a preset configuration
/// let config = GPT2Config::gpt2_small();
///
/// // Or customize
/// let config = GPT2Config::new()
///     .vocab_size(50257)
///     .n_positions(1024)
///     .n_embd(768)
///     .n_layer(12)
///     .n_head(12);
/// ```
#[derive(Debug, Clone)]
pub struct GPT2Config {
    /// Vocabulary size (default: 50257 for GPT-2)
    pub vocab_size: i32,
    /// Maximum sequence length (default: 1024)
    pub n_positions: i32,
    /// Embedding dimension / hidden size
    pub n_embd: i32,
    /// Number of transformer layers
    pub n_layer: i32,
    /// Number of attention heads
    pub n_head: i32,
    /// Dropout probability (default: 0.1)
    pub dropout: f32,
    /// Layer norm epsilon (default: 1e-5)
    pub layer_norm_eps: f32,
}

impl GPT2Config {
    /// Create a new GPT2Config with default values (small size).
    pub fn new() -> Self {
        Self {
            vocab_size: 50257,
            n_positions: 1024,
            n_embd: 768,
            n_layer: 12,
            n_head: 12,
            dropout: 0.1,
            layer_norm_eps: 1e-5,
        }
    }

    /// GPT-2 Small configuration (117M parameters).
    pub fn gpt2_small() -> Self {
        Self::new()
    }

    /// GPT-2 Medium configuration (345M parameters).
    pub fn gpt2_medium() -> Self {
        Self {
            vocab_size: 50257,
            n_positions: 1024,
            n_embd: 1024,
            n_layer: 24,
            n_head: 16,
            dropout: 0.1,
            layer_norm_eps: 1e-5,
        }
    }

    /// GPT-2 Large configuration (774M parameters).
    pub fn gpt2_large() -> Self {
        Self {
            vocab_size: 50257,
            n_positions: 1024,
            n_embd: 1280,
            n_layer: 36,
            n_head: 20,
            dropout: 0.1,
            layer_norm_eps: 1e-5,
        }
    }

    /// GPT-2 XL configuration (1.5B parameters).
    pub fn gpt2_xl() -> Self {
        Self {
            vocab_size: 50257,
            n_positions: 1024,
            n_embd: 1600,
            n_layer: 48,
            n_head: 25,
            dropout: 0.1,
            layer_norm_eps: 1e-5,
        }
    }

    // Builder methods
    pub fn vocab_size(mut self, vocab_size: i32) -> Self {
        self.vocab_size = vocab_size;
        self
    }

    pub fn n_positions(mut self, n_positions: i32) -> Self {
        self.n_positions = n_positions;
        self
    }

    pub fn n_embd(mut self, n_embd: i32) -> Self {
        self.n_embd = n_embd;
        self
    }

    pub fn n_layer(mut self, n_layer: i32) -> Self {
        self.n_layer = n_layer;
        self
    }

    pub fn n_head(mut self, n_head: i32) -> Self {
        self.n_head = n_head;
        self
    }

    pub fn dropout_prob(mut self, dropout: f32) -> Self {
        self.dropout = dropout;
        self
    }

    pub fn layer_norm_eps(mut self, eps: f32) -> Self {
        self.layer_norm_eps = eps;
        self
    }

    /// Get the head dimension.
    pub fn head_dim(&self) -> i32 {
        self.n_embd / self.n_head
    }

    /// Get the intermediate size (4x hidden for GPT-2).
    pub fn intermediate_size(&self) -> i32 {
        self.n_embd * 4
    }
}

impl Default for GPT2Config {
    fn default() -> Self {
        Self::new()
    }
}

/// Weights for a single GPT-2 transformer block.
#[derive(Debug, Clone)]
pub struct GPT2BlockWeights {
    /// Layer norm 1 weight (before attention)
    pub ln_1_weight: Array,
    /// Layer norm 1 bias
    pub ln_1_bias: Array,
    /// Attention query/key/value combined weight
    pub attn_c_attn_weight: Array,
    /// Attention query/key/value combined bias
    pub attn_c_attn_bias: Array,
    /// Attention output projection weight
    pub attn_c_proj_weight: Array,
    /// Attention output projection bias
    pub attn_c_proj_bias: Array,
    /// Layer norm 2 weight (before MLP)
    pub ln_2_weight: Array,
    /// Layer norm 2 bias
    pub ln_2_bias: Array,
    /// MLP first linear weight
    pub mlp_c_fc_weight: Array,
    /// MLP first linear bias
    pub mlp_c_fc_bias: Array,
    /// MLP second linear weight
    pub mlp_c_proj_weight: Array,
    /// MLP second linear bias
    pub mlp_c_proj_bias: Array,
}

/// All weights for the GPT-2 model.
#[derive(Debug, Clone)]
pub struct GPT2Weights {
    /// Token embeddings
    pub wte: Array,
    /// Position embeddings
    pub wpe: Array,
    /// Transformer block weights
    pub blocks: Vec<GPT2BlockWeights>,
    /// Final layer norm weight
    pub ln_f_weight: Array,
    /// Final layer norm bias
    pub ln_f_bias: Array,
}

impl GPT2Weights {
    /// Create random weights for testing.
    pub fn random(config: &GPT2Config) -> Result<Self> {
        use rand::Rng;
        let mut rng = rand::thread_rng();
        let scale = 0.02f32;

        let mut random_array = |shape: &[i32]| -> Result<Array> {
            let size: i32 = shape.iter().product();
            let data: Vec<f32> = (0..size).map(|_| rng.gen::<f32>() * scale - scale / 2.0).collect();
            Array::from_slice(&data, shape)
        };

        // Embeddings
        let wte = random_array(&[config.vocab_size, config.n_embd])?;
        let wpe = random_array(&[config.n_positions, config.n_embd])?;

        // Transformer blocks
        let mut blocks = Vec::new();
        for _ in 0..config.n_layer {
            blocks.push(GPT2BlockWeights {
                ln_1_weight: random_array(&[config.n_embd])?,
                ln_1_bias: random_array(&[config.n_embd])?,
                // Combined QKV projection: n_embd -> 3 * n_embd
                attn_c_attn_weight: random_array(&[config.n_embd, 3 * config.n_embd])?,
                attn_c_attn_bias: random_array(&[3 * config.n_embd])?,
                attn_c_proj_weight: random_array(&[config.n_embd, config.n_embd])?,
                attn_c_proj_bias: random_array(&[config.n_embd])?,
                ln_2_weight: random_array(&[config.n_embd])?,
                ln_2_bias: random_array(&[config.n_embd])?,
                mlp_c_fc_weight: random_array(&[config.n_embd, config.intermediate_size()])?,
                mlp_c_fc_bias: random_array(&[config.intermediate_size()])?,
                mlp_c_proj_weight: random_array(&[config.intermediate_size(), config.n_embd])?,
                mlp_c_proj_bias: random_array(&[config.n_embd])?,
            });
        }

        // Final layer norm
        let ln_f_weight = random_array(&[config.n_embd])?;
        let ln_f_bias = random_array(&[config.n_embd])?;

        Ok(Self {
            wte,
            wpe,
            blocks,
            ln_f_weight,
            ln_f_bias,
        })
    }
}

/// GPT-2 causal self-attention.
fn gpt2_attention(
    x: &Array,
    weights: &GPT2BlockWeights,
    config: &GPT2Config,
    mask: &Array,
) -> Result<Array> {
    let shape = x.shape();
    let batch_size = shape[0];
    let seq_len = shape[1];
    let n_head = config.n_head;
    let head_dim = config.head_dim();

    // Combined QKV projection
    let qkv = x.matmul(&weights.attn_c_attn_weight)?;
    let qkv = &qkv + &weights.attn_c_attn_bias;

    // Split into Q, K, V
    let q = qkv.slice(&[0, 0, 0], &[batch_size, seq_len, config.n_embd], None)?;
    let k = qkv.slice(&[0, 0, config.n_embd], &[batch_size, seq_len, 2 * config.n_embd], None)?;
    let v = qkv.slice(&[0, 0, 2 * config.n_embd], &[batch_size, seq_len, 3 * config.n_embd], None)?;

    // Reshape for multi-head attention: (batch, seq, n_head, head_dim) -> (batch, n_head, seq, head_dim)
    let q = q.reshape(&[batch_size, seq_len, n_head, head_dim])?.transpose_axes(&[0, 2, 1, 3])?;
    let k = k.reshape(&[batch_size, seq_len, n_head, head_dim])?.transpose_axes(&[0, 2, 1, 3])?;
    let v = v.reshape(&[batch_size, seq_len, n_head, head_dim])?.transpose_axes(&[0, 2, 1, 3])?;

    // Scaled dot-product attention
    let scale = Array::from_float((head_dim as f32).sqrt());
    let attn_weights = q.matmul(&k.transpose_axes(&[0, 1, 3, 2])?)?;
    let attn_weights = &attn_weights / &scale;

    // Apply causal mask
    let attn_weights = &attn_weights + mask;
    let attn_weights = softmax(&attn_weights, -1)?;

    let attn_output = attn_weights.matmul(&v)?;

    // Reshape back: (batch, n_head, seq, head_dim) -> (batch, seq, n_embd)
    let attn_output = attn_output.transpose_axes(&[0, 2, 1, 3])?;
    let attn_output = attn_output.reshape(&[batch_size, seq_len, config.n_embd])?;

    // Output projection
    let attn_output = attn_output.matmul(&weights.attn_c_proj_weight)?;
    Ok(&attn_output + &weights.attn_c_proj_bias)
}

/// GPT-2 MLP (feed-forward network).
fn gpt2_mlp(
    x: &Array,
    weights: &GPT2BlockWeights,
) -> Result<Array> {
    // First linear + GELU
    let hidden = x.matmul(&weights.mlp_c_fc_weight)?;
    let hidden = &hidden + &weights.mlp_c_fc_bias;
    let hidden = gelu(&hidden)?;

    // Second linear
    let output = hidden.matmul(&weights.mlp_c_proj_weight)?;
    Ok(&output + &weights.mlp_c_proj_bias)
}

/// Single GPT-2 transformer block.
fn gpt2_block(
    x: &Array,
    weights: &GPT2BlockWeights,
    config: &GPT2Config,
    mask: &Array,
) -> Result<Array> {
    // Pre-norm architecture (GPT-2 style)
    // Attention with residual
    let ln_1 = layer_norm(x, &weights.ln_1_weight, &weights.ln_1_bias, config.layer_norm_eps)?;
    let attn_out = gpt2_attention(&ln_1, weights, config, mask)?;
    let x = x + &attn_out;

    // MLP with residual
    let ln_2 = layer_norm(&x, &weights.ln_2_weight, &weights.ln_2_bias, config.layer_norm_eps)?;
    let mlp_out = gpt2_mlp(&ln_2, weights)?;
    Ok(&x + &mlp_out)
}

/// GPT-2 model for text generation.
///
/// # Example
///
/// ```ignore
/// use mlx_rs::nn::{GPT2Config, GPT2Model, GPT2Weights};
/// use mlx_rs::Array;
///
/// let config = GPT2Config::gpt2_small();
/// let weights = GPT2Weights::random(&config).unwrap();
/// let model = GPT2Model::new(config);
///
/// // Forward pass with token IDs
/// let tokens = Array::from_slice(&[15496i32, 11, 995], &[1, 3]).unwrap();
/// let logits = model.forward(&tokens, &weights).unwrap();
///
/// // Text generation
/// let output = model.generate(&tokens, &weights, 50, 0.8).unwrap();
/// ```
#[derive(Debug, Clone)]
pub struct GPT2Model {
    config: GPT2Config,
}

impl GPT2Model {
    /// Create a new GPT-2 model with the given configuration.
    pub fn new(config: GPT2Config) -> Self {
        Self { config }
    }

    /// Create a causal attention mask.
    fn create_causal_mask(&self, seq_len: i32) -> Result<Array> {
        let neg_inf = f32::NEG_INFINITY;
        let mut mask_data = vec![0.0f32; (seq_len * seq_len) as usize];

        for i in 0..seq_len {
            for j in 0..seq_len {
                if j > i {
                    mask_data[(i * seq_len + j) as usize] = neg_inf;
                }
            }
        }

        Array::from_slice(&mask_data, &[1, 1, seq_len, seq_len])
    }

    /// Forward pass through GPT-2.
    ///
    /// # Arguments
    /// * `input_ids` - Token IDs of shape (batch, seq_len)
    /// * `weights` - Model weights
    ///
    /// # Returns
    /// Logits of shape (batch, seq_len, vocab_size)
    pub fn forward(&self, input_ids: &Array, weights: &GPT2Weights) -> Result<Array> {
        let shape = input_ids.shape();
        if shape.len() != 2 {
            return Err(Error::InvalidShape("Expected 2D input (batch, seq_len)".into()));
        }
        let seq_len = shape[1];

        if seq_len > self.config.n_positions {
            return Err(Error::InvalidShape(format!(
                "Sequence length {} exceeds max positions {}",
                seq_len, self.config.n_positions
            )));
        }

        // Token embeddings
        let token_emb = embedding(&weights.wte, input_ids)?;

        // Position embeddings
        let positions = Array::arange::<i32>(0.0, seq_len as f64, 1.0)?;
        let positions = positions.reshape(&[1, seq_len])?;
        let pos_emb = embedding(&weights.wpe, &positions)?;

        // Combine embeddings
        let mut hidden = &token_emb + &pos_emb;

        // Create causal mask
        let mask = self.create_causal_mask(seq_len)?;

        // Transformer blocks
        for block_weights in &weights.blocks {
            hidden = gpt2_block(&hidden, block_weights, &self.config, &mask)?;
        }

        // Final layer norm
        let hidden = layer_norm(&hidden, &weights.ln_f_weight, &weights.ln_f_bias, self.config.layer_norm_eps)?;

        // Project to vocabulary (weight tying with token embedding)
        hidden.matmul(&weights.wte.transpose()?)
    }

    /// Generate text autoregressively.
    ///
    /// # Arguments
    /// * `input_ids` - Initial token IDs of shape (batch, seq_len)
    /// * `weights` - Model weights
    /// * `max_new_tokens` - Maximum number of tokens to generate
    /// * `temperature` - Sampling temperature (1.0 = no change, <1.0 = more deterministic)
    ///
    /// # Returns
    /// Generated token IDs of shape (batch, seq_len + max_new_tokens)
    pub fn generate(
        &self,
        input_ids: &Array,
        weights: &GPT2Weights,
        max_new_tokens: i32,
        temperature: f32,
    ) -> Result<Array> {
        let shape = input_ids.shape();
        if shape.len() != 2 {
            return Err(Error::InvalidShape("Expected 2D input (batch, seq_len)".into()));
        }
        let batch_size = shape[0];
        let mut current_len = shape[1];

        // Start with input tokens
        let mut all_tokens = input_ids.to_vec::<i32>()?;

        for _ in 0..max_new_tokens {
            if current_len >= self.config.n_positions {
                break;
            }

            // Create current sequence
            let current_tokens = Array::from_slice(&all_tokens, &[batch_size, current_len])?;

            // Forward pass
            let logits = self.forward(&current_tokens, weights)?;

            // Get logits for the last position
            let last_logits = logits.slice(
                &[0, current_len - 1, 0],
                &[batch_size, current_len, self.config.vocab_size],
                None,
            )?;
            let last_logits = last_logits.reshape(&[batch_size, self.config.vocab_size])?;

            // Apply temperature
            let temp = Array::from_float(temperature);
            let scaled_logits = &last_logits / &temp;

            // Softmax to get probabilities
            let probs = softmax(&scaled_logits, -1)?;

            // Greedy decoding: argmax along last dimension
            let next_token = probs.argmax_axis(-1, false)?;
            next_token.eval();

            // Append to all tokens (argmax returns UInt32, convert to i32)
            let next_token_vec: Vec<i32> = next_token
                .to_vec::<u32>()?
                .into_iter()
                .map(|x| x as i32)
                .collect();
            all_tokens.extend(next_token_vec);
            current_len += 1;
        }

        Array::from_slice(&all_tokens, &[batch_size, current_len])
    }

    /// Get the configuration.
    pub fn config(&self) -> &GPT2Config {
        &self.config
    }
}
