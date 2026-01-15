//! Automatic Differentiation Example
//!
//! This example demonstrates mlx-rs automatic differentiation capabilities.
//!
//! Run with: cargo run --example autodiff

use mlx_rs::{Array, transforms, nn};

fn main() -> mlx_rs::error::Result<()> {
    println!("=== MLX-RS Automatic Differentiation ===\n");

    // -------------------------------------------------------------------------
    // Basic Gradient Computation
    // -------------------------------------------------------------------------
    println!("1. Basic Gradient (f(x) = x^2)");
    println!("------------------------------");

    // f(x) = x^2, df/dx = 2x
    let f_square = |inputs: &[Array]| -> Vec<Array> {
        let x = &inputs[0];
        vec![x * x]
    };

    let x = Array::from_slice(&[3.0f32], &[1])?;
    let grads = transforms::grad(f_square, &[x.clone()])?;

    grads[0].eval();
    println!("x = 3.0");
    println!("f(x) = x^2 = 9.0");
    println!("df/dx at x=3: {:?} (expected: 6.0)", grads[0].to_vec::<f32>()?);

    // -------------------------------------------------------------------------
    // Value and Gradient
    // -------------------------------------------------------------------------
    println!("\n2. Value and Gradient Together");
    println!("------------------------------");

    let x = Array::from_slice(&[2.0f32], &[1])?;
    let (values, grads) = transforms::value_and_grad(f_square, &[x.clone()])?;

    values[0].eval();
    grads[0].eval();
    println!("x = 2.0");
    println!("f(x) = {:?} (expected: 4.0)", values[0].to_vec::<f32>()?);
    println!("df/dx = {:?} (expected: 4.0)", grads[0].to_vec::<f32>()?);

    // -------------------------------------------------------------------------
    // Gradient of More Complex Function
    // -------------------------------------------------------------------------
    println!("\n3. Complex Function (f(x) = sin(x) * x)");
    println!("---------------------------------------");

    // f(x) = sin(x) * x
    // df/dx = cos(x) * x + sin(x) (product rule)
    let f_sinx = |inputs: &[Array]| -> Vec<Array> {
        let x = &inputs[0];
        let sin_x = x.sin().unwrap();
        vec![&sin_x * x]
    };

    let x = Array::from_slice(&[std::f32::consts::PI / 4.0], &[1])?;
    let (values, grads) = transforms::value_and_grad(f_sinx, &[x.clone()])?;

    values[0].eval();
    grads[0].eval();

    let x_val = std::f32::consts::PI / 4.0;
    let expected_val = x_val.sin() * x_val;
    let expected_grad = x_val.cos() * x_val + x_val.sin();

    println!("x = pi/4 = {:.4}", x_val);
    println!("f(x) = sin(x) * x = {:?} (expected: {:.4})", values[0].to_vec::<f32>()?, expected_val);
    println!("df/dx = {:?} (expected: {:.4})", grads[0].to_vec::<f32>()?, expected_grad);

    // -------------------------------------------------------------------------
    // Multi-Input Gradient
    // -------------------------------------------------------------------------
    println!("\n4. Multi-Input Function (f(x, y) = x * y + x^2)");
    println!("------------------------------------------------");

    // f(x, y) = x * y + x^2
    // df/dx = y + 2x
    // df/dy = x
    let f_multi = |inputs: &[Array]| -> Vec<Array> {
        let x = &inputs[0];
        let y = &inputs[1];
        vec![x * y + x * x]
    };

    let x = Array::from_slice(&[2.0f32], &[1])?;
    let y = Array::from_slice(&[3.0f32], &[1])?;
    let grads = transforms::grad(f_multi, &[x.clone(), y.clone()])?;

    grads[0].eval();
    grads[1].eval();

    println!("x = 2.0, y = 3.0");
    println!("f(x, y) = x*y + x^2 = 2*3 + 4 = 10");
    println!("df/dx = {:?} (expected: y + 2x = 3 + 4 = 7.0)", grads[0].to_vec::<f32>()?);
    println!("df/dy = {:?} (expected: x = 2.0)", grads[1].to_vec::<f32>()?);

    // -------------------------------------------------------------------------
    // Vector-Jacobian Product (VJP)
    // -------------------------------------------------------------------------
    println!("\n5. Vector-Jacobian Product (VJP)");
    println!("--------------------------------");

    // f(x) = [x^2, x^3] (vector output)
    let f_vec = |inputs: &[Array]| -> Vec<Array> {
        let x = &inputs[0];
        let x2 = x * x;
        let x3 = &x2 * x;
        vec![x2, x3]
    };

    let x = Array::from_slice(&[2.0f32], &[1])?;
    let cotangent1 = Array::from_slice(&[1.0f32], &[1])?;  // "weight" for x^2
    let cotangent2 = Array::from_slice(&[1.0f32], &[1])?;  // "weight" for x^3

    let (primals, vjps) = transforms::vjp(f_vec, &[x.clone()], &[cotangent1, cotangent2])?;

    primals[0].eval();
    primals[1].eval();
    vjps[0].eval();

    println!("x = 2.0");
    println!("f(x) = [x^2, x^3] = [{:?}, {:?}]",
             primals[0].to_vec::<f32>()?,
             primals[1].to_vec::<f32>()?);
    println!("VJP with cotangents [1, 1]:");
    println!("  = 1 * d(x^2)/dx + 1 * d(x^3)/dx");
    println!("  = 2x + 3x^2 = 4 + 12 = {:?}", vjps[0].to_vec::<f32>()?);

    // -------------------------------------------------------------------------
    // Jacobian-Vector Product (JVP)
    // -------------------------------------------------------------------------
    println!("\n6. Jacobian-Vector Product (JVP)");
    println!("--------------------------------");

    let x = Array::from_slice(&[2.0f32], &[1])?;
    let tangent = Array::from_slice(&[1.0f32], &[1])?;  // direction vector

    let (primals, jvps) = transforms::jvp(f_vec, &[x.clone()], &[tangent])?;

    primals[0].eval();
    primals[1].eval();
    jvps[0].eval();
    jvps[1].eval();

    println!("x = 2.0, tangent = 1.0");
    println!("f(x) = [x^2, x^3] = [{:?}, {:?}]",
             primals[0].to_vec::<f32>()?,
             primals[1].to_vec::<f32>()?);
    println!("JVP (directional derivatives):");
    println!("  d(x^2)/dx * 1 = {:?} (expected: 2x = 4)", jvps[0].to_vec::<f32>()?);
    println!("  d(x^3)/dx * 1 = {:?} (expected: 3x^2 = 12)", jvps[1].to_vec::<f32>()?);

    // -------------------------------------------------------------------------
    // Neural Network Gradient
    // -------------------------------------------------------------------------
    println!("\n7. Neural Network Gradient");
    println!("--------------------------");

    // Simple loss: L = sum((W @ x - target)^2)
    let loss_fn = |inputs: &[Array]| -> Vec<Array> {
        let w = &inputs[0];
        let x = &inputs[1];
        let target = &inputs[2];

        let pred = w.matmul(x).unwrap();
        let diff = &pred - target;
        let sq = &diff * &diff;
        vec![sq.sum_all(false).unwrap()]
    };

    // 2x2 weight matrix, 2x1 input, 2x1 target
    let w = Array::from_slice(&[1.0f32, 0.0, 0.0, 1.0], &[2, 2])?;  // identity
    let x = Array::from_slice(&[1.0f32, 2.0], &[2, 1])?;
    let target = Array::from_slice(&[2.0f32, 3.0], &[2, 1])?;

    let (values, grads) = transforms::value_and_grad(loss_fn, &[w.clone(), x.clone(), target.clone()])?;

    values[0].eval();
    grads[0].eval();

    println!("W @ x = [1, 2], target = [2, 3]");
    println!("Loss = sum((pred - target)^2) = {:?}", values[0].to_vec::<f32>()?);
    println!("dL/dW shape: {:?}", grads[0].shape());
    println!("dL/dW: {:?}", grads[0].to_vec::<f32>()?);

    // -------------------------------------------------------------------------
    // Stop Gradient
    // -------------------------------------------------------------------------
    println!("\n8. Stop Gradient");
    println!("----------------");

    // f(x, y) = x * stop_gradient(y) + y
    // df/dx = stop_gradient(y) = y (treated as constant)
    // df/dy = 1 (gradient flows only through the +y term)
    let f_stop = |inputs: &[Array]| -> Vec<Array> {
        let x = &inputs[0];
        let y = &inputs[1];
        let y_stopped = transforms::stop_gradient(y);
        vec![x * &y_stopped + y]
    };

    let x = Array::from_slice(&[2.0f32], &[1])?;
    let y = Array::from_slice(&[3.0f32], &[1])?;

    let grads = transforms::grad(f_stop, &[x.clone(), y.clone()])?;
    grads[0].eval();
    grads[1].eval();

    println!("f(x, y) = x * stop_gradient(y) + y");
    println!("x = 2, y = 3");
    println!("df/dx = {:?} (y is treated as constant: 3.0)", grads[0].to_vec::<f32>()?);
    println!("df/dy = {:?} (only from +y term: 1.0)", grads[1].to_vec::<f32>()?);

    println!("\n=== Example Complete ===");
    Ok(())
}
