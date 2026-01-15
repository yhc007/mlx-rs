//! Optimizer Example
//!
//! This example demonstrates training a simple model with different optimizers.
//!
//! Run with: cargo run --example optimizer

use mlx_rs::{Array, nn, random, transforms};
use mlx_rs::nn::Optimizer;

fn main() -> mlx_rs::error::Result<()> {
    println!("=== MLX-RS Optimizer Example ===\n");

    // -------------------------------------------------------------------------
    // SGD Optimizer
    // -------------------------------------------------------------------------
    println!("1. SGD Optimizer");
    println!("----------------");

    let mut sgd = nn::SGD::new(0.1);
    println!("SGD with lr=0.1");

    // Simple optimization: minimize (x - 5)^2
    let mut x = Array::from_slice(&[0.0f32], &[1])?;

    for i in 0..10 {
        // Gradient of (x - 5)^2 is 2(x - 5)
        let five = Array::from_float(5.0);
        let diff = &x - &five;
        let two = Array::from_float(2.0);
        let grad = &two * &diff;
        grad.eval();

        let new_params = sgd.step(&[x.clone()], &[grad])?;
        x = new_params.into_iter().next().unwrap();
        x.eval();

        if i % 3 == 0 {
            println!("  Step {}: x = {:.4}", i, x.to_vec::<f32>()?[0]);
        }
    }
    println!("  Final x = {:.4} (target: 5.0)", x.to_vec::<f32>()?[0]);

    // -------------------------------------------------------------------------
    // SGD with Momentum
    // -------------------------------------------------------------------------
    println!("\n2. SGD with Momentum");
    println!("--------------------");

    let mut sgd_momentum = nn::SGD::new(0.1).momentum(0.9);
    println!("SGD with lr=0.1, momentum=0.9");

    let mut x = Array::from_slice(&[0.0f32], &[1])?;

    for i in 0..10 {
        let five = Array::from_float(5.0);
        let diff = &x - &five;
        let two = Array::from_float(2.0);
        let grad = &two * &diff;
        grad.eval();

        let new_params = sgd_momentum.step(&[x.clone()], &[grad])?;
        x = new_params.into_iter().next().unwrap();
        x.eval();

        if i % 3 == 0 {
            println!("  Step {}: x = {:.4}", i, x.to_vec::<f32>()?[0]);
        }
    }
    println!("  Final x = {:.4} (target: 5.0)", x.to_vec::<f32>()?[0]);

    // -------------------------------------------------------------------------
    // Adam Optimizer
    // -------------------------------------------------------------------------
    println!("\n3. Adam Optimizer");
    println!("-----------------");

    let mut adam = nn::Adam::new(0.5);
    println!("Adam with lr=0.5");

    let mut x = Array::from_slice(&[0.0f32], &[1])?;

    for i in 0..20 {
        let five = Array::from_float(5.0);
        let diff = &x - &five;
        let two = Array::from_float(2.0);
        let grad = &two * &diff;
        grad.eval();

        let new_params = adam.step(&[x.clone()], &[grad])?;
        x = new_params.into_iter().next().unwrap();
        x.eval();

        if i % 5 == 0 {
            println!("  Step {}: x = {:.4}", i, x.to_vec::<f32>()?[0]);
        }
    }
    println!("  Final x = {:.4} (target: 5.0)", x.to_vec::<f32>()?[0]);

    // -------------------------------------------------------------------------
    // AdamW Optimizer
    // -------------------------------------------------------------------------
    println!("\n4. AdamW Optimizer (with weight decay)");
    println!("--------------------------------------");

    let mut adamw = nn::AdamW::new(0.5).weight_decay(0.01);
    println!("AdamW with lr=0.5, weight_decay=0.01");

    let mut x = Array::from_slice(&[0.0f32], &[1])?;

    for i in 0..20 {
        let five = Array::from_float(5.0);
        let diff = &x - &five;
        let two = Array::from_float(2.0);
        let grad = &two * &diff;
        grad.eval();

        let new_params = adamw.step(&[x.clone()], &[grad])?;
        x = new_params.into_iter().next().unwrap();
        x.eval();

        if i % 5 == 0 {
            println!("  Step {}: x = {:.4}", i, x.to_vec::<f32>()?[0]);
        }
    }
    println!("  Final x = {:.4}", x.to_vec::<f32>()?[0]);

    // -------------------------------------------------------------------------
    // RMSprop Optimizer
    // -------------------------------------------------------------------------
    println!("\n5. RMSprop Optimizer");
    println!("--------------------");

    let mut rmsprop = nn::RMSprop::new(0.5);
    println!("RMSprop with lr=0.5");

    let mut x = Array::from_slice(&[0.0f32], &[1])?;

    for i in 0..20 {
        let five = Array::from_float(5.0);
        let diff = &x - &five;
        let two = Array::from_float(2.0);
        let grad = &two * &diff;
        grad.eval();

        let new_params = rmsprop.step(&[x.clone()], &[grad])?;
        x = new_params.into_iter().next().unwrap();
        x.eval();

        if i % 5 == 0 {
            println!("  Step {}: x = {:.4}", i, x.to_vec::<f32>()?[0]);
        }
    }
    println!("  Final x = {:.4} (target: 5.0)", x.to_vec::<f32>()?[0]);

    // -------------------------------------------------------------------------
    // Training a Linear Regression Model
    // -------------------------------------------------------------------------
    println!("\n6. Linear Regression Training");
    println!("-----------------------------");

    // Generate synthetic data: y = 2x + 3 + noise
    let n_samples = 100;
    let x_data = random::normal_with_params::<f32>(&[n_samples, 1], 0.0, 1.0, None)?;
    let noise = random::normal_with_params::<f32>(&[n_samples, 1], 0.0, 0.1, None)?;

    let two = Array::from_float(2.0);
    let three = Array::from_float(3.0);
    let y_data = &(&x_data * &two) + &three;
    let y_data = &y_data + &noise;
    y_data.eval();
    x_data.eval();

    // Initialize weights: w (slope), b (intercept)
    let mut w = Array::from_slice(&[0.0f32], &[1, 1])?;
    let mut b = Array::from_slice(&[0.0f32], &[1])?;

    let mut adam = nn::Adam::new(0.1);

    println!("True parameters: w=2.0, b=3.0");
    println!("Training linear regression...");

    for epoch in 0..100 {
        // Forward: y_pred = x @ w + b
        let y_pred = &x_data.matmul(&w)? + &b;

        // Loss: MSE
        let diff = &y_pred - &y_data;
        let loss = (&diff * &diff).mean_all(false)?;

        // Compute gradients manually
        // dL/dy_pred = 2 * (y_pred - y_data) / n
        let n_float = Array::from_float(n_samples as f32);
        let two = Array::from_float(2.0);
        let grad_pred = &(&two * &diff) / &n_float;

        // dL/dw = x^T @ grad_pred
        let grad_w = x_data.transpose()?.matmul(&grad_pred)?;

        // dL/db = sum(grad_pred)
        let grad_b = grad_pred.sum_all(false)?;

        grad_w.eval();
        grad_b.eval();

        // Update parameters
        let new_params = adam.step(&[w.clone(), b.clone()], &[grad_w, grad_b])?;
        w = new_params[0].clone();
        b = new_params[1].clone();

        if epoch % 20 == 0 {
            loss.eval();
            w.eval();
            b.eval();
            println!("  Epoch {}: loss={:.4}, w={:.4}, b={:.4}",
                     epoch,
                     loss.to_vec::<f32>()?[0],
                     w.to_vec::<f32>()?[0],
                     b.to_vec::<f32>()?[0]);
        }
    }

    w.eval();
    b.eval();
    println!("\nFinal parameters: w={:.4}, b={:.4}",
             w.to_vec::<f32>()?[0],
             b.to_vec::<f32>()?[0]);
    println!("True parameters:  w=2.0, b=3.0");

    // -------------------------------------------------------------------------
    // Optimizer Methods
    // -------------------------------------------------------------------------
    println!("\n7. Optimizer Methods");
    println!("--------------------");

    let mut opt = nn::Adam::new(0.001);

    println!("Initial learning rate: {}", opt.learning_rate());
    println!("Step count: {}", opt.step_count());

    // Dummy step
    let param = Array::from_slice(&[1.0f32], &[1])?;
    let grad = Array::from_slice(&[0.1f32], &[1])?;
    let _ = opt.step(&[param.clone()], &[grad.clone()])?;
    let _ = opt.step(&[param.clone()], &[grad.clone()])?;

    println!("After 2 steps: {}", opt.step_count());

    opt.set_learning_rate(0.0001);
    println!("New learning rate: {}", opt.learning_rate());

    opt.reset();
    println!("After reset - step count: {}", opt.step_count());

    println!("\n=== Example Complete ===");
    Ok(())
}
