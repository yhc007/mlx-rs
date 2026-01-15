//! Neural Network Example
//!
//! This example demonstrates building and training a simple neural network.
//!
//! Run with: cargo run --example neural_network

use mlx_rs::{Array, nn, random, transforms};

fn main() -> mlx_rs::error::Result<()> {
    println!("=== MLX-RS Neural Network Example ===\n");

    // -------------------------------------------------------------------------
    // Simple MLP Forward Pass
    // -------------------------------------------------------------------------
    println!("1. Simple MLP Forward Pass");
    println!("--------------------------");

    let batch_size = 4;
    let input_dim = 8;
    let hidden_dim = 16;
    let output_dim = 3;

    // Create random input
    let x = random::normal_with_params::<f32>(&[batch_size, input_dim], 0.0, 1.0, None)?;

    // Initialize weights
    let w1 = random::normal_with_params::<f32>(&[input_dim, hidden_dim], 0.0, 0.1, None)?;
    let w2 = random::normal_with_params::<f32>(&[hidden_dim, output_dim], 0.0, 0.1, None)?;

    // Forward pass: x -> Linear -> ReLU -> Linear -> Softmax
    let h = x.matmul(&w1)?;
    let h_relu = nn::relu(&h)?;
    let logits = h_relu.matmul(&w2)?;
    let probs = nn::softmax(&logits, -1)?;

    probs.eval();

    println!("Input shape: {:?}", x.shape());
    println!("Hidden shape: {:?}", h_relu.shape());
    println!("Output shape: {:?}", probs.shape());

    // Verify probabilities sum to 1
    let prob_sum = probs.sum_axes(&[1], false)?;
    prob_sum.eval();
    println!("Probability sums (should be ~1.0): {:?}", prob_sum.to_vec::<f32>()?);

    // -------------------------------------------------------------------------
    // Activation Functions
    // -------------------------------------------------------------------------
    println!("\n2. Activation Functions");
    println!("-----------------------");

    let x = Array::from_slice(&[-2.0f32, -1.0, 0.0, 1.0, 2.0], &[5])?;

    let relu_out = nn::relu(&x)?;
    let sigmoid_out = nn::sigmoid(&x)?;
    let tanh_out = nn::tanh(&x)?;
    let gelu_out = nn::gelu(&x)?;
    let silu_out = nn::silu(&x)?;

    relu_out.eval();
    sigmoid_out.eval();
    tanh_out.eval();
    gelu_out.eval();
    silu_out.eval();

    println!("Input: {:?}", x.to_vec::<f32>()?);
    println!("ReLU: {:?}", relu_out.to_vec::<f32>()?);
    println!("Sigmoid: {:?}", sigmoid_out.to_vec::<f32>()?);
    println!("Tanh: {:?}", tanh_out.to_vec::<f32>()?);
    println!("GELU: {:?}", gelu_out.to_vec::<f32>()?);
    println!("SiLU/Swish: {:?}", silu_out.to_vec::<f32>()?);

    // -------------------------------------------------------------------------
    // Normalization Layers
    // -------------------------------------------------------------------------
    println!("\n3. Normalization Layers");
    println!("-----------------------");

    let x = Array::from_slice(&[1.0f32, 2.0, 3.0, 4.0, 5.0, 6.0], &[2, 3])?;
    let gamma = Array::ones::<f32>(&[3])?;
    let beta = Array::zeros::<f32>(&[3])?;

    let ln_out = nn::layer_norm(&x, &gamma, &beta, 1e-5)?;
    ln_out.eval();

    println!("Input: {:?}", x.to_vec::<f32>()?);
    println!("LayerNorm output: {:?}", ln_out.to_vec::<f32>()?);

    // RMSNorm (used in Llama)
    let rms_weight = Array::ones::<f32>(&[3])?;
    let rms_out = nn::rms_norm(&x, &rms_weight, 1e-6)?;
    rms_out.eval();
    println!("RMSNorm output: {:?}", rms_out.to_vec::<f32>()?);

    // -------------------------------------------------------------------------
    // Loss Functions
    // -------------------------------------------------------------------------
    println!("\n4. Loss Functions");
    println!("-----------------");

    let predictions = Array::from_slice(&[0.9f32, 0.1, 0.2, 0.8], &[2, 2])?;
    let targets = Array::from_slice(&[1.0f32, 0.0, 0.0, 1.0], &[2, 2])?;

    let mse = nn::mse_loss(&predictions, &targets, "mean")?;
    let bce = nn::binary_cross_entropy(&predictions, &targets, "mean")?;

    mse.eval();
    bce.eval();

    println!("Predictions: {:?}", predictions.to_vec::<f32>()?);
    println!("Targets: {:?}", targets.to_vec::<f32>()?);
    println!("MSE Loss: {:?}", mse.to_vec::<f32>()?);
    println!("BCE Loss: {:?}", bce.to_vec::<f32>()?);

    // -------------------------------------------------------------------------
    // Dropout
    // -------------------------------------------------------------------------
    println!("\n5. Dropout");
    println!("----------");

    let x = Array::ones::<f32>(&[2, 5])?;

    // Training mode (p=0.5)
    let dropped = nn::dropout(&x, 0.5, true)?;
    dropped.eval();
    println!("Input (ones): {:?}", x.to_vec::<f32>()?);
    println!("After dropout (training, p=0.5): {:?}", dropped.to_vec::<f32>()?);

    // Eval mode (no dropout)
    let no_drop = nn::dropout(&x, 0.5, false)?;
    no_drop.eval();
    println!("After dropout (eval mode): {:?}", no_drop.to_vec::<f32>()?);

    // -------------------------------------------------------------------------
    // Convolution
    // -------------------------------------------------------------------------
    println!("\n6. Convolution (Conv2D)");
    println!("-----------------------");

    // MLX uses channels-last format: (N, H, W, C)
    let input = random::normal_with_params::<f32>(&[1, 8, 8, 3], 0.0, 1.0, None)?;
    // Weight: (C_out, kH, kW, C_in)
    let weight = random::normal_with_params::<f32>(&[16, 3, 3, 3], 0.0, 0.1, None)?;

    let output = nn::conv2d(&input, &weight, (1, 1), (1, 1), (1, 1), 1)?;
    output.eval();

    println!("Input shape (N, H, W, C): {:?}", input.shape());
    println!("Weight shape (C_out, kH, kW, C_in): {:?}", weight.shape());
    println!("Output shape: {:?}", output.shape());

    // -------------------------------------------------------------------------
    // Attention
    // -------------------------------------------------------------------------
    println!("\n7. Scaled Dot-Product Attention");
    println!("--------------------------------");

    let batch = 1;
    let num_heads = 2;
    let seq_len = 4;
    let head_dim = 8;

    // Attention expects 4D: (batch, num_heads, seq_len, head_dim)
    let q = random::normal_with_params::<f32>(&[batch, num_heads, seq_len, head_dim], 0.0, 0.1, None)?;
    let k = random::normal_with_params::<f32>(&[batch, num_heads, seq_len, head_dim], 0.0, 0.1, None)?;
    let v = random::normal_with_params::<f32>(&[batch, num_heads, seq_len, head_dim], 0.0, 0.1, None)?;

    // Standard attention
    let attn = nn::scaled_dot_product_attention(&q, &k, &v, None, nn::AttentionMask::None, None)?;
    attn.eval();
    println!("Q, K, V shape: {:?}", q.shape());
    println!("Attention output shape: {:?}", attn.shape());

    // Causal attention (for autoregressive models)
    let causal_attn = nn::scaled_dot_product_attention(&q, &k, &v, None, nn::AttentionMask::Causal, None)?;
    causal_attn.eval();
    println!("Causal attention output shape: {:?}", causal_attn.shape());

    // -------------------------------------------------------------------------
    // Embedding
    // -------------------------------------------------------------------------
    println!("\n8. Embedding Layer");
    println!("------------------");

    let vocab_size = 100;
    let embed_dim = 32;

    // Embedding table
    let embed_table = random::normal_with_params::<f32>(&[vocab_size, embed_dim], 0.0, 0.1, None)?;

    // Token indices
    let tokens = Array::from_slice(&[5i32, 10, 15, 20], &[1, 4])?;

    // Look up embeddings
    let embeddings = nn::embedding(&embed_table, &tokens)?;
    embeddings.eval();

    println!("Vocabulary size: {}", vocab_size);
    println!("Embedding dimension: {}", embed_dim);
    println!("Token indices shape: {:?}", tokens.shape());
    println!("Embeddings shape: {:?}", embeddings.shape());

    println!("\n=== Example Complete ===");
    Ok(())
}
