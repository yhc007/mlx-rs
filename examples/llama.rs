//! Llama Model Architecture Example
//!
//! This example demonstrates the Llama model components in mlx-rs.
//!
//! Run with: cargo run --example llama

use mlx_rs::{Array, nn, random};
use mlx_rs::nn::{LlamaConfig, LlamaModel, swiglu, llama_feedforward};

fn main() -> mlx_rs::error::Result<()> {
    println!("=== MLX-RS Llama Model Example ===\n");

    // -------------------------------------------------------------------------
    // Llama Configuration
    // -------------------------------------------------------------------------
    println!("1. Llama Configurations");
    println!("-----------------------");

    // Default configuration
    let default_config = LlamaConfig::default();
    println!("Default Llama config:");
    println!("  vocab_size: {}", default_config.vocab_size);
    println!("  hidden_size: {}", default_config.hidden_size);
    println!("  num_attention_heads: {}", default_config.num_attention_heads);
    println!("  num_hidden_layers: {}", default_config.num_hidden_layers);

    // Llama 2 7B
    let llama2_config = LlamaConfig::llama2_7b();
    println!("\nLlama 2 7B config:");
    println!("  vocab_size: {}", llama2_config.vocab_size);
    println!("  hidden_size: {}", llama2_config.hidden_size);
    println!("  intermediate_size: {}", llama2_config.intermediate_size);
    println!("  num_attention_heads: {}", llama2_config.num_attention_heads);
    println!("  num_key_value_heads: {}", llama2_config.num_key_value_heads);
    println!("  head_dim: {}", llama2_config.head_dim());

    // Llama 3 8B (with GQA)
    let llama3_config = LlamaConfig::llama3_8b();
    println!("\nLlama 3 8B config (with GQA):");
    println!("  vocab_size: {}", llama3_config.vocab_size);
    println!("  num_attention_heads: {}", llama3_config.num_attention_heads);
    println!("  num_key_value_heads: {} (GQA)", llama3_config.num_key_value_heads);
    println!("  rope_theta: {}", llama3_config.rope_theta);

    // Custom configuration using builder pattern
    let custom_config = LlamaConfig::new()
        .vocab_size(10000)
        .hidden_size(256)
        .num_hidden_layers(4)
        .num_attention_heads(8);
    println!("\nCustom mini config:");
    println!("  vocab_size: {}", custom_config.vocab_size);
    println!("  hidden_size: {}", custom_config.hidden_size);
    println!("  num_hidden_layers: {}", custom_config.num_hidden_layers);

    // -------------------------------------------------------------------------
    // SwiGLU Activation
    // -------------------------------------------------------------------------
    println!("\n2. SwiGLU Activation");
    println!("--------------------");

    // SwiGLU: swish(x @ W_gate) * (x @ W_up)
    // where swish(x) = x * sigmoid(x)

    let batch = 2;
    let seq_len = 4;
    let hidden = 8;
    let intermediate = 16;

    let x = random::normal_with_params::<f32>(&[batch, seq_len, hidden], 0.0, 1.0, None)?;
    let gate_proj = random::normal_with_params::<f32>(&[hidden, intermediate], 0.0, 0.1, None)?;
    let up_proj = random::normal_with_params::<f32>(&[hidden, intermediate], 0.0, 0.1, None)?;

    let swiglu_out = swiglu(&x, &gate_proj, &up_proj)?;
    swiglu_out.eval();

    println!("Input shape: {:?}", x.shape());
    println!("Gate projection: {:?}", gate_proj.shape());
    println!("Up projection: {:?}", up_proj.shape());
    println!("SwiGLU output shape: {:?}", swiglu_out.shape());

    // -------------------------------------------------------------------------
    // Llama FeedForward (MLP)
    // -------------------------------------------------------------------------
    println!("\n3. Llama FeedForward Network");
    println!("----------------------------");

    // FFN(x) = down_proj(swiglu(x, gate_proj, up_proj))

    let down_proj = random::normal_with_params::<f32>(&[intermediate, hidden], 0.0, 0.1, None)?;

    let ffn_out = llama_feedforward(&x, &gate_proj, &up_proj, &down_proj)?;
    ffn_out.eval();

    println!("Input shape: {:?}", x.shape());
    println!("FFN output shape: {:?}", ffn_out.shape());
    println!("(Input and output have same shape)");

    // -------------------------------------------------------------------------
    // RoPE (Rotary Position Embedding)
    // -------------------------------------------------------------------------
    println!("\n4. Rotary Position Embedding (RoPE)");
    println!("-----------------------------------");

    let max_len = 32;
    let head_dim = 64;
    let rope_theta = 10000.0;

    let (cos, sin) = nn::precompute_rope_frequencies(max_len, head_dim, rope_theta)?;
    cos.eval();
    sin.eval();

    println!("Precomputed RoPE frequencies:");
    println!("  max_seq_len: {}", max_len);
    println!("  head_dim: {}", head_dim);
    println!("  cos shape: {:?}", cos.shape());
    println!("  sin shape: {:?}", sin.shape());

    // Apply RoPE to queries/keys
    let q = random::normal_with_params::<f32>(&[1, 4, max_len, head_dim], 0.0, 0.1, None)?;
    let q_rotated = nn::apply_rotary_embedding(&q, &cos, &sin)?;
    q_rotated.eval();

    println!("  Query shape: {:?}", q.shape());
    println!("  Rotated query shape: {:?}", q_rotated.shape());

    // -------------------------------------------------------------------------
    // RMSNorm
    // -------------------------------------------------------------------------
    println!("\n5. RMSNorm (Pre-LayerNorm)");
    println!("--------------------------");

    let x = Array::from_slice(&[1.0f32, 2.0, 3.0, 4.0, 5.0, 6.0], &[2, 3])?;
    let weight = Array::ones::<f32>(&[3])?;

    let normed = nn::rms_norm(&x, &weight, 1e-6)?;
    normed.eval();

    println!("Input: {:?}", x.to_vec::<f32>()?);
    println!("RMSNorm output: {:?}", normed.to_vec::<f32>()?);

    // -------------------------------------------------------------------------
    // Causal Attention Mask
    // -------------------------------------------------------------------------
    println!("\n6. Causal Attention Mask");
    println!("------------------------");

    let seq_len = 4;
    let mask = nn::create_causal_mask(seq_len)?;
    mask.eval();

    println!("Causal mask for seq_len={}:", seq_len);
    let mask_vals = mask.to_vec::<f32>()?;
    for i in 0..seq_len as usize {
        let row: Vec<String> = mask_vals[i*seq_len as usize..(i+1)*seq_len as usize]
            .iter()
            .map(|&v| if v.is_infinite() { "-inf".to_string() } else { format!("{:.1}", v) })
            .collect();
        println!("  {:?}", row);
    }

    // -------------------------------------------------------------------------
    // Mini Llama Model (Demo)
    // -------------------------------------------------------------------------
    println!("\n7. Mini Llama Model Forward Pass");
    println!("--------------------------------");

    // Create a tiny model for demonstration
    let mini_config = LlamaConfig::new()
        .vocab_size(100)
        .hidden_size(32)
        .num_hidden_layers(2)
        .num_attention_heads(4);

    println!("Mini Llama config:");
    println!("  vocab_size: {}", mini_config.vocab_size);
    println!("  hidden_size: {}", mini_config.hidden_size);
    println!("  num_layers: {}", mini_config.num_hidden_layers);
    println!("  num_heads: {}", mini_config.num_attention_heads);
    println!("  head_dim: {}", mini_config.head_dim());

    // Note: In a real application, you would load pre-trained weights
    // Here we just demonstrate the architecture
    println!("\n(Note: Use LlamaWeights::random() for testing,");
    println!(" or load pre-trained weights for real inference)");

    // -------------------------------------------------------------------------
    // Grouped Query Attention (GQA)
    // -------------------------------------------------------------------------
    println!("\n8. Grouped Query Attention (GQA)");
    println!("--------------------------------");

    println!("GQA reduces memory by using fewer K/V heads than Q heads:");
    println!("");
    println!("Standard MHA (Llama 1/2):");
    println!("  num_attention_heads = 32");
    println!("  num_key_value_heads = 32");
    println!("  K/V memory: 32 heads * seq_len * head_dim");
    println!("");
    println!("GQA (Llama 3):");
    println!("  num_attention_heads = 32");
    println!("  num_key_value_heads = 8");
    println!("  K/V memory: 8 heads * seq_len * head_dim (4x reduction!)");
    println!("");
    println!("Each group of 4 Q heads shares 1 K/V head pair.");

    // -------------------------------------------------------------------------
    // Architecture Summary
    // -------------------------------------------------------------------------
    println!("\n9. Llama Architecture Summary");
    println!("-----------------------------");
    println!("
    Input IDs
        |
        v
    [Embedding] ──────────────────────────────────┐
        |                                          │
        v                                          │
    ┌───────────────────────────────┐              │
    │  Transformer Block (x N)      │              │
    │  ├─ RMSNorm                   │              │
    │  ├─ Self-Attention + RoPE    │              │
    │  ├─ Residual Connection       │              │
    │  ├─ RMSNorm                   │              │
    │  ├─ FFN (SwiGLU)              │              │
    │  └─ Residual Connection       │              │
    └───────────────────────────────┘              │
        |                                          │
        v                                          │
    [RMSNorm]                                      │
        |                                          │
        v                                          │
    [LM Head] ←──── (optional: tied weights) ─────┘
        |
        v
    Logits (vocab_size)
    ");

    println!("\n=== Example Complete ===");
    Ok(())
}
