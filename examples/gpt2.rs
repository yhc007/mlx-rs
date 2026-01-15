//! GPT-2 Model Example
//!
//! This example demonstrates the GPT-2 language model implementation
//! for text generation tasks.
//!
//! Run with: cargo run --example gpt2

use mlx_rs::nn::{GPT2Config, GPT2Model, GPT2Weights};
use mlx_rs::Array;

fn main() -> mlx_rs::error::Result<()> {
    println!("=== MLX-RS GPT-2 Model Example ===\n");

    // -------------------------------------------------------------------------
    // 1. GPT-2 Configuration Presets
    // -------------------------------------------------------------------------
    println!("1. GPT-2 Configuration Presets");
    println!("-------------------------------");

    let small = GPT2Config::gpt2_small();
    println!("GPT-2 Small (~117M params):");
    println!("  vocab_size: {}", small.vocab_size);
    println!("  n_positions: {}", small.n_positions);
    println!("  n_embd: {}", small.n_embd);
    println!("  n_layer: {}", small.n_layer);
    println!("  n_head: {}", small.n_head);

    let medium = GPT2Config::gpt2_medium();
    println!("\nGPT-2 Medium (~345M params):");
    println!("  n_embd: {}", medium.n_embd);
    println!("  n_layer: {}", medium.n_layer);
    println!("  n_head: {}", medium.n_head);

    let large = GPT2Config::gpt2_large();
    println!("\nGPT-2 Large (~774M params):");
    println!("  n_embd: {}", large.n_embd);
    println!("  n_layer: {}", large.n_layer);
    println!("  n_head: {}", large.n_head);

    let xl = GPT2Config::gpt2_xl();
    println!("\nGPT-2 XL (~1.5B params):");
    println!("  n_embd: {}", xl.n_embd);
    println!("  n_layer: {}", xl.n_layer);
    println!("  n_head: {}", xl.n_head);

    // -------------------------------------------------------------------------
    // 2. Custom Configuration
    // -------------------------------------------------------------------------
    println!("\n2. Custom Configuration (for demo)");
    println!("-----------------------------------");

    // Use a smaller config for demonstration
    let config = GPT2Config::new()
        .vocab_size(1000)
        .n_positions(64)
        .n_embd(64)
        .n_layer(2)
        .n_head(4);

    println!("Demo config:");
    println!("  vocab_size: {}", config.vocab_size);
    println!("  n_positions: {}", config.n_positions);
    println!("  n_embd: {}", config.n_embd);
    println!("  n_layer: {}", config.n_layer);
    println!("  n_head: {}", config.n_head);
    println!("  head_dim: {}", config.head_dim());

    // -------------------------------------------------------------------------
    // 3. Initialize Model and Weights
    // -------------------------------------------------------------------------
    println!("\n3. Initialize Model and Weights");
    println!("--------------------------------");

    let weights = GPT2Weights::random(&config)?;
    let model = GPT2Model::new(config.clone());
    println!("Model initialized with random weights");
    println!("Number of transformer layers: {}", weights.blocks.len());

    // -------------------------------------------------------------------------
    // 4. Forward Pass
    // -------------------------------------------------------------------------
    println!("\n4. Forward Pass");
    println!("----------------");

    // Create dummy input tokens: (batch, seq_len)
    let tokens = Array::from_slice(&[1i32, 2, 3, 4, 5], &[1, 5])?;
    println!("Input tokens shape: {:?} (batch, seq_len)", tokens.shape());

    let logits = model.forward(&tokens, &weights)?;
    logits.eval();

    println!("Output logits shape: {:?} (batch, seq_len, vocab_size)", logits.shape());

    // -------------------------------------------------------------------------
    // 5. Text Generation
    // -------------------------------------------------------------------------
    println!("\n5. Text Generation");
    println!("-------------------");

    let prompt_tokens = Array::from_slice(&[1i32, 2, 3], &[1, 3])?;
    println!("Prompt tokens shape: {:?}", prompt_tokens.shape());

    let output_tokens = model.generate(&prompt_tokens, &weights, 5, 1.0)?;
    output_tokens.eval();

    println!("Generated tokens shape: {:?}", output_tokens.shape());
    println!("  - Started with 3 tokens, generated 5 more = 8 total");

    // Show the generated token values
    let token_values = output_tokens.to_vec::<i32>()?;
    println!("Token values: {:?}", token_values);

    // -------------------------------------------------------------------------
    // 6. Temperature Sampling
    // -------------------------------------------------------------------------
    println!("\n6. Temperature Sampling");
    println!("------------------------");

    println!("Temperature controls randomness in generation:");
    println!("  - temperature = 0.1: More deterministic (sharper distribution)");
    println!("  - temperature = 1.0: Standard sampling");
    println!("  - temperature = 2.0: More random (flatter distribution)");

    // Generate with low temperature (more deterministic)
    let low_temp = model.generate(&prompt_tokens, &weights, 5, 0.5)?;
    low_temp.eval();
    println!("\nLow temp (0.5) output: {:?}", low_temp.to_vec::<i32>()?);

    // Generate with high temperature (more random)
    let high_temp = model.generate(&prompt_tokens, &weights, 5, 2.0)?;
    high_temp.eval();
    println!("High temp (2.0) output: {:?}", high_temp.to_vec::<i32>()?);

    // -------------------------------------------------------------------------
    // 7. Batch Processing
    // -------------------------------------------------------------------------
    println!("\n7. Batch Processing");
    println!("--------------------");

    // Batch of 2 sequences
    let batch_tokens = Array::from_slice(&[
        1i32, 2, 3, 4,
        5, 6, 7, 8,
    ], &[2, 4])?;

    let batch_logits = model.forward(&batch_tokens, &weights)?;
    batch_logits.eval();

    println!("Batch tokens shape: {:?}", batch_tokens.shape());
    println!("Batch logits shape: {:?}", batch_logits.shape());

    // -------------------------------------------------------------------------
    // 8. GPT-2 Architecture Overview
    // -------------------------------------------------------------------------
    println!("\n8. GPT-2 Architecture Overview");
    println!("-------------------------------");

    println!("
    Input Tokens [batch, seq_len]
              |
              v
    +--------------------+
    | Token Embedding    |  vocab_size x n_embd
    +--------------------+
              |
              v
    +--------------------+
    | + Position Embed   |  n_positions x n_embd
    +--------------------+
              |
              v
    +--------------------+
    | Dropout            |
    +--------------------+
              |
              v
    +--------------------+
    | LayerNorm (pre)    |  <-- Pre-norm architecture
    +--------------------+
              |
              v
    +--------------------+
    | Multi-Head Attn    |  Combined QKV projection
    | (Causal Mask)      |  n_head x head_dim
    +--------------------+
              |
              v
    +--------------------+
    | Residual Add       |
    +--------------------+
              |
              v
    +--------------------+
    | LayerNorm (pre)    |
    +--------------------+
              |
              v
    +--------------------+
    | MLP (GELU)         |  n_embd -> 4*n_embd -> n_embd
    +--------------------+
              |
              v
    +--------------------+
    | Residual Add       |
    +--------------------+
              |
    (Repeat N transformer layers)
              |
              v
    +--------------------+
    | Final LayerNorm    |
    +--------------------+
              |
              v
    +--------------------+
    | Output Linear      |  Weight-tied with token embedding
    +--------------------+
              |
              v
    Logits [batch, seq_len, vocab_size]
    ");

    // -------------------------------------------------------------------------
    // 9. Key GPT-2 Features
    // -------------------------------------------------------------------------
    println!("9. Key GPT-2 Features");
    println!("---------------------");

    println!("\nGPT-2 architecture highlights:");
    println!("  - Pre-norm: LayerNorm before attention/MLP (more stable training)");
    println!("  - Combined QKV: Single projection for queries, keys, values");
    println!("  - GELU activation: Smoother than ReLU in MLP");
    println!("  - Weight tying: Output projection shares weights with embedding");
    println!("  - Causal masking: Each position only attends to previous positions");

    println!("\nComparison with other architectures:");
    println!("  - GPT-2 vs BERT: Decoder-only vs Encoder-only");
    println!("  - GPT-2 vs Llama: Pre-norm (both), but Llama uses RMSNorm + RoPE");
    println!("  - GPT-2 vs GPT-3: Same architecture, different scale");

    // -------------------------------------------------------------------------
    // 10. Example Use Cases
    // -------------------------------------------------------------------------
    println!("\n10. Example Use Cases");
    println!("---------------------");

    println!("\nGPT-2 can be used for:");
    println!("  - Text generation / completion");
    println!("  - Story writing");
    println!("  - Code generation");
    println!("  - Dialogue systems");
    println!("  - Summarization (with fine-tuning)");

    println!("\nTypical inference pipeline:");
    println!("```rust");
    println!("// 1. Load tokenizer and pretrained weights");
    println!("let tokenizer = load_tokenizer(\"gpt2\")?;");
    println!("let weights = serialize::load_safetensors(\"gpt2.safetensors\")?;");
    println!("");
    println!("// 2. Create model");
    println!("let config = GPT2Config::gpt2_small();");
    println!("let model = GPT2Model::new(config);");
    println!("");
    println!("// 3. Encode prompt");
    println!("let prompt = \"Once upon a time\";");
    println!("let tokens = tokenizer.encode(prompt)?;");
    println!("");
    println!("// 4. Generate");
    println!("let output = model.generate(&tokens, &weights, 50, 0.8)?;");
    println!("");
    println!("// 5. Decode output");
    println!("let text = tokenizer.decode(&output)?;");
    println!("println!(\"Generated: {{}}\", text);");
    println!("```");

    println!("\n=== Example Complete ===");
    Ok(())
}
