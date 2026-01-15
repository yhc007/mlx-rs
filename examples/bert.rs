//! BERT Model Example
//!
//! This example demonstrates the BERT (Bidirectional Encoder Representations
//! from Transformers) model implementation for embeddings and classification.
//!
//! Run with: cargo run --example bert

use mlx_rs::nn::{BertConfig, BertModel, BertWeights};
use mlx_rs::Array;

fn main() -> mlx_rs::error::Result<()> {
    println!("=== MLX-RS BERT Model Example ===\n");

    // -------------------------------------------------------------------------
    // 1. BERT Configuration
    // -------------------------------------------------------------------------
    println!("1. BERT Configuration Presets");
    println!("------------------------------");

    let base = BertConfig::bert_base_uncased();
    println!("BERT Base Uncased:");
    println!("  vocab_size: {}", base.vocab_size);
    println!("  hidden_size: {}", base.hidden_size);
    println!("  num_hidden_layers: {}", base.num_hidden_layers);
    println!("  num_attention_heads: {}", base.num_attention_heads);
    println!("  intermediate_size: {}", base.intermediate_size);
    println!("  head_dim: {}", base.head_dim());

    let large = BertConfig::bert_large_uncased();
    println!("\nBERT Large Uncased:");
    println!("  hidden_size: {}", large.hidden_size);
    println!("  num_hidden_layers: {}", large.num_hidden_layers);
    println!("  num_attention_heads: {}", large.num_attention_heads);

    // -------------------------------------------------------------------------
    // 2. Custom Configuration
    // -------------------------------------------------------------------------
    println!("\n2. Custom Configuration (for demo)");
    println!("-----------------------------------");

    // Use a smaller config for demonstration
    let config = BertConfig::new()
        .vocab_size(1000)
        .hidden_size(128)
        .num_hidden_layers(2)
        .num_attention_heads(4)
        .intermediate_size(512)
        .max_position_embeddings(128);

    println!("Demo config:");
    println!("  vocab_size: {}", config.vocab_size);
    println!("  hidden_size: {}", config.hidden_size);
    println!("  num_hidden_layers: {}", config.num_hidden_layers);
    println!("  num_attention_heads: {}", config.num_attention_heads);

    // -------------------------------------------------------------------------
    // 3. Initialize Model and Weights
    // -------------------------------------------------------------------------
    println!("\n3. Initialize Model and Weights");
    println!("--------------------------------");

    let weights = BertWeights::random(&config)?;
    let model = BertModel::new(config.clone());
    println!("Model initialized with random weights");
    println!("Number of layers: {}", weights.layers.len());

    // -------------------------------------------------------------------------
    // 4. Basic Forward Pass
    // -------------------------------------------------------------------------
    println!("\n4. Basic Forward Pass");
    println!("---------------------");

    // Simulate token IDs (in practice, these come from a tokenizer)
    // [CLS] = 101, [SEP] = 102 (BERT special tokens)
    let input_ids = Array::from_slice(&[101i32, 7, 8, 9, 10, 102], &[1, 6])?;
    println!("Input shape: {:?}", input_ids.shape());

    let (last_hidden_state, pooled_output) = model.forward(&input_ids, None, None, &weights)?;
    last_hidden_state.eval();
    pooled_output.eval();

    println!("Last hidden state shape: {:?}", last_hidden_state.shape());
    println!("Pooled output shape: {:?} ([CLS] representation)", pooled_output.shape());

    // -------------------------------------------------------------------------
    // 5. With Attention Mask (for padding)
    // -------------------------------------------------------------------------
    println!("\n5. With Attention Mask (Padding Handling)");
    println!("------------------------------------------");

    // Sequence with padding: [CLS] token token [SEP] [PAD] [PAD]
    let input_ids = Array::from_slice(&[101i32, 7, 8, 102, 0, 0], &[1, 6])?;
    let attention_mask = Array::from_slice(&[1.0f32, 1.0, 1.0, 1.0, 0.0, 0.0], &[1, 6])?;

    println!("Input IDs: [101, 7, 8, 102, 0, 0]");
    println!("Attention Mask: [1, 1, 1, 1, 0, 0] (1=real, 0=padding)");

    let (hidden, pooled) = model.forward(&input_ids, None, Some(&attention_mask), &weights)?;
    hidden.eval();
    pooled.eval();
    println!("Hidden state shape: {:?}", hidden.shape());

    // -------------------------------------------------------------------------
    // 6. With Token Type IDs (for sentence pairs)
    // -------------------------------------------------------------------------
    println!("\n6. With Token Type IDs (Sentence Pairs)");
    println!("----------------------------------------");

    // Two sentences: [CLS] sent1 [SEP] sent2 [SEP]
    let input_ids = Array::from_slice(&[101i32, 7, 8, 102, 9, 10, 102], &[1, 7])?;
    let token_type_ids = Array::from_slice(&[0i32, 0, 0, 0, 1, 1, 1], &[1, 7])?;

    println!("Input: [CLS] sent1 [SEP] sent2 [SEP]");
    println!("Token types: [0, 0, 0, 0, 1, 1, 1] (0=sent1, 1=sent2)");

    let (hidden, pooled) = model.forward(&input_ids, Some(&token_type_ids), None, &weights)?;
    hidden.eval();
    pooled.eval();
    println!("Hidden state shape: {:?}", hidden.shape());

    // -------------------------------------------------------------------------
    // 7. Batch Processing
    // -------------------------------------------------------------------------
    println!("\n7. Batch Processing");
    println!("--------------------");

    let batch_input = Array::from_slice(&[
        101i32, 7, 8, 102,    // Sentence 1
        101,    9, 10, 102,   // Sentence 2
        101,    11, 12, 102,  // Sentence 3
    ], &[3, 4])?;

    let (hidden, pooled) = model.forward(&batch_input, None, None, &weights)?;
    hidden.eval();
    pooled.eval();

    println!("Batch size: 3, Sequence length: 4");
    println!("Hidden state shape: {:?}", hidden.shape());
    println!("Pooled output shape: {:?}", pooled.shape());

    // -------------------------------------------------------------------------
    // 8. Different Pooling Methods
    // -------------------------------------------------------------------------
    println!("\n8. Different Pooling Methods");
    println!("-----------------------------");

    let input_ids = Array::from_slice(&[101i32, 7, 8, 9, 102], &[1, 5])?;

    // Method 1: [CLS] token pooling (default BERT approach)
    let cls_pooled = model.get_pooled_output(&input_ids, None, None, &weights)?;
    cls_pooled.eval();
    println!("[CLS] pooling shape: {:?}", cls_pooled.shape());

    // Method 2: Mean pooling (average all tokens)
    let mean_pooled = model.get_mean_pooled(&input_ids, None, None, &weights)?;
    mean_pooled.eval();
    println!("Mean pooling shape: {:?}", mean_pooled.shape());

    // Method 3: Get full sequence embeddings
    let encoded = model.encode(&input_ids, None, None, &weights)?;
    encoded.eval();
    println!("Full encoding shape: {:?}", encoded.shape());

    // -------------------------------------------------------------------------
    // 9. Masked Mean Pooling (for sentence embeddings)
    // -------------------------------------------------------------------------
    println!("\n9. Masked Mean Pooling");
    println!("-----------------------");

    let input_ids = Array::from_slice(&[101i32, 7, 8, 102, 0, 0], &[1, 6])?;
    let attention_mask = Array::from_slice(&[1.0f32, 1.0, 1.0, 1.0, 0.0, 0.0], &[1, 6])?;

    // Mean pooling that ignores padding tokens
    let mean_pooled = model.get_mean_pooled(&input_ids, None, Some(&attention_mask), &weights)?;
    mean_pooled.eval();
    println!("Masked mean pooling shape: {:?}", mean_pooled.shape());
    println!("(Only averages non-padding tokens)");

    // -------------------------------------------------------------------------
    // 10. Example Use Cases
    // -------------------------------------------------------------------------
    println!("\n10. Example Use Cases");
    println!("---------------------");

    println!("\nBERT can be used for:");
    println!("  - Text Classification: Use pooled output + classifier head");
    println!("  - Named Entity Recognition: Use per-token hidden states");
    println!("  - Question Answering: Use hidden states for start/end prediction");
    println!("  - Sentence Embeddings: Use mean pooling for similarity tasks");
    println!("  - Feature Extraction: Use hidden states as features");

    println!("\nTypical training pipeline:");
    println!("```rust");
    println!("// 1. Load pretrained weights");
    println!("let weights = serialize::load_safetensors(\"bert-base.safetensors\")?;");
    println!("");
    println!("// 2. Create model");
    println!("let config = BertConfig::bert_base_uncased();");
    println!("let model = BertModel::new(config);");
    println!("");
    println!("// 3. Get embeddings");
    println!("let (_, pooled) = model.forward(&input_ids, None, None, &weights)?;");
    println!("");
    println!("// 4. Add classification head");
    println!("let logits = pooled.matmul(&classifier_weight)? + &classifier_bias;");
    println!("```");

    println!("\n=== Example Complete ===");
    Ok(())
}
