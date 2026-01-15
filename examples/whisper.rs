//! Whisper Model Example
//!
//! This example demonstrates the Whisper speech recognition model
//! implementation for automatic speech recognition (ASR) tasks.
//!
//! Run with: cargo run --example whisper

use mlx_rs::nn::{WhisperConfig, WhisperModel, WhisperWeights};
use mlx_rs::Array;

fn main() -> mlx_rs::error::Result<()> {
    println!("=== MLX-RS Whisper Model Example ===\n");

    // -------------------------------------------------------------------------
    // 1. Whisper Configuration Presets
    // -------------------------------------------------------------------------
    println!("1. Whisper Configuration Presets");
    println!("---------------------------------");

    let tiny = WhisperConfig::whisper_tiny();
    println!("Whisper Tiny (~39M params):");
    println!("  n_mels: {}", tiny.n_mels);
    println!("  n_audio_state: {}", tiny.n_audio_state);
    println!("  n_audio_layer: {}", tiny.n_audio_layer);
    println!("  n_text_state: {}", tiny.n_text_state);
    println!("  n_text_layer: {}", tiny.n_text_layer);

    let base = WhisperConfig::whisper_base();
    println!("\nWhisper Base (~74M params):");
    println!("  n_audio_state: {}", base.n_audio_state);
    println!("  n_audio_layer: {}", base.n_audio_layer);

    let small = WhisperConfig::whisper_small();
    println!("\nWhisper Small (~244M params):");
    println!("  n_audio_state: {}", small.n_audio_state);
    println!("  n_audio_layer: {}", small.n_audio_layer);

    let medium = WhisperConfig::whisper_medium();
    println!("\nWhisper Medium (~769M params):");
    println!("  n_audio_state: {}", medium.n_audio_state);
    println!("  n_audio_layer: {}", medium.n_audio_layer);

    let large = WhisperConfig::whisper_large();
    println!("\nWhisper Large (~1550M params):");
    println!("  n_audio_state: {}", large.n_audio_state);
    println!("  n_audio_layer: {}", large.n_audio_layer);

    let large_v3 = WhisperConfig::whisper_large_v3();
    println!("\nWhisper Large-v3:");
    println!("  n_mels: {} (increased from 80)", large_v3.n_mels);

    // -------------------------------------------------------------------------
    // 2. Custom Configuration
    // -------------------------------------------------------------------------
    println!("\n2. Custom Configuration (for demo)");
    println!("-----------------------------------");

    // Use a smaller config for demonstration
    let config = WhisperConfig::new()
        .n_mels(40)
        .n_audio_ctx(100)
        .n_audio_state(64)
        .n_audio_head(4)
        .n_audio_layer(2)
        .n_vocab(1000)
        .n_text_ctx(50)
        .n_text_state(64)
        .n_text_head(4)
        .n_text_layer(2);

    println!("Demo config:");
    println!("  n_mels: {}", config.n_mels);
    println!("  n_audio_ctx: {}", config.n_audio_ctx);
    println!("  n_audio_state: {}", config.n_audio_state);
    println!("  n_audio_layer: {}", config.n_audio_layer);
    println!("  n_vocab: {}", config.n_vocab);
    println!("  n_text_ctx: {}", config.n_text_ctx);
    println!("  audio_head_dim: {}", config.audio_head_dim());
    println!("  text_head_dim: {}", config.text_head_dim());

    // -------------------------------------------------------------------------
    // 3. Initialize Model and Weights
    // -------------------------------------------------------------------------
    println!("\n3. Initialize Model and Weights");
    println!("--------------------------------");

    let weights = WhisperWeights::random(&config)?;
    let model = WhisperModel::new(config.clone());
    println!("Model initialized with random weights");
    println!("Number of encoder layers: {}", weights.encoder_layers.len());
    println!("Number of decoder layers: {}", weights.decoder_layers.len());

    // -------------------------------------------------------------------------
    // 4. Audio Encoding
    // -------------------------------------------------------------------------
    println!("\n4. Audio Encoding");
    println!("-----------------");

    // Create dummy mel spectrogram: (batch, n_mels, n_frames)
    // In practice, this would come from audio preprocessing
    let mel = Array::zeros::<f32>(&[1, 40, 100])?;
    println!("Input mel spectrogram shape: {:?} (batch, n_mels, n_frames)", mel.shape());

    let audio_features = model.encode(&mel, &weights)?;
    audio_features.eval();

    println!("Audio features shape: {:?}", audio_features.shape());
    println!("  - Sequence length reduced by conv stride");

    // -------------------------------------------------------------------------
    // 5. Text Decoding
    // -------------------------------------------------------------------------
    println!("\n5. Text Decoding");
    println!("----------------");

    // Whisper special tokens (example values):
    // <|startoftranscript|> = 50258
    // <|en|> = 50259 (English)
    // <|transcribe|> = 50359
    // <|notimestamps|> = 50363
    let tokens = Array::from_slice(&[0i32, 1, 2, 3, 4], &[1, 5])?;
    println!("Input tokens shape: {:?}", tokens.shape());

    let logits = model.decode(&tokens, &audio_features, &weights)?;
    logits.eval();

    println!("Output logits shape: {:?} (batch, seq_len, vocab_size)", logits.shape());

    // -------------------------------------------------------------------------
    // 6. Full Forward Pass
    // -------------------------------------------------------------------------
    println!("\n6. Full Forward Pass");
    println!("--------------------");

    let mel = Array::zeros::<f32>(&[1, 40, 100])?;
    let tokens = Array::from_slice(&[0i32, 1, 2], &[1, 3])?;

    let logits = model.forward(&mel, &tokens, &weights)?;
    logits.eval();

    println!("Mel input shape: {:?}", mel.shape());
    println!("Token input shape: {:?}", tokens.shape());
    println!("Output logits shape: {:?}", logits.shape());

    // -------------------------------------------------------------------------
    // 7. Batch Processing
    // -------------------------------------------------------------------------
    println!("\n7. Batch Processing");
    println!("--------------------");

    // Batch of 2 audio samples
    let mel_batch = Array::zeros::<f32>(&[2, 40, 100])?;
    let tokens_batch = Array::from_slice(&[
        0i32, 1, 2, 3,
        4, 5, 6, 7,
    ], &[2, 4])?;

    let logits_batch = model.forward(&mel_batch, &tokens_batch, &weights)?;
    logits_batch.eval();

    println!("Batch mel shape: {:?}", mel_batch.shape());
    println!("Batch tokens shape: {:?}", tokens_batch.shape());
    println!("Batch logits shape: {:?}", logits_batch.shape());

    // -------------------------------------------------------------------------
    // 8. Whisper Architecture Overview
    // -------------------------------------------------------------------------
    println!("\n8. Whisper Architecture Overview");
    println!("---------------------------------");

    println!("
    Audio Input (Mel Spectrogram)
    [batch, n_mels, n_frames]
              |
              v
    +--------------------+
    | Conv1d (stride=1)  |  -> GELU
    +--------------------+
              |
              v
    +--------------------+
    | Conv1d (stride=2)  |  -> GELU (downsamples by 2x)
    +--------------------+
              |
              v
    +--------------------+
    | + Position Embed   |  Sinusoidal embeddings
    +--------------------+
              |
              v
    +--------------------+
    | Encoder Blocks     |  Self-attention + MLP
    | (N layers)         |
    +--------------------+
              |
              v
    Audio Features [batch, seq/2, d_model]
              |
              +---------------------------+
                                          |
    Text Tokens                           |
    [batch, seq]                          |
         |                                |
         v                                |
    +--------------------+                |
    | Token Embedding    |                |
    +--------------------+                |
         |                                |
         v                                |
    +--------------------+                |
    | + Position Embed   |                |
    +--------------------+                |
         |                                |
         v                                v
    +--------------------+     +--------------------+
    | Decoder Self-Attn  | <-- | Cross-Attention    |
    | (Causal Mask)      |     | (Audio Features)   |
    +--------------------+     +--------------------+
         |
         v
    +--------------------+
    | MLP                |
    +--------------------+
         |
         v
    (Repeat N decoder layers)
         |
         v
    +--------------------+
    | Linear (to vocab)  |
    +--------------------+
         |
         v
    Logits [batch, seq, vocab_size]
    ");

    // -------------------------------------------------------------------------
    // 9. Example Use Cases
    // -------------------------------------------------------------------------
    println!("9. Example Use Cases");
    println!("--------------------");

    println!("\nWhisper can be used for:");
    println!("  - Speech-to-Text transcription");
    println!("  - Multilingual speech recognition (99+ languages)");
    println!("  - Speech translation (to English)");
    println!("  - Language identification");
    println!("  - Voice activity detection");

    println!("\nTypical inference pipeline:");
    println!("```rust");
    println!("// 1. Load audio and compute mel spectrogram");
    println!("let audio = load_audio(\"speech.wav\")?;");
    println!("let mel = compute_mel_spectrogram(&audio)?;");
    println!("");
    println!("// 2. Load pretrained weights");
    println!("let weights = serialize::load_safetensors(\"whisper-base.safetensors\")?;");
    println!("");
    println!("// 3. Create model");
    println!("let config = WhisperConfig::whisper_base();");
    println!("let model = WhisperModel::new(config);");
    println!("");
    println!("// 4. Encode audio");
    println!("let audio_features = model.encode(&mel, &weights)?;");
    println!("");
    println!("// 5. Autoregressive decoding");
    println!("let mut tokens = vec![START_TOKEN];");
    println!("for _ in 0..max_tokens {{");
    println!("    let logits = model.decode(&tokens, &audio_features, &weights)?;");
    println!("    let next_token = logits.argmax(-1)?;");
    println!("    tokens.push(next_token);");
    println!("    if next_token == END_TOKEN {{ break; }}");
    println!("}}");
    println!("```");

    println!("\n=== Example Complete ===");
    Ok(())
}
