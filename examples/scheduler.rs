//! Learning Rate Scheduler Example
//!
//! This example demonstrates various learning rate schedulers for training neural networks.
//!
//! Run with: cargo run --example scheduler

use mlx_rs::scheduler::{
    LRScheduler, StepLR, MultiStepLR, ExponentialLR, LinearLR,
    CosineAnnealingLR, CosineAnnealingWarmRestarts, WarmupCosine,
    WarmupLinear, OneCycleLR, PolynomialLR, ConstantLR,
};

fn main() {
    println!("=== MLX-RS Learning Rate Scheduler Example ===\n");

    // -------------------------------------------------------------------------
    // StepLR - Step decay every N steps
    // -------------------------------------------------------------------------
    println!("1. StepLR - Decay by gamma every step_size steps");
    println!("------------------------------------------------");

    let mut scheduler = StepLR::new(0.1, 30, 0.1);
    println!("Config: initial_lr=0.1, step_size=30, gamma=0.1");

    for step in [0, 15, 29, 30, 59, 60, 89, 90] {
        scheduler.reset();
        for _ in 0..step {
            scheduler.step();
        }
        println!("  Step {:3}: lr = {:.6}", step, scheduler.get_lr());
    }

    // -------------------------------------------------------------------------
    // MultiStepLR - Decay at specific milestones
    // -------------------------------------------------------------------------
    println!("\n2. MultiStepLR - Decay at specific milestones");
    println!("----------------------------------------------");

    let mut scheduler = MultiStepLR::new(0.1, vec![30, 60, 90], 0.1);
    println!("Config: initial_lr=0.1, milestones=[30, 60, 90], gamma=0.1");

    for step in [0, 29, 30, 59, 60, 89, 90, 100] {
        scheduler.reset();
        for _ in 0..step {
            scheduler.step();
        }
        println!("  Step {:3}: lr = {:.6}", step, scheduler.get_lr());
    }

    // -------------------------------------------------------------------------
    // ExponentialLR - Decay every step
    // -------------------------------------------------------------------------
    println!("\n3. ExponentialLR - Exponential decay every step");
    println!("------------------------------------------------");

    let mut scheduler = ExponentialLR::new(0.1, 0.99);
    println!("Config: initial_lr=0.1, gamma=0.99");

    for step in [0, 10, 50, 100, 200] {
        scheduler.reset();
        for _ in 0..step {
            scheduler.step();
        }
        println!("  Step {:3}: lr = {:.6}", step, scheduler.get_lr());
    }

    // -------------------------------------------------------------------------
    // LinearLR - Linear decay from initial to final
    // -------------------------------------------------------------------------
    println!("\n4. LinearLR - Linear interpolation");
    println!("-----------------------------------");

    let mut scheduler = LinearLR::new(0.1, 0.001, 100);
    println!("Config: initial_lr=0.1, final_lr=0.001, total_steps=100");

    for step in [0, 25, 50, 75, 100] {
        scheduler.reset();
        for _ in 0..step {
            scheduler.step();
        }
        println!("  Step {:3}: lr = {:.6}", step, scheduler.get_lr());
    }

    // -------------------------------------------------------------------------
    // CosineAnnealingLR - Cosine decay
    // -------------------------------------------------------------------------
    println!("\n5. CosineAnnealingLR - Cosine decay");
    println!("------------------------------------");

    let mut scheduler = CosineAnnealingLR::new(0.1, 100).min_lr(0.001);
    println!("Config: max_lr=0.1, min_lr=0.001, total_steps=100");

    for step in [0, 25, 50, 75, 100] {
        scheduler.reset();
        for _ in 0..step {
            scheduler.step();
        }
        println!("  Step {:3}: lr = {:.6}", step, scheduler.get_lr());
    }

    // -------------------------------------------------------------------------
    // CosineAnnealingWarmRestarts - SGDR
    // -------------------------------------------------------------------------
    println!("\n6. CosineAnnealingWarmRestarts - SGDR with warm restarts");
    println!("---------------------------------------------------------");

    let mut scheduler = CosineAnnealingWarmRestarts::new(0.1, 50)
        .t_mult(2.0)
        .min_lr(0.001);
    println!("Config: max_lr=0.1, min_lr=0.001, t_0=50, t_mult=2.0");

    for step in [0, 25, 49, 50, 100, 149, 150] {
        scheduler.reset();
        for _ in 0..step {
            scheduler.step();
        }
        println!("  Step {:3}: lr = {:.6}", step, scheduler.get_lr());
    }

    // -------------------------------------------------------------------------
    // WarmupCosine - Most common for transformers
    // -------------------------------------------------------------------------
    println!("\n7. WarmupCosine - Linear warmup + cosine decay (transformer training)");
    println!("----------------------------------------------------------------------");

    let mut scheduler = WarmupCosine::new(0.001, 100, 1000).min_lr(1e-6);
    println!("Config: max_lr=0.001, warmup_steps=100, total_steps=1000, min_lr=1e-6");

    for step in [0, 50, 100, 250, 500, 750, 1000] {
        scheduler.reset();
        for _ in 0..step {
            scheduler.step();
        }
        println!("  Step {:4}: lr = {:.8}", step, scheduler.get_lr());
    }

    // -------------------------------------------------------------------------
    // WarmupLinear - Linear warmup + linear decay
    // -------------------------------------------------------------------------
    println!("\n8. WarmupLinear - Linear warmup + linear decay");
    println!("-----------------------------------------------");

    let mut scheduler = WarmupLinear::new(0.001, 100, 1000).min_lr(1e-6);
    println!("Config: max_lr=0.001, warmup_steps=100, total_steps=1000, min_lr=1e-6");

    for step in [0, 50, 100, 250, 500, 750, 1000] {
        scheduler.reset();
        for _ in 0..step {
            scheduler.step();
        }
        println!("  Step {:4}: lr = {:.8}", step, scheduler.get_lr());
    }

    // -------------------------------------------------------------------------
    // OneCycleLR - Super-convergence training
    // -------------------------------------------------------------------------
    println!("\n9. OneCycleLR - Super-convergence (one cycle policy)");
    println!("-----------------------------------------------------");

    let mut scheduler = OneCycleLR::new(0.1, 1000)
        .pct_start(0.3)
        .div_factor(25.0)
        .final_div_factor(1e4);
    println!("Config: max_lr=0.1, total_steps=1000, pct_start=0.3");
    println!("       div_factor=25.0 (initial_lr = max_lr/25 = 0.004)");
    println!("       final_div_factor=1e4 (final_lr = initial_lr/1e4 = 4e-7)");

    for step in [0, 150, 300, 500, 750, 1000] {
        scheduler.reset();
        for _ in 0..step {
            scheduler.step();
        }
        println!("  Step {:4}: lr = {:.8}", step, scheduler.get_lr());
    }

    // -------------------------------------------------------------------------
    // PolynomialLR - Polynomial decay
    // -------------------------------------------------------------------------
    println!("\n10. PolynomialLR - Polynomial decay");
    println!("------------------------------------");

    let mut scheduler = PolynomialLR::new(0.1, 100)
        .final_lr(0.001)
        .power(2.0);
    println!("Config: initial_lr=0.1, final_lr=0.001, total_steps=100, power=2.0");

    for step in [0, 25, 50, 75, 100] {
        scheduler.reset();
        for _ in 0..step {
            scheduler.step();
        }
        println!("  Step {:3}: lr = {:.6}", step, scheduler.get_lr());
    }

    // -------------------------------------------------------------------------
    // ConstantLR - For completeness
    // -------------------------------------------------------------------------
    println!("\n11. ConstantLR - Constant learning rate");
    println!("----------------------------------------");

    let mut scheduler = ConstantLR::new(0.01);
    println!("Config: lr=0.01");

    for step in [0, 50, 100, 500] {
        scheduler.reset();
        for _ in 0..step {
            scheduler.step();
        }
        println!("  Step {:3}: lr = {:.6}", step, scheduler.get_lr());
    }

    // -------------------------------------------------------------------------
    // Typical training loop example
    // -------------------------------------------------------------------------
    println!("\n12. Example Training Loop Pattern");
    println!("----------------------------------");
    println!("```rust");
    println!("use mlx_rs::scheduler::{{WarmupCosine, LRScheduler}};");
    println!("use mlx_rs::nn::{{Adam, Optimizer}};");
    println!();
    println!("let mut optimizer = Adam::new(0.001);");
    println!("let mut scheduler = WarmupCosine::new(0.001, 1000, 10000);");
    println!();
    println!("for epoch in 0..num_epochs {{");
    println!("    for batch in dataloader {{");
    println!("        // Forward pass, compute loss, backward pass...");
    println!("        ");
    println!("        // Get current learning rate");
    println!("        let lr = scheduler.step();");
    println!("        optimizer.set_learning_rate(lr);");
    println!("        ");
    println!("        // Apply gradients");
    println!("        let new_params = optimizer.step(&params, &grads).unwrap();");
    println!("    }}");
    println!("}}");
    println!("```");

    println!("\n=== Example Complete ===");
}
