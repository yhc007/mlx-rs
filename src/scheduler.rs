//! Learning rate schedulers for training neural networks.
//!
//! Schedulers adjust the learning rate during training to improve convergence.
//!
//! # Example
//!
//! ```ignore
//! use mlx_rs::scheduler::{CosineAnnealingLR, LRScheduler};
//! use mlx_rs::nn::{Adam, Optimizer};
//!
//! let mut optimizer = Adam::new(0.001);
//! let mut scheduler = CosineAnnealingLR::new(0.001, 1000);
//!
//! for step in 0..1000 {
//!     // Training step...
//!     let lr = scheduler.step();
//!     optimizer.set_learning_rate(lr);
//! }
//! ```

use std::f32::consts::PI;

/// Trait for learning rate schedulers.
pub trait LRScheduler {
    /// Get the current learning rate.
    fn get_lr(&self) -> f32;

    /// Advance the scheduler by one step and return the new learning rate.
    fn step(&mut self) -> f32;

    /// Get the current step count.
    fn current_step(&self) -> usize;

    /// Reset the scheduler to its initial state.
    fn reset(&mut self);
}

// ============================================================================
// StepLR - Step decay
// ============================================================================

/// Decays the learning rate by a factor every `step_size` steps.
///
/// ```text
/// lr = initial_lr * gamma^(step // step_size)
/// ```
///
/// # Example
///
/// ```ignore
/// use mlx_rs::scheduler::StepLR;
///
/// // Decay by 0.1 every 30 steps
/// let mut scheduler = StepLR::new(0.1, 30, 0.1);
///
/// for step in 0..100 {
///     let lr = scheduler.step();
///     // step 0-29: lr = 0.1
///     // step 30-59: lr = 0.01
///     // step 60-89: lr = 0.001
///     // step 90-99: lr = 0.0001
/// }
/// ```
#[derive(Debug, Clone)]
pub struct StepLR {
    initial_lr: f32,
    step_size: usize,
    gamma: f32,
    current_step: usize,
}

impl StepLR {
    /// Create a new StepLR scheduler.
    ///
    /// # Arguments
    ///
    /// * `initial_lr` - Initial learning rate
    /// * `step_size` - Period of learning rate decay
    /// * `gamma` - Multiplicative factor of learning rate decay (default: 0.1)
    pub fn new(initial_lr: f32, step_size: usize, gamma: f32) -> Self {
        Self {
            initial_lr,
            step_size,
            gamma,
            current_step: 0,
        }
    }
}

impl LRScheduler for StepLR {
    fn get_lr(&self) -> f32 {
        let num_decays = self.current_step / self.step_size;
        self.initial_lr * self.gamma.powi(num_decays as i32)
    }

    fn step(&mut self) -> f32 {
        let lr = self.get_lr();
        self.current_step += 1;
        lr
    }

    fn current_step(&self) -> usize {
        self.current_step
    }

    fn reset(&mut self) {
        self.current_step = 0;
    }
}

// ============================================================================
// MultiStepLR - Multi-step decay
// ============================================================================

/// Decays the learning rate by gamma at specified milestones.
///
/// # Example
///
/// ```ignore
/// use mlx_rs::scheduler::MultiStepLR;
///
/// // Decay at steps 30, 60, 90
/// let mut scheduler = MultiStepLR::new(0.1, vec![30, 60, 90], 0.1);
/// ```
#[derive(Debug, Clone)]
pub struct MultiStepLR {
    initial_lr: f32,
    milestones: Vec<usize>,
    gamma: f32,
    current_step: usize,
}

impl MultiStepLR {
    /// Create a new MultiStepLR scheduler.
    ///
    /// # Arguments
    ///
    /// * `initial_lr` - Initial learning rate
    /// * `milestones` - List of step indices where LR is decayed
    /// * `gamma` - Multiplicative factor of learning rate decay
    pub fn new(initial_lr: f32, milestones: Vec<usize>, gamma: f32) -> Self {
        let mut sorted_milestones = milestones;
        sorted_milestones.sort();
        Self {
            initial_lr,
            milestones: sorted_milestones,
            gamma,
            current_step: 0,
        }
    }
}

impl LRScheduler for MultiStepLR {
    fn get_lr(&self) -> f32 {
        let num_decays = self.milestones.iter()
            .filter(|&&m| m <= self.current_step)
            .count();
        self.initial_lr * self.gamma.powi(num_decays as i32)
    }

    fn step(&mut self) -> f32 {
        let lr = self.get_lr();
        self.current_step += 1;
        lr
    }

    fn current_step(&self) -> usize {
        self.current_step
    }

    fn reset(&mut self) {
        self.current_step = 0;
    }
}

// ============================================================================
// ExponentialLR - Exponential decay
// ============================================================================

/// Decays the learning rate exponentially every step.
///
/// ```text
/// lr = initial_lr * gamma^step
/// ```
///
/// # Example
///
/// ```ignore
/// use mlx_rs::scheduler::ExponentialLR;
///
/// let mut scheduler = ExponentialLR::new(0.1, 0.99);
/// // After 100 steps: lr ≈ 0.1 * 0.99^100 ≈ 0.0366
/// ```
#[derive(Debug, Clone)]
pub struct ExponentialLR {
    initial_lr: f32,
    gamma: f32,
    current_step: usize,
}

impl ExponentialLR {
    /// Create a new ExponentialLR scheduler.
    ///
    /// # Arguments
    ///
    /// * `initial_lr` - Initial learning rate
    /// * `gamma` - Multiplicative factor applied every step
    pub fn new(initial_lr: f32, gamma: f32) -> Self {
        Self {
            initial_lr,
            gamma,
            current_step: 0,
        }
    }
}

impl LRScheduler for ExponentialLR {
    fn get_lr(&self) -> f32 {
        self.initial_lr * self.gamma.powi(self.current_step as i32)
    }

    fn step(&mut self) -> f32 {
        let lr = self.get_lr();
        self.current_step += 1;
        lr
    }

    fn current_step(&self) -> usize {
        self.current_step
    }

    fn reset(&mut self) {
        self.current_step = 0;
    }
}

// ============================================================================
// LinearLR - Linear decay
// ============================================================================

/// Linearly decays the learning rate from initial to final over total_steps.
///
/// ```text
/// lr = initial_lr + (final_lr - initial_lr) * (step / total_steps)
/// ```
///
/// # Example
///
/// ```ignore
/// use mlx_rs::scheduler::LinearLR;
///
/// // Linearly decay from 0.1 to 0.001 over 1000 steps
/// let mut scheduler = LinearLR::new(0.1, 0.001, 1000);
/// ```
#[derive(Debug, Clone)]
pub struct LinearLR {
    initial_lr: f32,
    final_lr: f32,
    total_steps: usize,
    current_step: usize,
}

impl LinearLR {
    /// Create a new LinearLR scheduler.
    ///
    /// # Arguments
    ///
    /// * `initial_lr` - Initial learning rate
    /// * `final_lr` - Final learning rate
    /// * `total_steps` - Total number of steps for the decay
    pub fn new(initial_lr: f32, final_lr: f32, total_steps: usize) -> Self {
        Self {
            initial_lr,
            final_lr,
            total_steps,
            current_step: 0,
        }
    }
}

impl LRScheduler for LinearLR {
    fn get_lr(&self) -> f32 {
        if self.current_step >= self.total_steps {
            return self.final_lr;
        }
        let progress = self.current_step as f32 / self.total_steps as f32;
        self.initial_lr + (self.final_lr - self.initial_lr) * progress
    }

    fn step(&mut self) -> f32 {
        let lr = self.get_lr();
        self.current_step += 1;
        lr
    }

    fn current_step(&self) -> usize {
        self.current_step
    }

    fn reset(&mut self) {
        self.current_step = 0;
    }
}

// ============================================================================
// CosineAnnealingLR - Cosine annealing
// ============================================================================

/// Cosine annealing learning rate schedule.
///
/// ```text
/// lr = min_lr + 0.5 * (max_lr - min_lr) * (1 + cos(π * step / total_steps))
/// ```
///
/// This schedule starts at `max_lr`, decreases following a cosine curve,
/// and reaches `min_lr` at `total_steps`.
///
/// # Example
///
/// ```ignore
/// use mlx_rs::scheduler::CosineAnnealingLR;
///
/// let mut scheduler = CosineAnnealingLR::new(0.1, 1000).min_lr(1e-6);
/// ```
#[derive(Debug, Clone)]
pub struct CosineAnnealingLR {
    max_lr: f32,
    min_lr: f32,
    total_steps: usize,
    current_step: usize,
}

impl CosineAnnealingLR {
    /// Create a new CosineAnnealingLR scheduler.
    ///
    /// # Arguments
    ///
    /// * `max_lr` - Maximum (initial) learning rate
    /// * `total_steps` - Total number of steps
    pub fn new(max_lr: f32, total_steps: usize) -> Self {
        Self {
            max_lr,
            min_lr: 0.0,
            total_steps,
            current_step: 0,
        }
    }

    /// Set the minimum learning rate (default: 0.0).
    pub fn min_lr(mut self, min_lr: f32) -> Self {
        self.min_lr = min_lr;
        self
    }
}

impl LRScheduler for CosineAnnealingLR {
    fn get_lr(&self) -> f32 {
        if self.current_step >= self.total_steps {
            return self.min_lr;
        }
        let progress = self.current_step as f32 / self.total_steps as f32;
        self.min_lr + 0.5 * (self.max_lr - self.min_lr) * (1.0 + (PI * progress).cos())
    }

    fn step(&mut self) -> f32 {
        let lr = self.get_lr();
        self.current_step += 1;
        lr
    }

    fn current_step(&self) -> usize {
        self.current_step
    }

    fn reset(&mut self) {
        self.current_step = 0;
    }
}

// ============================================================================
// CosineAnnealingWarmRestarts - Cosine annealing with warm restarts
// ============================================================================

/// Cosine annealing with warm restarts (SGDR).
///
/// The learning rate follows a cosine schedule and restarts at the maximum
/// after each cycle. Optionally, the cycle length can increase by a factor.
///
/// # Example
///
/// ```ignore
/// use mlx_rs::scheduler::CosineAnnealingWarmRestarts;
///
/// // Restart every 100 steps, double cycle length each restart
/// let mut scheduler = CosineAnnealingWarmRestarts::new(0.1, 100)
///     .t_mult(2.0)
///     .min_lr(1e-6);
/// ```
#[derive(Debug, Clone)]
pub struct CosineAnnealingWarmRestarts {
    max_lr: f32,
    min_lr: f32,
    t_0: usize,       // Initial cycle length
    t_mult: f32,      // Cycle length multiplier
    current_step: usize,
}

impl CosineAnnealingWarmRestarts {
    /// Create a new CosineAnnealingWarmRestarts scheduler.
    ///
    /// # Arguments
    ///
    /// * `max_lr` - Maximum learning rate
    /// * `t_0` - Number of steps for the first restart
    pub fn new(max_lr: f32, t_0: usize) -> Self {
        Self {
            max_lr,
            min_lr: 0.0,
            t_0,
            t_mult: 1.0,
            current_step: 0,
        }
    }

    /// Set the cycle length multiplier (default: 1.0).
    pub fn t_mult(mut self, t_mult: f32) -> Self {
        self.t_mult = t_mult;
        self
    }

    /// Set the minimum learning rate (default: 0.0).
    pub fn min_lr(mut self, min_lr: f32) -> Self {
        self.min_lr = min_lr;
        self
    }

    /// Get the current position within the cycle.
    fn get_cycle_position(&self) -> (f32, f32) {
        if self.t_mult == 1.0 {
            // Simple case: constant cycle length
            let cycle_pos = self.current_step % self.t_0;
            (cycle_pos as f32, self.t_0 as f32)
        } else {
            // Variable cycle length
            let mut t_cur = 0usize;
            let mut t_i = self.t_0;

            while t_cur + t_i <= self.current_step {
                t_cur += t_i;
                t_i = (t_i as f32 * self.t_mult) as usize;
            }

            let pos_in_cycle = self.current_step - t_cur;
            (pos_in_cycle as f32, t_i as f32)
        }
    }
}

impl LRScheduler for CosineAnnealingWarmRestarts {
    fn get_lr(&self) -> f32 {
        let (pos, cycle_len) = self.get_cycle_position();
        let progress = pos / cycle_len;
        self.min_lr + 0.5 * (self.max_lr - self.min_lr) * (1.0 + (PI * progress).cos())
    }

    fn step(&mut self) -> f32 {
        let lr = self.get_lr();
        self.current_step += 1;
        lr
    }

    fn current_step(&self) -> usize {
        self.current_step
    }

    fn reset(&mut self) {
        self.current_step = 0;
    }
}

// ============================================================================
// WarmupCosine - Linear warmup + cosine decay
// ============================================================================

/// Linear warmup followed by cosine decay.
///
/// This is the most common schedule for training transformers:
/// 1. Linear warmup from 0 to max_lr over warmup_steps
/// 2. Cosine decay from max_lr to min_lr over remaining steps
///
/// # Example
///
/// ```ignore
/// use mlx_rs::scheduler::WarmupCosine;
///
/// // 1000 warmup steps, 10000 total steps
/// let mut scheduler = WarmupCosine::new(0.001, 1000, 10000);
/// ```
#[derive(Debug, Clone)]
pub struct WarmupCosine {
    max_lr: f32,
    min_lr: f32,
    warmup_steps: usize,
    total_steps: usize,
    current_step: usize,
}

impl WarmupCosine {
    /// Create a new WarmupCosine scheduler.
    ///
    /// # Arguments
    ///
    /// * `max_lr` - Maximum learning rate (reached at end of warmup)
    /// * `warmup_steps` - Number of warmup steps
    /// * `total_steps` - Total number of training steps
    pub fn new(max_lr: f32, warmup_steps: usize, total_steps: usize) -> Self {
        Self {
            max_lr,
            min_lr: 0.0,
            warmup_steps,
            total_steps,
            current_step: 0,
        }
    }

    /// Set the minimum learning rate (default: 0.0).
    pub fn min_lr(mut self, min_lr: f32) -> Self {
        self.min_lr = min_lr;
        self
    }
}

impl LRScheduler for WarmupCosine {
    fn get_lr(&self) -> f32 {
        if self.current_step < self.warmup_steps {
            // Linear warmup
            let progress = self.current_step as f32 / self.warmup_steps as f32;
            self.min_lr + (self.max_lr - self.min_lr) * progress
        } else if self.current_step >= self.total_steps {
            self.min_lr
        } else {
            // Cosine decay
            let decay_steps = self.total_steps - self.warmup_steps;
            let decay_progress = (self.current_step - self.warmup_steps) as f32 / decay_steps as f32;
            self.min_lr + 0.5 * (self.max_lr - self.min_lr) * (1.0 + (PI * decay_progress).cos())
        }
    }

    fn step(&mut self) -> f32 {
        let lr = self.get_lr();
        self.current_step += 1;
        lr
    }

    fn current_step(&self) -> usize {
        self.current_step
    }

    fn reset(&mut self) {
        self.current_step = 0;
    }
}

// ============================================================================
// WarmupLinear - Linear warmup + linear decay
// ============================================================================

/// Linear warmup followed by linear decay.
///
/// # Example
///
/// ```ignore
/// use mlx_rs::scheduler::WarmupLinear;
///
/// let mut scheduler = WarmupLinear::new(0.001, 1000, 10000);
/// ```
#[derive(Debug, Clone)]
pub struct WarmupLinear {
    max_lr: f32,
    min_lr: f32,
    warmup_steps: usize,
    total_steps: usize,
    current_step: usize,
}

impl WarmupLinear {
    /// Create a new WarmupLinear scheduler.
    pub fn new(max_lr: f32, warmup_steps: usize, total_steps: usize) -> Self {
        Self {
            max_lr,
            min_lr: 0.0,
            warmup_steps,
            total_steps,
            current_step: 0,
        }
    }

    /// Set the minimum learning rate (default: 0.0).
    pub fn min_lr(mut self, min_lr: f32) -> Self {
        self.min_lr = min_lr;
        self
    }
}

impl LRScheduler for WarmupLinear {
    fn get_lr(&self) -> f32 {
        if self.current_step < self.warmup_steps {
            // Linear warmup
            let progress = self.current_step as f32 / self.warmup_steps as f32;
            self.min_lr + (self.max_lr - self.min_lr) * progress
        } else if self.current_step >= self.total_steps {
            self.min_lr
        } else {
            // Linear decay
            let decay_steps = self.total_steps - self.warmup_steps;
            let decay_progress = (self.current_step - self.warmup_steps) as f32 / decay_steps as f32;
            self.max_lr - (self.max_lr - self.min_lr) * decay_progress
        }
    }

    fn step(&mut self) -> f32 {
        let lr = self.get_lr();
        self.current_step += 1;
        lr
    }

    fn current_step(&self) -> usize {
        self.current_step
    }

    fn reset(&mut self) {
        self.current_step = 0;
    }
}

// ============================================================================
// OneCycleLR - One cycle policy
// ============================================================================

/// One cycle learning rate policy.
///
/// The learning rate goes through three phases:
/// 1. Linear warmup from initial_lr to max_lr
/// 2. Cosine annealing from max_lr to initial_lr
/// 3. Cooldown from initial_lr to final_lr
///
/// # Example
///
/// ```ignore
/// use mlx_rs::scheduler::OneCycleLR;
///
/// let mut scheduler = OneCycleLR::new(0.1, 10000)
///     .pct_start(0.3)  // 30% warmup
///     .div_factor(25.0)  // initial_lr = max_lr / 25
///     .final_div_factor(1e4);  // final_lr = initial_lr / 10000
/// ```
#[derive(Debug, Clone)]
pub struct OneCycleLR {
    max_lr: f32,
    total_steps: usize,
    pct_start: f32,      // Percentage of cycle spent increasing LR
    div_factor: f32,     // initial_lr = max_lr / div_factor
    final_div_factor: f32, // final_lr = initial_lr / final_div_factor
    current_step: usize,
}

impl OneCycleLR {
    /// Create a new OneCycleLR scheduler.
    ///
    /// # Arguments
    ///
    /// * `max_lr` - Maximum learning rate
    /// * `total_steps` - Total number of training steps
    pub fn new(max_lr: f32, total_steps: usize) -> Self {
        Self {
            max_lr,
            total_steps,
            pct_start: 0.3,
            div_factor: 25.0,
            final_div_factor: 1e4,
            current_step: 0,
        }
    }

    /// Set the percentage of cycle spent increasing LR (default: 0.3).
    pub fn pct_start(mut self, pct_start: f32) -> Self {
        self.pct_start = pct_start;
        self
    }

    /// Set the initial LR divisor: initial_lr = max_lr / div_factor (default: 25.0).
    pub fn div_factor(mut self, div_factor: f32) -> Self {
        self.div_factor = div_factor;
        self
    }

    /// Set the final LR divisor: final_lr = initial_lr / final_div_factor (default: 1e4).
    pub fn final_div_factor(mut self, final_div_factor: f32) -> Self {
        self.final_div_factor = final_div_factor;
        self
    }
}

impl LRScheduler for OneCycleLR {
    fn get_lr(&self) -> f32 {
        let initial_lr = self.max_lr / self.div_factor;
        let final_lr = initial_lr / self.final_div_factor;
        let warmup_steps = (self.total_steps as f32 * self.pct_start) as usize;

        if self.current_step < warmup_steps {
            // Phase 1: Linear warmup
            let progress = self.current_step as f32 / warmup_steps as f32;
            initial_lr + (self.max_lr - initial_lr) * progress
        } else if self.current_step >= self.total_steps {
            final_lr
        } else {
            // Phase 2: Cosine annealing to final_lr
            let decay_steps = self.total_steps - warmup_steps;
            let decay_progress = (self.current_step - warmup_steps) as f32 / decay_steps as f32;
            final_lr + 0.5 * (self.max_lr - final_lr) * (1.0 + (PI * decay_progress).cos())
        }
    }

    fn step(&mut self) -> f32 {
        let lr = self.get_lr();
        self.current_step += 1;
        lr
    }

    fn current_step(&self) -> usize {
        self.current_step
    }

    fn reset(&mut self) {
        self.current_step = 0;
    }
}

// ============================================================================
// PolynomialLR - Polynomial decay
// ============================================================================

/// Polynomial learning rate decay.
///
/// ```text
/// lr = (initial_lr - final_lr) * (1 - step/total_steps)^power + final_lr
/// ```
///
/// # Example
///
/// ```ignore
/// use mlx_rs::scheduler::PolynomialLR;
///
/// // Quadratic decay (power=2)
/// let mut scheduler = PolynomialLR::new(0.1, 1000).power(2.0);
/// ```
#[derive(Debug, Clone)]
pub struct PolynomialLR {
    initial_lr: f32,
    final_lr: f32,
    total_steps: usize,
    power: f32,
    current_step: usize,
}

impl PolynomialLR {
    /// Create a new PolynomialLR scheduler.
    ///
    /// # Arguments
    ///
    /// * `initial_lr` - Initial learning rate
    /// * `total_steps` - Total number of steps
    pub fn new(initial_lr: f32, total_steps: usize) -> Self {
        Self {
            initial_lr,
            final_lr: 0.0,
            total_steps,
            power: 1.0,
            current_step: 0,
        }
    }

    /// Set the final learning rate (default: 0.0).
    pub fn final_lr(mut self, final_lr: f32) -> Self {
        self.final_lr = final_lr;
        self
    }

    /// Set the polynomial power (default: 1.0 = linear).
    pub fn power(mut self, power: f32) -> Self {
        self.power = power;
        self
    }
}

impl LRScheduler for PolynomialLR {
    fn get_lr(&self) -> f32 {
        if self.current_step >= self.total_steps {
            return self.final_lr;
        }
        let progress = 1.0 - self.current_step as f32 / self.total_steps as f32;
        (self.initial_lr - self.final_lr) * progress.powf(self.power) + self.final_lr
    }

    fn step(&mut self) -> f32 {
        let lr = self.get_lr();
        self.current_step += 1;
        lr
    }

    fn current_step(&self) -> usize {
        self.current_step
    }

    fn reset(&mut self) {
        self.current_step = 0;
    }
}

// ============================================================================
// ConstantLR - Constant learning rate (for completeness)
// ============================================================================

/// Constant learning rate scheduler.
///
/// Useful for combining with other schedulers or as a baseline.
#[derive(Debug, Clone)]
pub struct ConstantLR {
    lr: f32,
    current_step: usize,
}

impl ConstantLR {
    /// Create a new ConstantLR scheduler.
    pub fn new(lr: f32) -> Self {
        Self {
            lr,
            current_step: 0,
        }
    }
}

impl LRScheduler for ConstantLR {
    fn get_lr(&self) -> f32 {
        self.lr
    }

    fn step(&mut self) -> f32 {
        self.current_step += 1;
        self.lr
    }

    fn current_step(&self) -> usize {
        self.current_step
    }

    fn reset(&mut self) {
        self.current_step = 0;
    }
}
