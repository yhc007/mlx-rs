//! Device management for MLX

use mlx_sys;
use std::fmt;

/// Device types supported by MLX
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum DeviceType {
    /// CPU device
    Cpu,
    /// GPU device (Metal on Apple silicon)
    Gpu,
}

impl fmt::Display for DeviceType {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            DeviceType::Cpu => write!(f, "cpu"),
            DeviceType::Gpu => write!(f, "gpu"),
        }
    }
}

impl From<DeviceType> for mlx_sys::mlx_device_type {
    fn from(dt: DeviceType) -> Self {
        match dt {
            DeviceType::Cpu => mlx_sys::mlx_device_type__MLX_CPU,
            DeviceType::Gpu => mlx_sys::mlx_device_type__MLX_GPU,
        }
    }
}

impl From<mlx_sys::mlx_device_type> for DeviceType {
    fn from(dt: mlx_sys::mlx_device_type) -> Self {
        match dt {
            mlx_sys::mlx_device_type__MLX_CPU => DeviceType::Cpu,
            mlx_sys::mlx_device_type__MLX_GPU => DeviceType::Gpu,
            _ => DeviceType::Cpu,
        }
    }
}

/// A device on which MLX operations can run
#[derive(Debug)]
pub struct Device {
    pub(crate) inner: mlx_sys::mlx_device,
}

impl Device {
    /// Create a new device of the specified type
    pub fn new(device_type: DeviceType) -> Self {
        let inner = unsafe { mlx_sys::mlx_device_new_type(device_type.into(), 0) };
        Self { inner }
    }

    /// Create a new CPU device
    pub fn cpu() -> Self {
        Self::new(DeviceType::Cpu)
    }

    /// Create a new GPU device
    pub fn gpu() -> Self {
        Self::new(DeviceType::Gpu)
    }

    /// Get the default device
    pub fn default_device() -> Self {
        let mut inner = unsafe { mlx_sys::mlx_device_new() };
        unsafe {
            mlx_sys::mlx_get_default_device(&mut inner);
        }
        Self { inner }
    }

    /// Set the default device
    pub fn set_default(device: &Device) {
        unsafe {
            mlx_sys::mlx_set_default_device(device.inner);
        }
    }

    /// Get the device type
    pub fn device_type(&self) -> DeviceType {
        let mut dtype: mlx_sys::mlx_device_type = mlx_sys::mlx_device_type__MLX_CPU;
        unsafe {
            mlx_sys::mlx_device_get_type(&mut dtype, self.inner);
        }
        dtype.into()
    }

    /// Check if this is a CPU device
    pub fn is_cpu(&self) -> bool {
        self.device_type() == DeviceType::Cpu
    }

    /// Check if this is a GPU device
    pub fn is_gpu(&self) -> bool {
        self.device_type() == DeviceType::Gpu
    }

    /// Get the raw device handle (for internal use)
    pub(crate) fn as_raw(&self) -> mlx_sys::mlx_device {
        self.inner
    }
}

impl Clone for Device {
    fn clone(&self) -> Self {
        let mut inner = unsafe { mlx_sys::mlx_device_new() };
        unsafe {
            mlx_sys::mlx_device_set(&mut inner, self.inner);
        }
        Self { inner }
    }
}

impl Drop for Device {
    fn drop(&mut self) {
        unsafe {
            mlx_sys::mlx_device_free(self.inner);
        }
    }
}

impl Default for Device {
    fn default() -> Self {
        Self::default_device()
    }
}

// Device is Send and Sync since MLX handles thread safety internally
unsafe impl Send for Device {}
unsafe impl Sync for Device {}
