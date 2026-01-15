//! Stream management for MLX

use mlx_sys;

use crate::device::Device;

/// A stream for ordering MLX operations
#[derive(Debug)]
pub struct Stream {
    pub(crate) inner: mlx_sys::mlx_stream,
}

impl Stream {
    /// Get the default stream for a device
    pub fn default_stream(device: &Device) -> Self {
        let mut inner = unsafe { mlx_sys::mlx_stream_new() };
        unsafe {
            mlx_sys::mlx_get_default_stream(&mut inner, device.as_raw());
        }
        Self { inner }
    }

    /// Get the default stream for the default device
    pub fn default() -> Self {
        let device = Device::default_device();
        Self::default_stream(&device)
    }

    /// Create a new stream on a device
    pub fn new(device: &Device) -> Self {
        let inner = unsafe { mlx_sys::mlx_stream_new_device(device.as_raw()) };
        Self { inner }
    }

    /// Get a CPU stream (required for some operations not yet supported on GPU)
    pub fn cpu() -> Self {
        let device = Device::cpu();
        Self::default_stream(&device)
    }

    /// Synchronize this stream
    pub fn synchronize(&self) {
        unsafe {
            mlx_sys::mlx_synchronize(self.inner);
        }
    }

    /// Get the raw stream handle (for internal use)
    pub(crate) fn as_raw(&self) -> mlx_sys::mlx_stream {
        self.inner
    }
}

impl Clone for Stream {
    fn clone(&self) -> Self {
        let mut inner = unsafe { mlx_sys::mlx_stream_new() };
        unsafe {
            mlx_sys::mlx_stream_set(&mut inner, self.inner);
        }
        Self { inner }
    }
}

impl Drop for Stream {
    fn drop(&mut self) {
        unsafe {
            mlx_sys::mlx_stream_free(self.inner);
        }
    }
}

// Stream is Send and Sync since MLX handles thread safety internally
unsafe impl Send for Stream {}
unsafe impl Sync for Stream {}
