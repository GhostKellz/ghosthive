//! By convention, root.zig is the root source file when making a library.
const std = @import("std");
const builtin = @import("builtin");
const build_options = @import("build_options");

const gpu_enabled: bool = build_options.enable_gpu;
pub const gpu_support_enabled = gpu_enabled;

pub const TensorError = error{
    IncompatibleShapes,
    ShapeMismatch,
    Not2D,
    InvalidShape,
    InvalidDimension,
    OutOfMemory,
    NullPointer,
    InvalidInput,
    ZeroDivision,
    InvalidIndex,
    PoolError,
    DeviceNotAvailable,
    CudaError,
    ComputeShaderError,
    DeviceSyncError,
    UnsupportedDevice,
    GpuUnavailable,
    GpuMemoryError,
    GpuKernelError,
    InvalidDevice,
};

// Enhanced error context for better debugging
pub const TensorErrorContext = struct {
    error_type: TensorError,
    message: []const u8,
    expected_shape: ?[]const usize = null,
    actual_shape: ?[]const usize = null,
    operation: ?[]const u8 = null,
    file: []const u8 = "unknown",
    line: u32 = 0,

    pub fn format(
        self: TensorErrorContext,
        comptime fmt: []const u8,
        options: anytype,
        writer: anytype,
    ) !void {
        _ = fmt;
        _ = options;

        try writer.print("TensorError.{s}: {s}", .{ @errorName(self.error_type), self.message });

        if (self.operation) |op| {
            try writer.print(" (during {s})", .{op});
        }

        if (self.expected_shape) |expected| {
            try writer.print("\n  Expected shape: [");
            for (expected, 0..) |dim, i| {
                if (i > 0) try writer.print(", ");
                try writer.print("{d}", .{dim});
            }
            try writer.print("]");
        }

        if (self.actual_shape) |actual| {
            try writer.print("\n  Actual shape: [");
            for (actual, 0..) |dim, i| {
                if (i > 0) try writer.print(", ");
                try writer.print("{d}", .{dim});
            }
            try writer.print("]");
        }

        try writer.print("\n  Location: {s}:{d}", .{ self.file, self.line });
    }
};

// Helper functions for creating descriptive errors
pub fn tensorError(
    err: TensorError,
    message: []const u8,
    operation: ?[]const u8,
    expected_shape: ?[]const usize,
    actual_shape: ?[]const usize,
    src: std.builtin.SourceLocation,
) TensorErrorContext {
    return TensorErrorContext{
        .error_type = err,
        .message = message,
        .expected_shape = expected_shape,
        .actual_shape = actual_shape,
        .operation = operation,
        .file = src.file,
        .line = src.line,
    };
}

pub fn shapeError(
    expected: []const usize,
    actual: []const usize,
    operation: []const u8,
    src: std.builtin.SourceLocation,
) TensorErrorContext {
    return tensorError(
        TensorError.ShapeMismatch,
        "Tensor shapes are incompatible for this operation",
        operation,
        expected,
        actual,
        src,
    );
}

pub fn dimensionError(
    expected_dims: usize,
    actual_dims: usize,
    operation: []const u8,
    src: std.builtin.SourceLocation,
) TensorErrorContext {
    const expected_shape = [_]usize{expected_dims};
    const actual_shape = [_]usize{actual_dims};
    return tensorError(
        TensorError.InvalidDimension,
        "Tensor has wrong number of dimensions",
        operation,
        &expected_shape,
        &actual_shape,
        src,
    );
}

pub const DeviceType = enum {
    cpu,
    cuda,
    vulkan,
    webgpu,
    metal,
};

pub const Device = struct {
    device_type: DeviceType,
    device_id: u32,
    name: []const u8,
    memory_size: u64,
    is_available: bool,

    pub fn init(device_type: DeviceType, device_id: u32, name: []const u8) Device {
        return Device{
            .device_type = device_type,
            .device_id = device_id,
            .name = name,
            .memory_size = 0, // Will be set during initialization
            .is_available = false,
        };
    }

    pub fn isCpu(self: *const Device) bool {
        return self.device_type == .cpu;
    }

    pub fn isGpu(self: *const Device) bool {
        return !self.isCpu();
    }
};

pub const DeviceManager = struct {
    devices: std.ArrayList(Device),
    current_device: ?*Device,
    allocator: std.mem.Allocator,

    pub fn init(allocator: std.mem.Allocator) DeviceManager {
        return DeviceManager{
            .devices = std.ArrayList(Device){},
            .current_device = null,
            .allocator = allocator,
        };
    }

    pub fn deinit(self: *DeviceManager) void {
        self.devices.deinit(self.allocator);
    }

    pub fn detectDevices(self: *DeviceManager) !void {
        // Always add CPU device
        const cpu_device = Device.init(.cpu, 0, "CPU");
        try self.devices.append(self.allocator, cpu_device);

        if (!gpu_enabled) {
            self.current_device = &self.devices.items[0];
            self.current_device.?.is_available = true;
            return;
        }

        // Detect CUDA devices (simulation for now)
        if (builtin.target.os.tag == .linux or builtin.target.os.tag == .windows) {
            // In a real implementation, we would call CUDA runtime API here
            if (self.detectCudaRuntime()) {
                var cuda_device = Device.init(.cuda, 0, "NVIDIA GPU");
                cuda_device.memory_size = 8 * 1024 * 1024 * 1024; // 8GB simulation
                cuda_device.is_available = true;
                try self.devices.append(self.allocator, cuda_device);
            }
        }

        // Detect Vulkan support (simulation)
        if (self.detectVulkanSupport()) {
            var vulkan_device = Device.init(.vulkan, 0, "Vulkan GPU");
            vulkan_device.memory_size = 6 * 1024 * 1024 * 1024; // 6GB simulation
            vulkan_device.is_available = true;
            try self.devices.append(self.allocator, vulkan_device);
        }

        // Set default device
        if (self.devices.items.len > 0) {
            self.current_device = &self.devices.items[0];
            self.current_device.?.is_available = true;
        }
    }

    pub fn setDevice(self: *DeviceManager, device_type: DeviceType) TensorError!void {
        for (self.devices.items) |*device| {
            if (device.device_type == device_type and device.is_available) {
                self.current_device = device;
                return;
            }
        }
        return TensorError.DeviceNotAvailable;
    }

    pub fn getCurrentDevice(self: *const DeviceManager) ?*const Device {
        return self.current_device;
    }

    pub fn listDevices(self: *const DeviceManager) []const Device {
        return self.devices.items;
    }

    fn detectCudaRuntime(self: *DeviceManager) bool {
        _ = self;
        return gpu_enabled;
    }

    fn detectVulkanSupport(self: *DeviceManager) bool {
        _ = self;
        // In a real implementation, this would check for Vulkan loader
        // For now, return false to avoid dependency issues
        return false;
    }
};

pub const GpuMemory = struct {
    ptr: ?*anyopaque,
    size: usize,
    device: *const Device,
    allocator: std.mem.Allocator,

    pub fn init(allocator: std.mem.Allocator, device: *const Device, size: usize) !GpuMemory {
        if (device.isCpu()) {
            // For CPU, just allocate regular memory
            const ptr = try allocator.alloc(u8, size);
            return GpuMemory{
                .ptr = ptr.ptr,
                .size = size,
                .device = device,
                .allocator = allocator,
            };
        } else {
            // For GPU devices, we would allocate GPU memory
            // For now, simulate with CPU memory
            const ptr = try allocator.alloc(u8, size);
            return GpuMemory{
                .ptr = ptr.ptr,
                .size = size,
                .device = device,
                .allocator = allocator,
            };
        }
    }

    pub fn deinit(self: *GpuMemory) void {
        if (self.ptr) |ptr| {
            const slice = @as([*]u8, @ptrCast(ptr))[0..self.size];
            self.allocator.free(slice);
            self.ptr = null;
        }
    }

    pub fn copyFromHost(self: *GpuMemory, data: []const u8) TensorError!void {
        if (self.ptr == null or data.len > self.size) return TensorError.InvalidInput;

        const dest = @as([*]u8, @ptrCast(self.ptr.?))[0..self.size];
        @memcpy(dest[0..data.len], data);

        if (self.device.isGpu()) {
            // In a real implementation, this would be a CUDA memcpy or similar
            // For now, it's just a regular memory copy
        }
    }

    pub fn copyToHost(self: *const GpuMemory, data: []u8) TensorError!void {
        if (self.ptr == null or data.len > self.size) return TensorError.InvalidInput;

        const src = @as([*]const u8, @ptrCast(self.ptr.?))[0..self.size];
        @memcpy(data[0..@min(data.len, self.size)], src[0..@min(data.len, self.size)]);

        if (self.device.isGpu()) {
            // In a real implementation, this would be a CUDA memcpy or similar
        }
    }
};

pub const TensorPool = struct {
    allocator: std.mem.Allocator,
    pools: std.HashMap(usize, std.ArrayList([]f32), std.hash_map.AutoContext(usize), std.hash_map.default_max_load_percentage),
    mutex: std.Thread.Mutex,

    pub fn init(allocator: std.mem.Allocator) TensorPool {
        return TensorPool{
            .allocator = allocator,
            .pools = std.HashMap(usize, std.ArrayList([]f32), std.hash_map.AutoContext(usize), std.hash_map.default_max_load_percentage).init(allocator),
            .mutex = std.Thread.Mutex{},
        };
    }

    pub fn deinit(self: *TensorPool) void {
        var iterator = self.pools.iterator();
        while (iterator.next()) |entry| {
            for (entry.value_ptr.items) |data| {
                self.allocator.free(data);
            }
            entry.value_ptr.deinit();
        }
        self.pools.deinit();
    }

    pub fn acquire(self: *TensorPool, size: usize) ![]f32 {
        self.mutex.lock();
        defer self.mutex.unlock();

        if (self.pools.getPtr(size)) |pool| {
            if (pool.items.len > 0) {
                return pool.pop() orelse unreachable;
            }
        }

        return try self.allocator.alloc(f32, size);
    }

    pub fn release(self: *TensorPool, data: []f32) !void {
        self.mutex.lock();
        defer self.mutex.unlock();

        const size = data.len;
        const gop = try self.pools.getOrPut(size);
        if (!gop.found_existing) {
            gop.value_ptr.* = std.ArrayList([]f32){};
        }
        try gop.value_ptr.append(self.allocator, data);
    }
};

// CUDA C API Bindings (minimal subset)
const CudaError = enum(c_int) {
    success = 0,
    invalid_value = 1,
    out_of_memory = 2,
    not_initialized = 3,
    deinitialized = 4,
    no_device = 38,
    invalid_device = 10,
    _,
};

// Stub CUDA functions that always return errors (CUDA not available)
fn cuInit(flags: c_uint) CudaError {
    _ = flags;
    return .not_initialized;
}

fn cuDeviceGetCount(count: *c_int) CudaError {
    count.* = 0;
    return .no_device;
}

fn cuMemAlloc(dptr: *?*anyopaque, bytesize: usize) CudaError {
    _ = dptr;
    _ = bytesize;
    return .not_initialized;
}

fn cuMemFree(dptr: ?*anyopaque) CudaError {
    _ = dptr;
    return .not_initialized;
}

fn cuMemcpyHtoD(dstDevice: ?*anyopaque, srcHost: *const anyopaque, ByteCount: usize) CudaError {
    _ = dstDevice;
    _ = srcHost;
    _ = ByteCount;
    return .not_initialized;
}

fn cuMemcpyDtoH(dstHost: *anyopaque, srcDevice: ?*anyopaque, ByteCount: usize) CudaError {
    _ = dstHost;
    _ = srcDevice;
    _ = ByteCount;
    return .not_initialized;
}

pub const CudaContext = struct {
    initialized: bool,
    device_count: i32,

    pub fn init() CudaContext {
        var ctx = CudaContext{
            .initialized = false,
            .device_count = 0,
        };

        if (gpu_enabled) {
            ctx.initialized = true;
            ctx.device_count = 1;
            return ctx;
        }

        // Try to initialize CUDA
        const init_result = cuInit(0);
        if (init_result == .success) {
            const count_result = cuDeviceGetCount(&ctx.device_count);
            if (count_result == .success and ctx.device_count > 0) {
                ctx.initialized = true;
            }
        }

        return ctx;
    }

    pub fn isAvailable(self: *const CudaContext) bool {
        return self.initialized and self.device_count > 0;
    }
};

// Global CUDA context
var cuda_context: ?CudaContext = null;

pub fn getCudaContext() *CudaContext {
    if (cuda_context == null) {
        cuda_context = CudaContext.init();
    }
    return &cuda_context.?;
}

pub const GpuKernel = struct {
    pub fn vectorAdd(a: []const f32, b: []const f32, result: []f32, device: *const Device) TensorError!void {
        return vectorAddWithFallback(a, b, result, device) catch |err| {
            switch (err) {
                TensorError.GpuUnavailable, TensorError.UnsupportedDevice => {
                    // Automatic fallback to optimized CPU
                    return vectorAddCpu(a, b, result);
                },
                else => return err,
            }
        };
    }

    fn vectorAddWithFallback(a: []const f32, b: []const f32, result: []f32, device: *const Device) TensorError!void {
        if (a.len != b.len or a.len != result.len) return TensorError.ShapeMismatch;

        if (!gpu_enabled) {
            if (device.isGpu()) return TensorError.GpuUnavailable;
            return vectorAddCpu(a, b, result);
        }

        switch (device.device_type) {
            .cpu => return vectorAddCpu(a, b, result),
            .cuda => return vectorAddCuda(a, b, result),
            .vulkan => return vectorAddVulkan(a, b, result),
            .webgpu => return vectorAddWebGpu(a, b, result),
            .metal => return vectorAddMetal(a, b, result),
        }
    }

    fn vectorAddCpu(a: []const f32, b: []const f32, result: []f32) TensorError!void {
        // CPU implementation with SIMD optimization
        const Vec4 = @Vector(4, f32);
        const simd_len = (a.len / 4) * 4;

        var i: usize = 0;
        while (i < simd_len) : (i += 4) {
            const a_vec: Vec4 = a[i..i+4][0..4].*;
            const b_vec: Vec4 = b[i..i+4][0..4].*;
            const result_vec = a_vec + b_vec;
            result[i..i+4][0..4].* = result_vec;
        }

        while (i < a.len) : (i += 1) {
            result[i] = a[i] + b[i];
        }
    }

    fn vectorAddCuda(a: []const f32, b: []const f32, result: []f32) TensorError!void {
        const ctx = getCudaContext();
        if (!ctx.isAvailable()) {
            return TensorError.UnsupportedDevice;
        }

        // Use CUDA functions directly (they will return errors if CUDA is unavailable)

        const size_bytes = a.len * @sizeOf(f32);

        // Allocate GPU memory
        var d_a: ?*anyopaque = null;
        var d_b: ?*anyopaque = null;
        var d_result: ?*anyopaque = null;

        var cuda_result = cuMemAlloc(&d_a, size_bytes);
        if (cuda_result != .success) return TensorError.GpuMemoryError;

        cuda_result = cuMemAlloc(&d_b, size_bytes);
        if (cuda_result != .success) {
            _ = cuMemFree(d_a);
            return TensorError.GpuMemoryError;
        }

        cuda_result = cuMemAlloc(&d_result, size_bytes);
        if (cuda_result != .success) {
            _ = cuMemFree(d_a);
            _ = cuMemFree(d_b);
            return TensorError.GpuMemoryError;
        }

        // Copy data to GPU
        cuda_result = cuMemcpyHtoD(d_a, a.ptr, size_bytes);
        if (cuda_result != .success) {
            _ = cuMemFree(d_a);
            _ = cuMemFree(d_b);
            _ = cuMemFree(d_result);
            return TensorError.GpuMemoryError;
        }

        cuda_result = cuMemcpyHtoD(d_b, b.ptr, size_bytes);
        if (cuda_result != .success) {
            _ = cuMemFree(d_a);
            _ = cuMemFree(d_b);
            _ = cuMemFree(d_result);
            return TensorError.GpuMemoryError;
        }

        // TODO: Launch CUDA kernel here
        // For now, simulate kernel execution by copying one input to output
        // In a real implementation, this would launch a vectorAdd kernel
        cuda_result = cuMemcpyHtoD(d_result, a.ptr, size_bytes);
        if (cuda_result != .success) {
            _ = cuMemFree(d_a);
            _ = cuMemFree(d_b);
            _ = cuMemFree(d_result);
            return TensorError.GpuKernelError;
        }

        // Copy result back to host
        cuda_result = cuMemcpyDtoH(result.ptr, d_result, size_bytes);
        if (cuda_result != .success) {
            _ = cuMemFree(d_a);
            _ = cuMemFree(d_b);
            _ = cuMemFree(d_result);
            return TensorError.GpuMemoryError;
        }

        // Clean up GPU memory
        _ = cuMemFree(d_a);
        _ = cuMemFree(d_b);
        _ = cuMemFree(d_result);

        // Since we don't have actual kernel, fall back to CPU computation on the result
        for (0..result.len) |i| {
            result[i] = a[i] + b[i];
        }
    }

    fn vectorAddVulkan(a: []const f32, b: []const f32, result: []f32) TensorError!void {
        _ = a;
        _ = b;
        _ = result;
        // Vulkan compute shader implementation
        // GLSL compute shader source for vector addition:
        // #version 450
        //
        // layout(local_size_x = 64, local_size_y = 1, local_size_z = 1) in;
        //
        // layout(set = 0, binding = 0) readonly buffer InputA {
        //     float data_a[];
        // };
        //
        // layout(set = 0, binding = 1) readonly buffer InputB {
        //     float data_b[];
        // };
        //
        // layout(set = 0, binding = 2) writeonly buffer Output {
        //     float data_out[];
        // };
        //
        // void main() {
        //     uint index = gl_GlobalInvocationID.x;
        //     if (index >= data_a.length()) return;
        //     data_out[index] = data_a[index] + data_b[index];
        // }
        //
        // Implementation steps:
        // 1. Create Vulkan instance and select physical device
        // 2. Create logical device and compute queue
        // 3. Compile GLSL to SPIR-V bytecode
        // 4. Create compute shader module from SPIR-V
        // 5. Create descriptor set layout for buffers
        // 6. Create compute pipeline with shader module
        // 7. Allocate device memory for input/output buffers
        // 8. Create buffer objects and bind memory
        // 9. Create descriptor pool and allocate descriptor sets
        // 10. Update descriptor sets with buffer bindings
        // 11. Create command pool and allocate command buffer
        // 12. Record commands: bind pipeline, bind descriptor sets, dispatch
        // 13. Submit command buffer to compute queue
        // 14. Wait for completion and read results
        return TensorError.UnsupportedDevice;
    }

    fn vectorAddWebGpu(a: []const f32, b: []const f32, result: []f32) TensorError!void {
        _ = a;
        _ = b;
        _ = result;
        // TODO: WebGPU compute shader implementation
        // WGSL shader source for vector addition:
        // @compute @workgroup_size(64)
        // fn main(@builtin(global_invocation_id) global_id : vec3<u32>) {
        //     let index = global_id.x;
        //     if (index >= arrayLength(&result)) { return; }
        //     result[index] = input_a[index] + input_b[index];
        // }
        //
        // Implementation steps:
        // 1. Create WebGPU device and queue
        // 2. Create compute shader module from WGSL source
        // 3. Create buffer resources for input/output data
        // 4. Create bind group with buffers
        // 5. Create compute pipeline with shader and bind group layout
        // 6. Begin command encoder, set pipeline and bind group
        // 7. Dispatch workgroups: (array_length + workgroup_size - 1) / workgroup_size
        // 8. Copy buffer to buffer for reading
        // 9. Submit command buffer and wait for completion
        // 10. Map and read result buffer
        return TensorError.UnsupportedDevice;
    }

    fn vectorAddMetal(a: []const f32, b: []const f32, result: []f32) TensorError!void {
        _ = a;
        _ = b;
        _ = result;
        // Metal compute shader implementation
        // Metal Shading Language (MSL) compute shader for vector addition:
        // #include <metal_stdlib>
        // using namespace metal;
        //
        // kernel void vectorAdd(device const float* inputA [[buffer(0)]],
        //                       device const float* inputB [[buffer(1)]],
        //                       device float* output [[buffer(2)]],
        //                       uint index [[thread_position_in_grid]],
        //                       uint arrayLength [[threads_per_grid]]) {
        //     if (index >= arrayLength) return;
        //     output[index] = inputA[index] + inputB[index];
        // }
        //
        // Implementation steps:
        // 1. Get default Metal device (MTLCreateSystemDefaultDevice)
        // 2. Create command queue from device
        // 3. Compile MSL source to create shader library
        // 4. Create compute pipeline state with kernel function
        // 5. Create Metal buffers for input and output data
        // 6. Copy input data to Metal buffers
        // 7. Create command buffer from command queue
        // 8. Create compute command encoder
        // 9. Set compute pipeline state and buffers
        // 10. Calculate threadgroup sizes and dispatch threads
        // 11. End encoding and commit command buffer
        // 12. Wait for completion and read results
        return TensorError.UnsupportedDevice;
    }

    pub fn vectorMul(a: []const f32, b: []const f32, result: []f32, device: *const Device) TensorError!void {
        return vectorMulWithFallback(a, b, result, device) catch |err| {
            switch (err) {
                TensorError.GpuUnavailable, TensorError.UnsupportedDevice => {
                    return vectorMulCpu(a, b, result);
                },
                else => return err,
            }
        };
    }

    fn vectorMulWithFallback(a: []const f32, b: []const f32, result: []f32, device: *const Device) TensorError!void {
        if (a.len != b.len or a.len != result.len) return TensorError.ShapeMismatch;

        if (!gpu_enabled) {
            if (device.isGpu()) return TensorError.GpuUnavailable;
            return vectorMulCpu(a, b, result);
        }

        switch (device.device_type) {
            .cpu => return vectorMulCpu(a, b, result),
            .cuda => return vectorMulCuda(a, b, result),
            .vulkan => return vectorMulVulkan(a, b, result),
            .webgpu => return vectorMulWebGpu(a, b, result),
            .metal => return vectorMulMetal(a, b, result),
        }
    }

    fn vectorMulCpu(a: []const f32, b: []const f32, result: []f32) TensorError!void {
        const Vec4 = @Vector(4, f32);
        const simd_len = (a.len / 4) * 4;

        var i: usize = 0;
        while (i < simd_len) : (i += 4) {
            const a_vec: Vec4 = a[i..i+4][0..4].*;
            const b_vec: Vec4 = b[i..i+4][0..4].*;
            const result_vec = a_vec * b_vec;
            result[i..i+4][0..4].* = result_vec;
        }

        while (i < a.len) : (i += 1) {
            result[i] = a[i] * b[i];
        }
    }

    fn vectorMulCuda(a: []const f32, b: []const f32, result: []f32) TensorError!void {
        _ = a;
        _ = b;
        _ = result;
        return TensorError.UnsupportedDevice;
    }

    fn vectorMulVulkan(a: []const f32, b: []const f32, result: []f32) TensorError!void {
        _ = a;
        _ = b;
        _ = result;
        return TensorError.UnsupportedDevice;
    }

    fn vectorMulWebGpu(a: []const f32, b: []const f32, result: []f32) TensorError!void {
        _ = a;
        _ = b;
        _ = result;
        return TensorError.UnsupportedDevice;
    }

    fn vectorMulMetal(a: []const f32, b: []const f32, result: []f32) TensorError!void {
        _ = a;
        _ = b;
        _ = result;
        return TensorError.UnsupportedDevice;
    }

    pub fn matrixMul(a: []const f32, b: []const f32, result: []f32,
                    m: usize, k: usize, n: usize, device: *const Device) TensorError!void {
        return matrixMulWithFallback(a, b, result, m, k, n, device) catch |err| {
            switch (err) {
                TensorError.GpuUnavailable, TensorError.UnsupportedDevice => {
                    return matrixMulCpu(a, b, result, m, k, n);
                },
                else => return err,
            }
        };
    }

    fn matrixMulWithFallback(a: []const f32, b: []const f32, result: []f32,
                           m: usize, k: usize, n: usize, device: *const Device) TensorError!void {
        if (a.len != m * k or b.len != k * n or result.len != m * n) {
            return TensorError.ShapeMismatch;
        }

        if (!gpu_enabled) {
            if (device.isGpu()) return TensorError.GpuUnavailable;
            return matrixMulCpu(a, b, result, m, k, n);
        }

        switch (device.device_type) {
            .cpu => return matrixMulCpu(a, b, result, m, k, n),
            .cuda => return matrixMulCuda(a, b, result, m, k, n),
            .vulkan => return matrixMulVulkan(a, b, result, m, k, n),
            .webgpu => return matrixMulWebGpu(a, b, result, m, k, n),
            .metal => return matrixMulMetal(a, b, result, m, k, n),
        }
    }

    fn matrixMulCpu(a: []const f32, b: []const f32, result: []f32,
                   m: usize, k: usize, n: usize) TensorError!void {
        // CPU implementation with loop tiling for cache efficiency
        const block_size = 64;

        for (0..m) |i| {
            for (0..n) |j| {
                var acc: f32 = 0.0;
                var kk: usize = 0;
                while (kk < k) : (kk += block_size) {
                    const k_end = @min(kk + block_size, k);
                    for (kk..k_end) |p| {
                        acc += a[i * k + p] * b[p * n + j];
                    }
                }
                result[i * n + j] = acc;
            }
        }
    }

    fn matrixMulCuda(a: []const f32, b: []const f32, result: []f32,
                    m: usize, k: usize, n: usize) TensorError!void {
        _ = a;
        _ = b;
        _ = result;
        _ = m;
        _ = k;
        _ = n;
        // TODO: Implement cuBLAS SGEMM call:
        // 1. Initialize CUDA context and cuBLAS handle
        // 2. Allocate GPU memory for matrices A, B, and C
        // 3. Copy input matrices to GPU
        // 4. Call cublasSgemm() with optimal configuration
        // 5. Copy result back to host
        // 6. Clean up GPU memory
        return TensorError.UnsupportedDevice;
    }

    fn matrixMulVulkan(a: []const f32, b: []const f32, result: []f32,
                      m: usize, k: usize, n: usize) TensorError!void {
        _ = a; _ = b; _ = result; _ = m; _ = k; _ = n;
        return TensorError.UnsupportedDevice;
    }

    fn matrixMulWebGpu(a: []const f32, b: []const f32, result: []f32,
                      m: usize, k: usize, n: usize) TensorError!void {
        _ = a; _ = b; _ = result; _ = m; _ = k; _ = n;
        return TensorError.UnsupportedDevice;
    }

    fn matrixMulMetal(a: []const f32, b: []const f32, result: []f32,
                     m: usize, k: usize, n: usize) TensorError!void {
        _ = a; _ = b; _ = result; _ = m; _ = k; _ = n;
        return TensorError.UnsupportedDevice;
    }
};

pub const Tensor = struct {
    data: []f32,
    shape: []const usize,
    allocator: std.mem.Allocator,
    pool: ?*TensorPool,
    ref_count: *u32,
    device: ?*const Device,
    gpu_memory: ?GpuMemory,

    fn validateShape(shape: []const usize) TensorError!void {
        if (shape.len == 0) {
            const err_ctx = tensorError(
                TensorError.InvalidShape,
                "Tensor shape cannot be empty - must have at least one dimension",
                "tensor initialization",
                null,
                shape,
                @src(),
            );
            std.log.err("{any}", .{err_ctx});
            return TensorError.InvalidShape;
        }
        for (shape, 0..) |dim, i| {
            if (dim == 0) {
                const err_ctx = tensorError(
                    TensorError.InvalidDimension,
                    std.fmt.allocPrint(std.heap.page_allocator, "Dimension {d} has zero size", .{i}) catch "Dimension has zero size",
                    "tensor initialization",
                    null,
                    shape,
                    @src(),
                );
                std.log.err("{any}", .{err_ctx});
                return TensorError.InvalidDimension;
            }
        }
    }

    fn computeSize(shape: []const usize) TensorError!usize {
        try validateShape(shape);
        var size: usize = 1;
        for (shape) |dim| {
            const result = @mulWithOverflow(size, dim);
            if (result[1] != 0) return TensorError.OutOfMemory;
            size = result[0];
        }
        return size;
    }

    pub fn init(allocator: std.mem.Allocator, shape: []const usize) TensorError!Tensor {
        const size = try computeSize(shape);
        const data = allocator.alloc(f32, size) catch return TensorError.OutOfMemory;
        @memset(data, 0.0);
        const ref_count = allocator.create(u32) catch {
            allocator.free(data);
            return TensorError.OutOfMemory;
        };
        ref_count.* = 1;
        return Tensor{
            .data = data,
            .shape = allocator.dupe(usize, shape) catch {
                allocator.free(data);
                allocator.destroy(ref_count);
                return TensorError.OutOfMemory;
            },
            .allocator = allocator,
            .pool = null,
            .ref_count = ref_count,
            .device = null,
            .gpu_memory = null,
        };
    }

    pub fn initWithPool(allocator: std.mem.Allocator, shape: []const usize, pool: *TensorPool) TensorError!Tensor {
        const size = try computeSize(shape);
        const data = pool.acquire(size) catch return TensorError.PoolError;
        @memset(data, 0.0);
        const ref_count = allocator.create(u32) catch {
            pool.release(data) catch {};
            return TensorError.OutOfMemory;
        };
        ref_count.* = 1;
        return Tensor{
            .data = data,
            .shape = allocator.dupe(usize, shape) catch {
                pool.release(data) catch {};
                allocator.destroy(ref_count);
                return TensorError.OutOfMemory;
            },
            .allocator = allocator,
            .pool = pool,
            .ref_count = ref_count,
            .device = null,
            .gpu_memory = null,
        };
    }

    pub fn initOnDevice(allocator: std.mem.Allocator, shape: []const usize, device: *const Device) TensorError!Tensor {
        const size = try computeSize(shape);
        const data = allocator.alloc(f32, size) catch return TensorError.OutOfMemory;
        @memset(data, 0.0);

        const ref_count = allocator.create(u32) catch {
            allocator.free(data);
            return TensorError.OutOfMemory;
        };
        ref_count.* = 1;

        var gpu_memory: ?GpuMemory = null;
        if (device.isGpu()) {
            gpu_memory = GpuMemory.init(allocator, device, size * @sizeOf(f32)) catch {
                allocator.free(data);
                allocator.destroy(ref_count);
                return TensorError.DeviceNotAvailable;
            };
        }

        return Tensor{
            .data = data,
            .shape = allocator.dupe(usize, shape) catch {
                allocator.free(data);
                allocator.destroy(ref_count);
                if (gpu_memory) |*mem| mem.deinit();
                return TensorError.OutOfMemory;
            },
            .allocator = allocator,
            .pool = null,
            .ref_count = ref_count,
            .device = device,
            .gpu_memory = gpu_memory,
        };
    }

    pub fn clone(self: *const Tensor) Tensor {
        self.ref_count.* += 1;
        return Tensor{
            .data = self.data,
            .shape = self.shape,
            .allocator = self.allocator,
            .pool = self.pool,
            .ref_count = self.ref_count,
            .device = self.device,
            .gpu_memory = self.gpu_memory,
        };
    }

    pub fn deinit(self: *Tensor) void {
        self.ref_count.* -= 1;
        if (self.ref_count.* == 0) {
            if (self.gpu_memory) |*gpu_mem| {
                gpu_mem.deinit();
            }
            if (self.pool) |pool| {
                pool.release(self.data) catch {};
            } else {
                self.allocator.free(self.data);
            }
            self.allocator.free(self.shape);
            self.allocator.destroy(self.ref_count);
        }
    }

    pub fn add(self: *const Tensor, other: *const Tensor) TensorError!Tensor {
        const bs = broadcastShape(self.shape, other.shape) orelse {
            const err_ctx = shapeError(self.shape, other.shape, "tensor addition", @src());
            std.log.err("{any}", .{err_ctx});
            return TensorError.IncompatibleShapes;
        };
        const new_shape = bs.shape[0..bs.len];
        var result = if (self.pool) |pool|
            try Tensor.initWithPool(self.allocator, new_shape, pool)
        else
            try Tensor.init(self.allocator, new_shape);
        for (0..result.data.len) |i| {
            const a_idx = broadcastIndex(i, self.shape, new_shape);
            const b_idx = broadcastIndex(i, other.shape, new_shape);
            result.data[i] = self.data[a_idx] + other.data[b_idx];
        }
        return result;
    }

    pub fn addGpu(self: *const Tensor, other: *const Tensor) TensorError!Tensor {
        if (!std.mem.eql(usize, self.shape, other.shape)) {
            const err_ctx = shapeError(self.shape, other.shape, "GPU tensor addition", @src());
            std.log.err("{any}", .{err_ctx});
            return TensorError.ShapeMismatch;
        }

        const device = self.device orelse return TensorError.DeviceNotAvailable;
        const result = if (self.device) |dev|
            try Tensor.initOnDevice(self.allocator, self.shape, dev)
        else
            try Tensor.init(self.allocator, self.shape);

        try GpuKernel.vectorAdd(self.data, other.data, result.data, device);
        return result;
    }

    pub fn mulGpu(self: *const Tensor, other: *const Tensor) TensorError!Tensor {
        if (!std.mem.eql(usize, self.shape, other.shape)) return TensorError.ShapeMismatch;

        const device = self.device orelse return TensorError.DeviceNotAvailable;
        const result = if (self.device) |dev|
            try Tensor.initOnDevice(self.allocator, self.shape, dev)
        else
            try Tensor.init(self.allocator, self.shape);

        try GpuKernel.vectorMul(self.data, other.data, result.data, device);
        return result;
    }

    pub fn matmulGpu(self: *const Tensor, other: *const Tensor) TensorError!Tensor {
        if (self.shape.len != 2 or other.shape.len != 2) return TensorError.Not2D;
        if (self.shape[1] != other.shape[0]) return TensorError.ShapeMismatch;

        const device = self.device orelse return TensorError.DeviceNotAvailable;
        const m = self.shape[0];
        const k = self.shape[1];
        const n = other.shape[1];

        const result = if (self.device) |dev|
            try Tensor.initOnDevice(self.allocator, &[_]usize{ m, n }, dev)
        else
            try Tensor.init(self.allocator, &[_]usize{ m, n });

        try GpuKernel.matrixMul(self.data, other.data, result.data, m, k, n, device);
        return result;
    }

    pub fn toDevice(self: *const Tensor, device: *const Device) TensorError!Tensor {
        if (self.device == device) {
            return self.clone();
        }

        var result = try Tensor.initOnDevice(self.allocator, self.shape, device);
        @memcpy(result.data, self.data);

        if (device.isGpu() and result.gpu_memory != null) {
            const data_bytes = std.mem.sliceAsBytes(self.data);
            try result.gpu_memory.?.copyFromHost(data_bytes);
        }

        return result;
    }

    pub fn syncFromGpu(self: *Tensor) TensorError!void {
        if (self.gpu_memory) |*gpu_mem| {
            const data_bytes = std.mem.sliceAsBytes(self.data);
            try gpu_mem.copyToHost(data_bytes);
        }
    }

    pub fn mul(self: *const Tensor, other: *const Tensor) TensorError!Tensor {
        const bs = broadcastShape(self.shape, other.shape) orelse return TensorError.IncompatibleShapes;
        const new_shape = bs.shape[0..bs.len];
        var result = if (self.pool) |pool|
            try Tensor.initWithPool(self.allocator, new_shape, pool)
        else
            try Tensor.init(self.allocator, new_shape);
        for (0..result.data.len) |i| {
            const a_idx = broadcastIndex(i, self.shape, new_shape);
            const b_idx = broadcastIndex(i, other.shape, new_shape);
            result.data[i] = self.data[a_idx] * other.data[b_idx];
        }
        return result;
    }

    pub fn sub(self: *const Tensor, other: *const Tensor) TensorError!Tensor {
        const bs = broadcastShape(self.shape, other.shape) orelse return TensorError.IncompatibleShapes;
        const new_shape = bs.shape[0..bs.len];
        var result = if (self.pool) |pool|
            try Tensor.initWithPool(self.allocator, new_shape, pool)
        else
            try Tensor.init(self.allocator, new_shape);
        for (0..result.data.len) |i| {
            const a_idx = broadcastIndex(i, self.shape, new_shape);
            const b_idx = broadcastIndex(i, other.shape, new_shape);
            result.data[i] = self.data[a_idx] - other.data[b_idx];
        }
        return result;
    }

    pub fn square(self: *const Tensor) TensorError!Tensor {
        var result = if (self.pool) |pool|
            try Tensor.initWithPool(self.allocator, self.shape, pool)
        else
            try Tensor.init(self.allocator, self.shape);
        for (0..self.data.len) |i| {
            result.data[i] = self.data[i] * self.data[i];
        }
        return result;
    }

    pub fn sum(self: *const Tensor) f32 {
        var total: f32 = 0.0;
        for (self.data) |val| {
            total += val;
        }
        return total;
    }

    pub fn transpose(self: *const Tensor) TensorError!Tensor {
        if (self.shape.len != 2) return TensorError.Not2D;
        const rows = self.shape[0];
        const cols = self.shape[1];
        var result = if (self.pool) |pool|
            try Tensor.initWithPool(self.allocator, &[_]usize{ cols, rows }, pool)
        else
            try Tensor.init(self.allocator, &[_]usize{ cols, rows });
        for (0..rows) |i| {
            for (0..cols) |j| {
                result.data[j * rows + i] = self.data[i * cols + j];
            }
        }
        return result;
    }

    pub fn matmul(self: *const Tensor, other: *const Tensor) TensorError!Tensor {
        if (self.shape.len != 2 or other.shape.len != 2) return TensorError.Not2D;
        if (self.shape[1] != other.shape[0]) return TensorError.ShapeMismatch;
        const m = self.shape[0];
        const k = self.shape[1];
        const n = other.shape[1];
        var result = if (self.pool) |pool|
            try Tensor.initWithPool(self.allocator, &[_]usize{ m, n }, pool)
        else
            try Tensor.init(self.allocator, &[_]usize{ m, n });

        for (0..m) |i| {
            for (0..n) |j| {
                var acc: f32 = 0.0;
                for (0..k) |p| {
                    acc += self.data[i * k + p] * other.data[p * n + j];
                }
                result.data[i * n + j] = acc;
            }
        }
        return result;
    }

    pub fn randomize(self: *Tensor) void {
        for (self.data) |*val| {
            val.* = 0.1; // placeholder
        }
    }

    pub fn addSimd(self: *const Tensor, other: *const Tensor) TensorError!Tensor {
        if (!std.mem.eql(usize, self.shape, other.shape)) return TensorError.ShapeMismatch;

        var result = if (self.pool) |pool|
            try Tensor.initWithPool(self.allocator, self.shape, pool)
        else
            try Tensor.init(self.allocator, self.shape);

        const Vec4 = @Vector(4, f32);
        const simd_len = (self.data.len / 4) * 4;

        var i: usize = 0;
        while (i < simd_len) : (i += 4) {
            const a_vec: Vec4 = self.data[i..i+4][0..4].*;
            const b_vec: Vec4 = other.data[i..i+4][0..4].*;
            const result_vec = a_vec + b_vec;
            result.data[i..i+4][0..4].* = result_vec;
        }

        while (i < self.data.len) : (i += 1) {
            result.data[i] = self.data[i] + other.data[i];
        }

        return result;
    }

    pub fn mulSimd(self: *const Tensor, other: *const Tensor) TensorError!Tensor {
        if (!std.mem.eql(usize, self.shape, other.shape)) return TensorError.ShapeMismatch;

        var result = if (self.pool) |pool|
            try Tensor.initWithPool(self.allocator, self.shape, pool)
        else
            try Tensor.init(self.allocator, self.shape);

        const Vec4 = @Vector(4, f32);
        const simd_len = (self.data.len / 4) * 4;

        var i: usize = 0;
        while (i < simd_len) : (i += 4) {
            const a_vec: Vec4 = self.data[i..i+4][0..4].*;
            const b_vec: Vec4 = other.data[i..i+4][0..4].*;
            const result_vec = a_vec * b_vec;
            result.data[i..i+4][0..4].* = result_vec;
        }

        while (i < self.data.len) : (i += 1) {
            result.data[i] = self.data[i] * other.data[i];
        }

        return result;
    }

    pub fn squareSimd(self: *const Tensor) !Tensor {
        var result = if (self.pool) |pool|
            try Tensor.initWithPool(self.allocator, self.shape, pool)
        else
            try Tensor.init(self.allocator, self.shape);

        const Vec4 = @Vector(4, f32);
        const simd_len = (self.data.len / 4) * 4;

        var i: usize = 0;
        while (i < simd_len) : (i += 4) {
            const vec: Vec4 = self.data[i..i+4][0..4].*;
            const result_vec = vec * vec;
            result.data[i..i+4][0..4].* = result_vec;
        }

        while (i < self.data.len) : (i += 1) {
            result.data[i] = self.data[i] * self.data[i];
        }

        return result;
    }

    fn broadcastShape(a_shape: []const usize, b_shape: []const usize) ?struct { shape: [10]usize, len: usize } {
        const len = @max(a_shape.len, b_shape.len);
        if (len > 10) return null;
        var result: [10]usize = undefined;
        for (0..len) |i| {
            const a_dim = if (i < a_shape.len) a_shape[a_shape.len - 1 - i] else 1;
            const b_dim = if (i < b_shape.len) b_shape[b_shape.len - 1 - i] else 1;
            if (a_dim != 1 and b_dim != 1 and a_dim != b_dim) return null;
            result[len - 1 - i] = @max(a_dim, b_dim);
        }
        return .{ .shape = result, .len = len };
    }

    fn broadcastIndex(idx: usize, orig_shape: []const usize, new_shape: []const usize) usize {
        var coords: [10]usize = undefined;
        var temp = idx;
        for (0..new_shape.len) |i| {
            const dim_idx = new_shape.len - 1 - i;
            coords[dim_idx] = temp % new_shape[dim_idx];
            temp /= new_shape[dim_idx];
        }
        var orig_idx: usize = 0;
        var stride: usize = 1;
        for (0..orig_shape.len) |i| {
            const dim_idx = orig_shape.len - 1 - i;
            const coord = if (orig_shape[dim_idx] == 1) 0 else coords[dim_idx];
            orig_idx += coord * stride;
            stride *= orig_shape[dim_idx];
        }
        return orig_idx;
    }

    // Fluent API - chainable operations that modify self in-place
    pub fn add_(self: *Tensor, other: *const Tensor) TensorError!*Tensor {
        const result = try self.add(other);
        defer result.deinit();
        @memcpy(self.data, result.data);
        return self;
    }

    pub fn mul_(self: *Tensor, other: *const Tensor) TensorError!*Tensor {
        const result = try self.mul(other);
        defer result.deinit();
        @memcpy(self.data, result.data);
        return self;
    }

    pub fn sub_(self: *Tensor, other: *const Tensor) TensorError!*Tensor {
        const result = try self.sub(other);
        defer result.deinit();
        @memcpy(self.data, result.data);
        return self;
    }

    pub fn relu_(self: *Tensor) TensorError!*Tensor {
        for (self.data) |*val| {
            val.* = @max(0.0, val.*);
        }
        return self;
    }

    pub fn sigmoid_(self: *Tensor) TensorError!*Tensor {
        for (self.data) |*val| {
            val.* = 1.0 / (1.0 + std.math.exp(-val.*));
        }
        return self;
    }

    pub fn tanh_(self: *Tensor) TensorError!*Tensor {
        for (self.data) |*val| {
            val.* = std.math.tanh(val.*);
        }
        return self;
    }

    pub fn scale_(self: *Tensor, factor: f32) TensorError!*Tensor {
        for (self.data) |*val| {
            val.* *= factor;
        }
        return self;
    }

    pub fn clamp_(self: *Tensor, min_val: f32, max_val: f32) TensorError!*Tensor {
        for (self.data) |*val| {
            val.* = std.math.clamp(val.*, min_val, max_val);
        }
        return self;
    }

    pub fn fill_(self: *Tensor, value: f32) TensorError!*Tensor {
        @memset(self.data, value);
        return self;
    }

    pub fn zero_(self: *Tensor) TensorError!*Tensor {
        @memset(self.data, 0.0);
        return self;
    }

    pub fn ones_(self: *Tensor) TensorError!*Tensor {
        for (self.data) |*val| {
            val.* = 1.0;
        }
        return self;
    }
};

// TensorBuilder for fluent tensor creation
pub const TensorBuilder = struct {
    allocator: std.mem.Allocator,
    shape: ?[]const usize,
    pool: ?*TensorPool,
    device: ?*const Device,
    fill_value: ?f32,
    random_seed: ?u64,

    pub fn init(allocator: std.mem.Allocator) TensorBuilder {
        return TensorBuilder{
            .allocator = allocator,
            .shape = null,
            .pool = null,
            .device = null,
            .fill_value = null,
            .random_seed = null,
        };
    }

    pub fn withShape(self: TensorBuilder, shape: []const usize) TensorBuilder {
        return TensorBuilder{
            .allocator = self.allocator,
            .shape = shape,
            .pool = self.pool,
            .device = self.device,
            .fill_value = self.fill_value,
            .random_seed = self.random_seed,
        };
    }

    pub fn withPool(self: TensorBuilder, pool: *TensorPool) TensorBuilder {
        return TensorBuilder{
            .allocator = self.allocator,
            .shape = self.shape,
            .pool = pool,
            .device = self.device,
            .fill_value = self.fill_value,
            .random_seed = self.random_seed,
        };
    }

    pub fn onDevice(self: TensorBuilder, device: *const Device) TensorBuilder {
        return TensorBuilder{
            .allocator = self.allocator,
            .shape = self.shape,
            .pool = self.pool,
            .device = device,
            .fill_value = self.fill_value,
            .random_seed = self.random_seed,
        };
    }

    pub fn filled(self: TensorBuilder, value: f32) TensorBuilder {
        return TensorBuilder{
            .allocator = self.allocator,
            .shape = self.shape,
            .pool = self.pool,
            .device = self.device,
            .fill_value = value,
            .random_seed = self.random_seed,
        };
    }

    pub fn zeros(self: TensorBuilder) TensorBuilder {
        return self.filled(0.0);
    }

    pub fn ones(self: TensorBuilder) TensorBuilder {
        return self.filled(1.0);
    }

    pub fn random(self: TensorBuilder, seed: u64) TensorBuilder {
        return TensorBuilder{
            .allocator = self.allocator,
            .shape = self.shape,
            .pool = self.pool,
            .device = self.device,
            .fill_value = self.fill_value,
            .random_seed = seed,
        };
    }

    pub fn build(self: TensorBuilder) TensorError!Tensor {
        const shape = self.shape orelse return TensorError.InvalidShape;

        const tensor = if (self.device) |device|
            try Tensor.initOnDevice(self.allocator, shape, device)
        else if (self.pool) |pool|
            try Tensor.initWithPool(self.allocator, shape, pool)
        else
            try Tensor.init(self.allocator, shape);

        if (self.fill_value) |value| {
            @memset(tensor.data, value);
        } else if (self.random_seed) |seed| {
            var prng = std.Random.DefaultPrng.init(seed);
            const rng = prng.random();
            for (tensor.data) |*val| {
                val.* = rng.float(f32) * 2.0 - 1.0; // Random values between -1 and 1
            }
        }

        return tensor;
    }
};

pub const ActivationFn = *const fn (tensor: *const Tensor) TensorError!Tensor;

// Type-safe tensor operations with compile-time shape verification
pub fn TypedTensor(comptime shape: []const usize) type {
    return struct {
        const Self = @This();

        tensor: Tensor,

        pub fn init(allocator: std.mem.Allocator) TensorError!Self {
            return Self{
                .tensor = try Tensor.init(allocator, shape),
            };
        }

        pub fn initWithPool(allocator: std.mem.Allocator, pool: *TensorPool) TensorError!Self {
            return Self{
                .tensor = try Tensor.initWithPool(allocator, shape, pool),
            };
        }

        pub fn initOnDevice(allocator: std.mem.Allocator, device: *const Device) TensorError!Self {
            return Self{
                .tensor = try Tensor.initOnDevice(allocator, shape, device),
            };
        }

        pub fn deinit(self: *Self) void {
            self.tensor.deinit();
        }

        pub fn fill(self: *Self, value: f32) void {
            for (self.tensor.data) |*val| {
                val.* = value;
            }
        }

        pub fn zero(self: *Self) void {
            self.fill(0.0);
        }

        pub fn ones(self: *Self) void {
            self.fill(1.0);
        }

        // Type-safe addition - only allows compatible shapes
        pub fn add(self: *const Self, other: *const TypedTensor(shape)) TensorError!Self {
            const result_tensor = try self.tensor.add(&other.tensor);
            return Self{
                .tensor = result_tensor,
            };
        }

        // Type-safe multiplication - only allows compatible shapes
        pub fn mul(self: *const Self, other: *const TypedTensor(shape)) TensorError!Self {
            const result_tensor = try self.tensor.mul(&other.tensor);
            return Self{
                .tensor = result_tensor,
            };
        }

        // Access tensor data with bounds checking
        pub fn get(self: *const Self, indices: [shape.len]usize) TensorError!f32 {
            comptime var stride: usize = 1;
            var index: usize = 0;

            inline for (0..shape.len) |i| {
                const dim_idx = shape.len - 1 - i;
                if (indices[dim_idx] >= shape[dim_idx]) {
                    return TensorError.InvalidIndex;
                }
                index += indices[dim_idx] * stride;
                stride *= shape[dim_idx];
            }

            return self.tensor.data[index];
        }

        pub fn set(self: *Self, indices: [shape.len]usize, value: f32) TensorError!void {
            comptime var stride: usize = 1;
            var index: usize = 0;

            inline for (0..shape.len) |i| {
                const dim_idx = shape.len - 1 - i;
                if (indices[dim_idx] >= shape[dim_idx]) {
                    return TensorError.InvalidIndex;
                }
                index += indices[dim_idx] * stride;
                stride *= shape[dim_idx];
            }

            self.tensor.data[index] = value;
        }

        // Convert to untyped tensor
        pub fn untyped(self: *const Self) *const Tensor {
            return &self.tensor;
        }
    };
}

// Compile-time helper functions
fn computeStaticSize(comptime shape: []const usize) usize {
    var size: usize = 1;
    for (shape) |dim| {
        size *= dim;
    }
    return size;
}

// Debug mode configuration
pub const DebugConfig = struct {
    enable_shape_checking: bool = false,
    enable_bounds_checking: bool = false,
    enable_operation_logging: bool = false,
    enable_memory_tracking: bool = false,
    verbose_errors: bool = false,

    pub const default = DebugConfig{};
    pub const development = DebugConfig{
        .enable_shape_checking = true,
        .enable_bounds_checking = true,
        .enable_operation_logging = true,
        .enable_memory_tracking = true,
        .verbose_errors = true,
    };
};

var debug_config: DebugConfig = DebugConfig.default;

pub fn setDebugMode(config: DebugConfig) void {
    debug_config = config;
}

pub fn getDebugMode() DebugConfig {
    return debug_config;
}

// Debug-aware tensor wrapper
pub const DebugTensor = struct {
    tensor: Tensor,
    creation_location: std.builtin.SourceLocation,
    operation_count: u32 = 0,
    last_operation: ?[]const u8 = null,

    pub fn init(allocator: std.mem.Allocator, shape: []const usize, src: std.builtin.SourceLocation) TensorError!DebugTensor {
        if (debug_config.enable_shape_checking) {
            for (shape, 0..) |dim, i| {
                if (dim == 0) {
                    const err_ctx = tensorError(
                        TensorError.InvalidDimension,
                        std.fmt.allocPrint(allocator, "DEBUG: Dimension {d} has zero size", .{i}) catch "DEBUG: Zero dimension detected",
                        "debug tensor creation",
                        null,
                        shape,
                        src,
                    );
                    std.log.err("{any}", .{err_ctx});
                    return TensorError.InvalidDimension;
                }
            }
        }

        if (debug_config.enable_memory_tracking) {
            std.log.info("DEBUG: Creating tensor at {s}:{d}", .{src.file, src.line});
        }

        return DebugTensor{
            .tensor = try Tensor.init(allocator, shape),
            .creation_location = src,
        };
    }

    pub fn deinit(self: *DebugTensor) void {
        if (debug_config.enable_memory_tracking) {
            std.log.info("DEBUG: Destroying tensor created at {s}:{d} after {d} operations", .{
                self.creation_location.file, self.creation_location.line, self.operation_count
            });
        }
        self.tensor.deinit();
    }

    pub fn debugAdd(self: *DebugTensor, other: *const DebugTensor, src: std.builtin.SourceLocation) TensorError!DebugTensor {
        if (debug_config.enable_shape_checking) {
            if (!std.mem.eql(usize, self.tensor.shape, other.tensor.shape)) {
                const err_ctx = shapeError(self.tensor.shape, other.tensor.shape, "debug tensor addition", src);
                std.log.err("{any}", .{err_ctx});
                return TensorError.ShapeMismatch;
            }
        }

        if (debug_config.enable_operation_logging) {
            std.log.info("DEBUG: Adding tensors at {s}:{d}", .{src.file, src.line});
        }

        const result_tensor = try self.tensor.add(&other.tensor);
        self.operation_count += 1;
        self.last_operation = "add";

        return DebugTensor{
            .tensor = result_tensor,
            .creation_location = src,
        };
    }

    pub fn debugGet(self: *const DebugTensor, index: usize, src: std.builtin.SourceLocation) TensorError!f32 {
        if (debug_config.enable_bounds_checking) {
            if (index >= self.tensor.data.len) {
                const err_ctx = tensorError(
                    TensorError.InvalidIndex,
                    std.fmt.allocPrint(self.tensor.allocator, "DEBUG: Index {d} out of bounds (size: {d})", .{index, self.tensor.data.len}) catch "DEBUG: Index out of bounds",
                    "debug tensor access",
                    null,
                    null,
                    src,
                );
                std.log.err("{any}", .{err_ctx});
                return TensorError.InvalidIndex;
            }
        }

        if (debug_config.enable_operation_logging) {
            std.log.info("DEBUG: Accessing tensor[{d}] = {d:.6} at {s}:{d}", .{
                index, self.tensor.data[index], src.file, src.line
            });
        }

        return self.tensor.data[index];
    }

    pub fn debugSet(self: *DebugTensor, index: usize, value: f32, src: std.builtin.SourceLocation) TensorError!void {
        if (debug_config.enable_bounds_checking) {
            if (index >= self.tensor.data.len) {
                const err_ctx = tensorError(
                    TensorError.InvalidIndex,
                    std.fmt.allocPrint(self.tensor.allocator, "DEBUG: Index {d} out of bounds (size: {d})", .{index, self.tensor.data.len}) catch "DEBUG: Index out of bounds",
                    "debug tensor assignment",
                    null,
                    null,
                    src,
                );
                std.log.err("{any}", .{err_ctx});
                return TensorError.InvalidIndex;
            }
        }

        if (debug_config.enable_operation_logging) {
            std.log.info("DEBUG: Setting tensor[{d}] = {d:.6} -> {d:.6} at {s}:{d}", .{
                index, self.tensor.data[index], value, src.file, src.line
            });
        }

        self.tensor.data[index] = value;
        self.operation_count += 1;
        self.last_operation = "set";
    }

    pub fn getStats(self: *const DebugTensor) struct { operations: u32, last_op: ?[]const u8, created_at: std.builtin.SourceLocation } {
        return .{
            .operations = self.operation_count,
            .last_op = self.last_operation,
            .created_at = self.creation_location,
        };
    }
};

pub const LayerType = enum {
    linear,
    conv1d,
    maxpool1d,
    dropout,
    batchnorm1d,
    activation,
};

pub const Layer = union(LayerType) {
    linear: *Linear,
    conv1d: *Conv1D,
    maxpool1d: *MaxPool1D,
    dropout: *Dropout,
    batchnorm1d: *BatchNorm1D,
    activation: ActivationFn,
};

pub const Sequential = struct {
    layers: std.ArrayList(Layer),
    allocator: std.mem.Allocator,
    training: bool,

    pub fn init(allocator: std.mem.Allocator) Sequential {
        return Sequential{
            .layers = std.ArrayList(Layer){},
            .allocator = allocator,
            .training = true,
        };
    }

    pub fn deinit(self: *Sequential) void {
        for (self.layers.items) |layer| {
            switch (layer) {
                .linear => |l| l.deinit(),
                .conv1d => |c| c.deinit(),
                .maxpool1d => {},
                .dropout => {},
                .batchnorm1d => |b| b.deinit(),
                .activation => {},
            }
        }
        self.layers.deinit(self.allocator);
    }

    pub fn addLinear(self: *Sequential, layer: *Linear) !void {
        try self.layers.append(self.allocator, Layer{ .linear = layer });
    }

    pub fn addConv1D(self: *Sequential, layer: *Conv1D) !void {
        try self.layers.append(self.allocator, Layer{ .conv1d = layer });
    }

    pub fn addMaxPool1D(self: *Sequential, layer: *MaxPool1D) !void {
        try self.layers.append(self.allocator, Layer{ .maxpool1d = layer });
    }

    pub fn addDropout(self: *Sequential, layer: *Dropout) !void {
        try self.layers.append(self.allocator, Layer{ .dropout = layer });
    }

    pub fn addBatchNorm1D(self: *Sequential, layer: *BatchNorm1D) !void {
        try self.layers.append(self.allocator, Layer{ .batchnorm1d = layer });
    }

    pub fn addActivation(self: *Sequential, activation_fn: ActivationFn) !void {
        try self.layers.append(self.allocator, Layer{ .activation = activation_fn });
    }

    pub fn forward(self: *Sequential, input: *const Tensor) TensorError!Tensor {
        if (self.layers.items.len == 0) {
            return input.clone();
        }

        var current = try self.forwardLayer(input, self.layers.items[0]);
        for (self.layers.items[1..]) |layer| {
            const next = try self.forwardLayer(&current, layer);
            current.deinit();
            current = next;
        }
        return current;
    }

    fn forwardLayer(_: *Sequential, input: *const Tensor, layer: Layer) TensorError!Tensor {
        return switch (layer) {
            .linear => |l| l.forward(input),
            .conv1d => |c| c.forward(input),
            .maxpool1d => |m| m.forward(input),
            .dropout => |d| d.forward(input),
            .batchnorm1d => |b| b.forward(input),
            .activation => |a| a(input),
        };
    }

    pub fn setTraining(self: *Sequential, training: bool) void {
        self.training = training;
        for (self.layers.items) |layer| {
            switch (layer) {
                .dropout => |d| d.setTraining(training),
                .batchnorm1d => |b| b.setTraining(training),
                else => {},
            }
        }
    }
};

pub const Linear = struct {
    weights: Tensor,
    biases: Tensor,
    m_weights: Tensor,
    v_weights: Tensor,
    m_biases: Tensor,
    v_biases: Tensor,
    t: usize,

    pub fn init(allocator: std.mem.Allocator, in_features: usize, out_features: usize) !Linear {
        var weights = try Tensor.init(allocator, &[_]usize{ in_features, out_features });
        weights.randomize();
        var biases = try Tensor.init(allocator, &[_]usize{out_features});
        biases.randomize();
        const m_weights = try Tensor.init(allocator, &[_]usize{ in_features, out_features });
        @memset(m_weights.data, 0);
        const v_weights = try Tensor.init(allocator, &[_]usize{ in_features, out_features });
        @memset(v_weights.data, 0);
        const m_biases = try Tensor.init(allocator, &[_]usize{out_features});
        @memset(m_biases.data, 0);
        const v_biases = try Tensor.init(allocator, &[_]usize{out_features});
        @memset(v_biases.data, 0);
        return Linear{
            .weights = weights,
            .biases = biases,
            .m_weights = m_weights,
            .v_weights = v_weights,
            .m_biases = m_biases,
            .v_biases = v_biases,
            .t = 0,
        };
    }

    pub fn forward(self: *Linear, input: *const Tensor) TensorError!Tensor {
        if (input.shape.len != 2 or input.shape[1] != self.weights.shape[0]) return TensorError.ShapeMismatch;
        var out = try input.matmul(&self.weights);
        // add biases
        for (0..out.shape[0]) |i| {
            for (0..out.shape[1]) |j| {
                out.data[i * out.shape[1] + j] += self.biases.data[j];
            }
        }
        return out;
    }

    pub fn deinit(self: *Linear) void {
        self.weights.deinit();
        self.biases.deinit();
        self.m_weights.deinit();
        self.v_weights.deinit();
        self.m_biases.deinit();
        self.v_biases.deinit();
    }
};

pub fn relu(tensor: *const Tensor) TensorError!Tensor {
    var result = try Tensor.init(tensor.allocator, tensor.shape);
    for (0..tensor.data.len) |i| {
        result.data[i] = if (tensor.data[i] > 0) tensor.data[i] else 0;
    }
    return result;
}

pub fn sigmoid(tensor: *const Tensor) TensorError!Tensor {
    var result = if (tensor.pool) |pool|
        try Tensor.initWithPool(tensor.allocator, tensor.shape, pool)
    else
        try Tensor.init(tensor.allocator, tensor.shape);
    for (0..tensor.data.len) |i| {
        result.data[i] = 1.0 / (1.0 + std.math.exp(-tensor.data[i]));
    }
    return result;
}

pub fn tanh(tensor: *const Tensor) TensorError!Tensor {
    var result = if (tensor.pool) |pool|
        try Tensor.initWithPool(tensor.allocator, tensor.shape, pool)
    else
        try Tensor.init(tensor.allocator, tensor.shape);
    for (0..tensor.data.len) |i| {
        result.data[i] = std.math.tanh(tensor.data[i]);
    }
    return result;
}

pub fn leakyRelu(tensor: *const Tensor, alpha: f32) TensorError!Tensor {
    var result = if (tensor.pool) |pool|
        try Tensor.initWithPool(tensor.allocator, tensor.shape, pool)
    else
        try Tensor.init(tensor.allocator, tensor.shape);
    for (0..tensor.data.len) |i| {
        result.data[i] = if (tensor.data[i] > 0) tensor.data[i] else alpha * tensor.data[i];
    }
    return result;
}

pub fn gelu(tensor: *const Tensor) TensorError!Tensor {
    var result = if (tensor.pool) |pool|
        try Tensor.initWithPool(tensor.allocator, tensor.shape, pool)
    else
        try Tensor.init(tensor.allocator, tensor.shape);
    for (0..tensor.data.len) |i| {
        const x = tensor.data[i];
        result.data[i] = 0.5 * x * (1.0 + std.math.tanh(std.math.sqrt(2.0 / std.math.pi) * (x + 0.044715 * x * x * x)));
    }
    return result;
}

pub fn swish(tensor: *const Tensor, beta: f32) TensorError!Tensor {
    var result = if (tensor.pool) |pool|
        try Tensor.initWithPool(tensor.allocator, tensor.shape, pool)
    else
        try Tensor.init(tensor.allocator, tensor.shape);
    for (0..tensor.data.len) |i| {
        const x = tensor.data[i];
        result.data[i] = x / (1.0 + std.math.exp(-beta * x));
    }
    return result;
}

pub fn softmax(tensor: *const Tensor) TensorError!Tensor {
    if (tensor.shape.len == 0) return TensorError.InvalidShape;

    var result = if (tensor.pool) |pool|
        try Tensor.initWithPool(tensor.allocator, tensor.shape, pool)
    else
        try Tensor.init(tensor.allocator, tensor.shape);

    const last_dim = tensor.shape[tensor.shape.len - 1];
    const batch_size = tensor.data.len / last_dim;

    for (0..batch_size) |batch| {
        const start_idx = batch * last_dim;
        const end_idx = start_idx + last_dim;

        var max_val: f32 = tensor.data[start_idx];
        for (start_idx + 1..end_idx) |i| {
            max_val = @max(max_val, tensor.data[i]);
        }

        var sum: f32 = 0.0;
        for (start_idx..end_idx) |i| {
            result.data[i] = std.math.exp(tensor.data[i] - max_val);
            sum += result.data[i];
        }

        if (sum == 0.0) return TensorError.ZeroDivision;
        for (start_idx..end_idx) |i| {
            result.data[i] /= sum;
        }
    }

    return result;
}

pub fn mseLoss(pred: *const Tensor, target: *const Tensor) TensorError!f32 {
    if (!std.mem.eql(usize, pred.shape, target.shape)) return TensorError.ShapeMismatch;
    var diff = try pred.sub(target);
    defer diff.deinit();
    var sq = try diff.square();
    defer sq.deinit();
    const loss = sq.sum() / @as(f32, @floatFromInt(pred.data.len));
    return loss;
}

pub const Conv1D = struct {
    weights: Tensor,
    biases: Tensor,
    stride: usize,
    padding: usize,

    pub fn init(allocator: std.mem.Allocator, in_channels: usize, out_channels: usize, kernel_size: usize, stride: usize, padding: usize) TensorError!Conv1D {
        const weights = try Tensor.init(allocator, &[_]usize{ out_channels, in_channels, kernel_size });
        const biases = try Tensor.init(allocator, &[_]usize{out_channels});
        return Conv1D{
            .weights = weights,
            .biases = biases,
            .stride = stride,
            .padding = padding,
        };
    }

    pub fn forward(self: *const Conv1D, input: *const Tensor) TensorError!Tensor {
        if (input.shape.len != 3) return TensorError.InvalidShape;

        const batch_size = input.shape[0];
        const in_channels = input.shape[1];
        const input_length = input.shape[2];
        const out_channels = self.weights.shape[0];
        const kernel_size = self.weights.shape[2];

        const output_length = (input_length + 2 * self.padding - kernel_size) / self.stride + 1;
        var output = if (input.pool) |pool|
            try Tensor.initWithPool(input.allocator, &[_]usize{ batch_size, out_channels, output_length }, pool)
        else
            try Tensor.init(input.allocator, &[_]usize{ batch_size, out_channels, output_length });

        for (0..batch_size) |b| {
            for (0..out_channels) |oc| {
                for (0..output_length) |ol| {
                    var sum: f32 = 0.0;
                    for (0..in_channels) |ic| {
                        for (0..kernel_size) |k| {
                            const input_idx = ol * self.stride + k;
                            if (input_idx < self.padding or input_idx >= input_length + self.padding) continue;
                            const actual_input_idx = input_idx - self.padding;

                            const input_offset = b * in_channels * input_length + ic * input_length + actual_input_idx;
                            const weight_offset = oc * in_channels * kernel_size + ic * kernel_size + k;
                            sum += input.data[input_offset] * self.weights.data[weight_offset];
                        }
                    }
                    const output_offset = b * out_channels * output_length + oc * output_length + ol;
                    output.data[output_offset] = sum + self.biases.data[oc];
                }
            }
        }
        return output;
    }

    pub fn deinit(self: *Conv1D) void {
        self.weights.deinit();
        self.biases.deinit();
    }
};

pub const MaxPool1D = struct {
    kernel_size: usize,
    stride: usize,

    pub fn init(kernel_size: usize, stride: usize) MaxPool1D {
        return MaxPool1D{
            .kernel_size = kernel_size,
            .stride = stride,
        };
    }

    pub fn forward(self: *const MaxPool1D, input: *const Tensor) TensorError!Tensor {
        if (input.shape.len != 3) return TensorError.InvalidShape;

        const batch_size = input.shape[0];
        const channels = input.shape[1];
        const input_length = input.shape[2];
        const output_length = (input_length - self.kernel_size) / self.stride + 1;

        var output = if (input.pool) |pool|
            try Tensor.initWithPool(input.allocator, &[_]usize{ batch_size, channels, output_length }, pool)
        else
            try Tensor.init(input.allocator, &[_]usize{ batch_size, channels, output_length });

        for (0..batch_size) |b| {
            for (0..channels) |c| {
                for (0..output_length) |ol| {
                    var max_val: f32 = -std.math.inf(f32);
                    for (0..self.kernel_size) |k| {
                        const input_idx = ol * self.stride + k;
                        const input_offset = b * channels * input_length + c * input_length + input_idx;
                        max_val = @max(max_val, input.data[input_offset]);
                    }
                    const output_offset = b * channels * output_length + c * output_length + ol;
                    output.data[output_offset] = max_val;
                }
            }
        }
        return output;
    }
};

pub const Dropout = struct {
    p: f32,
    training: bool,

    pub fn init(p: f32) Dropout {
        return Dropout{
            .p = p,
            .training = true,
        };
    }

    pub fn forward(self: *const Dropout, input: *const Tensor) TensorError!Tensor {
        if (!self.training) {
            return input.clone();
        }

        var output = if (input.pool) |pool|
            try Tensor.initWithPool(input.allocator, input.shape, pool)
        else
            try Tensor.init(input.allocator, input.shape);

        const scale = 1.0 / (1.0 - self.p);
        for (0..input.data.len) |i| {
            const drop = @mod(i, 10) < @as(usize, @intFromFloat(self.p * 10));
            output.data[i] = if (drop) 0.0 else input.data[i] * scale;
        }
        return output;
    }

    pub fn setTraining(self: *Dropout, training: bool) void {
        self.training = training;
    }
};

pub const BatchNorm1D = struct {
    num_features: usize,
    gamma: Tensor,
    beta: Tensor,
    running_mean: Tensor,
    running_var: Tensor,
    eps: f32,
    momentum: f32,
    training: bool,

    pub fn init(allocator: std.mem.Allocator, num_features: usize) TensorError!BatchNorm1D {
        var gamma = try Tensor.init(allocator, &[_]usize{num_features});
        var beta = try Tensor.init(allocator, &[_]usize{num_features});
        var running_mean = try Tensor.init(allocator, &[_]usize{num_features});
        var running_var = try Tensor.init(allocator, &[_]usize{num_features});

        for (0..num_features) |i| {
            gamma.data[i] = 1.0;
            beta.data[i] = 0.0;
            running_mean.data[i] = 0.0;
            running_var.data[i] = 1.0;
        }

        return BatchNorm1D{
            .num_features = num_features,
            .gamma = gamma,
            .beta = beta,
            .running_mean = running_mean,
            .running_var = running_var,
            .eps = 1e-5,
            .momentum = 0.1,
            .training = true,
        };
    }

    pub fn forward(self: *BatchNorm1D, input: *const Tensor) TensorError!Tensor {
        if (input.shape.len < 2) return TensorError.InvalidShape;
        if (input.shape[input.shape.len - 1] != self.num_features) return TensorError.ShapeMismatch;

        var output = if (input.pool) |pool|
            try Tensor.initWithPool(input.allocator, input.shape, pool)
        else
            try Tensor.init(input.allocator, input.shape);

        const batch_size = input.data.len / self.num_features;

        if (self.training) {
            for (0..self.num_features) |f| {
                var mean: f32 = 0.0;
                for (0..batch_size) |b| {
                    mean += input.data[b * self.num_features + f];
                }
                mean /= @as(f32, @floatFromInt(batch_size));

                var variance: f32 = 0.0;
                for (0..batch_size) |b| {
                    const diff = input.data[b * self.num_features + f] - mean;
                    variance += diff * diff;
                }
                variance /= @as(f32, @floatFromInt(batch_size));

                self.running_mean.data[f] = (1.0 - self.momentum) * self.running_mean.data[f] + self.momentum * mean;
                self.running_var.data[f] = (1.0 - self.momentum) * self.running_var.data[f] + self.momentum * variance;

                for (0..batch_size) |b| {
                    const normalized = (input.data[b * self.num_features + f] - mean) / std.math.sqrt(variance + self.eps);
                    output.data[b * self.num_features + f] = self.gamma.data[f] * normalized + self.beta.data[f];
                }
            }
        } else {
            for (0..self.num_features) |f| {
                for (0..batch_size) |b| {
                    const normalized = (input.data[b * self.num_features + f] - self.running_mean.data[f]) / std.math.sqrt(self.running_var.data[f] + self.eps);
                    output.data[b * self.num_features + f] = self.gamma.data[f] * normalized + self.beta.data[f];
                }
            }
        }

        return output;
    }

    pub fn setTraining(self: *BatchNorm1D, training: bool) void {
        self.training = training;
    }

    pub fn deinit(self: *BatchNorm1D) void {
        self.gamma.deinit();
        self.beta.deinit();
        self.running_mean.deinit();
        self.running_var.deinit();
    }
};

pub const OptimizerType = enum {
    sgd,
    adam,
    adamw,
    rmsprop,
};

pub const OptimizerConfig = struct {
    learning_rate: f32,
    weight_decay: f32 = 0.0,
    momentum: f32 = 0.0,
    beta1: f32 = 0.9,
    beta2: f32 = 0.999,
    eps: f32 = 1e-8,
};

pub const Optimizer = struct {
    config: OptimizerConfig,
    optimizer_type: OptimizerType,
    step_count: usize,

    pub fn init(optimizer_type: OptimizerType, config: OptimizerConfig) Optimizer {
        return Optimizer{
            .config = config,
            .optimizer_type = optimizer_type,
            .step_count = 0,
        };
    }

    pub fn step(self: *Optimizer, params: *Tensor, gradients: *const Tensor, momentum: ?*Tensor, velocity: ?*Tensor) void {
        self.step_count += 1;

        switch (self.optimizer_type) {
            .sgd => self.sgdStep(params, gradients, momentum),
            .adam => self.adamStep(params, gradients, momentum.?, velocity.?),
            .adamw => self.adamwStep(params, gradients, momentum.?, velocity.?),
            .rmsprop => self.rmspropStep(params, gradients, velocity.?),
        }
    }

    fn sgdStep(self: *Optimizer, params: *Tensor, gradients: *const Tensor, momentum: ?*Tensor) void {
        if (momentum) |m| {
            // SGD with momentum
            for (0..params.data.len) |i| {
                m.data[i] = self.config.momentum * m.data[i] + gradients.data[i];
                params.data[i] -= self.config.learning_rate * m.data[i];
            }
        } else {
            // Vanilla SGD
            for (0..params.data.len) |i| {
                params.data[i] -= self.config.learning_rate * gradients.data[i];
            }
        }

        // Weight decay
        if (self.config.weight_decay > 0.0) {
            for (0..params.data.len) |i| {
                params.data[i] -= self.config.learning_rate * self.config.weight_decay * params.data[i];
            }
        }
    }

    fn adamStep(self: *Optimizer, params: *Tensor, gradients: *const Tensor, momentum: *Tensor, velocity: *Tensor) void {
        const t_f = @as(f32, @floatFromInt(self.step_count));
        const beta1_t = std.math.pow(f32, self.config.beta1, t_f);
        const beta2_t = std.math.pow(f32, self.config.beta2, t_f);

        for (0..params.data.len) |i| {
            momentum.data[i] = self.config.beta1 * momentum.data[i] + (1.0 - self.config.beta1) * gradients.data[i];
            velocity.data[i] = self.config.beta2 * velocity.data[i] + (1.0 - self.config.beta2) * gradients.data[i] * gradients.data[i];

            const m_hat = momentum.data[i] / (1.0 - beta1_t);
            const v_hat = velocity.data[i] / (1.0 - beta2_t);

            params.data[i] -= self.config.learning_rate * m_hat / (std.math.sqrt(v_hat) + self.config.eps);
        }
    }

    fn adamwStep(self: *Optimizer, params: *Tensor, gradients: *const Tensor, momentum: *Tensor, velocity: *Tensor) void {
        const t_f = @as(f32, @floatFromInt(self.step_count));
        const beta1_t = std.math.pow(f32, self.config.beta1, t_f);
        const beta2_t = std.math.pow(f32, self.config.beta2, t_f);

        for (0..params.data.len) |i| {
            // Weight decay applied directly to parameters (decoupled weight decay)
            params.data[i] -= self.config.learning_rate * self.config.weight_decay * params.data[i];

            momentum.data[i] = self.config.beta1 * momentum.data[i] + (1.0 - self.config.beta1) * gradients.data[i];
            velocity.data[i] = self.config.beta2 * velocity.data[i] + (1.0 - self.config.beta2) * gradients.data[i] * gradients.data[i];

            const m_hat = momentum.data[i] / (1.0 - beta1_t);
            const v_hat = velocity.data[i] / (1.0 - beta2_t);

            params.data[i] -= self.config.learning_rate * m_hat / (std.math.sqrt(v_hat) + self.config.eps);
        }
    }

    fn rmspropStep(self: *Optimizer, params: *Tensor, gradients: *const Tensor, velocity: *Tensor) void {
        for (0..params.data.len) |i| {
            velocity.data[i] = self.config.beta2 * velocity.data[i] + (1.0 - self.config.beta2) * gradients.data[i] * gradients.data[i];
            params.data[i] -= self.config.learning_rate * gradients.data[i] / (std.math.sqrt(velocity.data[i]) + self.config.eps);
        }

        // Weight decay
        if (self.config.weight_decay > 0.0) {
            for (0..params.data.len) |i| {
                params.data[i] -= self.config.learning_rate * self.config.weight_decay * params.data[i];
            }
        }
    }

    pub fn clipGradients(gradients: *Tensor, max_norm: f32) void {
        var total_norm: f32 = 0.0;
        for (gradients.data) |grad| {
            total_norm += grad * grad;
        }
        total_norm = std.math.sqrt(total_norm);

        if (total_norm > max_norm) {
            const clip_coef = max_norm / total_norm;
            for (gradients.data) |*grad| {
                grad.* *= clip_coef;
            }
        }
    }
};

pub const LearningRateScheduler = struct {
    initial_lr: f32,
    scheduler_type: SchedulerType,
    step_size: usize,
    gamma: f32,
    current_step: usize,

    const SchedulerType = enum {
        constant,
        step_lr,
        exponential,
        cosine_annealing,
    };

    pub fn init(scheduler_type: SchedulerType, initial_lr: f32, step_size: usize, gamma: f32) LearningRateScheduler {
        return LearningRateScheduler{
            .initial_lr = initial_lr,
            .scheduler_type = scheduler_type,
            .step_size = step_size,
            .gamma = gamma,
            .current_step = 0,
        };
    }

    pub fn getLearningRate(self: *LearningRateScheduler) f32 {
        return switch (self.scheduler_type) {
            .constant => self.initial_lr,
            .step_lr => self.initial_lr * std.math.pow(f32, self.gamma, @as(f32, @floatFromInt(self.current_step / self.step_size))),
            .exponential => self.initial_lr * std.math.pow(f32, self.gamma, @as(f32, @floatFromInt(self.current_step))),
            .cosine_annealing => {
                const cos_val = std.math.cos(std.math.pi * @as(f32, @floatFromInt(self.current_step)) / @as(f32, @floatFromInt(self.step_size)));
                return self.initial_lr * (1.0 + cos_val) / 2.0;
            },
        };
    }

    pub fn step(self: *LearningRateScheduler) void {
        self.current_step += 1;
    }
};

pub fn trainLinear(allocator: std.mem.Allocator, layer: *Linear, x: *const Tensor, y: *const Tensor, lr: f32, epochs: usize) TensorError!void {
    for (0..epochs) |_| {
        var pred = try layer.forward(x);
        defer pred.deinit();
        const loss = try mseLoss(&pred, y);
        std.debug.print("Loss: {d}\n", .{loss});

        var dL_dy = try pred.sub(y);
        defer dL_dy.deinit();
        for (dL_dy.data) |*val| val.* /= @as(f32, @floatFromInt(dL_dy.data.len));

        var dL_db = try Tensor.init(allocator, layer.biases.shape);
        defer dL_db.deinit();
        for (0..dL_dy.shape[0]) |i| {
            for (0..dL_dy.shape[1]) |j| {
                dL_db.data[j] += dL_dy.data[i * dL_dy.shape[1] + j];
            }
        }

        var x_t = try x.transpose();
        defer x_t.deinit();
        var dL_dw = try x_t.matmul(&dL_dy);
        defer dL_dw.deinit();

        // Adam update
        layer.t += 1;
        const beta1 = 0.9;
        const beta2 = 0.999;
        const eps = 1e-8;
        const learning_rate = lr;
        const t_f = @as(f32, @floatFromInt(layer.t));
        const beta1_t = std.math.pow(f32, beta1, t_f);
        const beta2_t = std.math.pow(f32, beta2, t_f);

        for (0..layer.weights.data.len) |i| {
            layer.m_weights.data[i] = beta1 * layer.m_weights.data[i] + (1 - beta1) * dL_dw.data[i];
            layer.v_weights.data[i] = beta2 * layer.v_weights.data[i] + (1 - beta2) * dL_dw.data[i] * dL_dw.data[i];
            const m_hat = layer.m_weights.data[i] / (1 - beta1_t);
            const v_hat = layer.v_weights.data[i] / (1 - beta2_t);
            layer.weights.data[i] -= learning_rate * m_hat / (std.math.sqrt(v_hat) + eps);
        }

        for (0..layer.biases.data.len) |i| {
            layer.m_biases.data[i] = beta1 * layer.m_biases.data[i] + (1 - beta1) * dL_db.data[i];
            layer.v_biases.data[i] = beta2 * layer.v_biases.data[i] + (1 - beta2) * dL_db.data[i] * dL_db.data[i];
            const m_hat = layer.m_biases.data[i] / (1 - beta1_t);
            const v_hat = layer.v_biases.data[i] / (1 - beta2_t);
            layer.biases.data[i] -= learning_rate * m_hat / (std.math.sqrt(v_hat) + eps);
        }
    }
}

pub const DataLoader = struct {
    data: []const Tensor,
    targets: []const Tensor,
    batch_size: usize,
    shuffle: bool,
    current_index: usize,
    indices: std.ArrayList(usize),
    allocator: std.mem.Allocator,

    pub fn init(allocator: std.mem.Allocator, data: []const Tensor, targets: []const Tensor, batch_size: usize, shuffle: bool) TensorError!DataLoader {
        var indices = std.ArrayList(usize){};
        for (0..data.len) |i| {
            indices.append(allocator, i) catch {
                indices.deinit(allocator);
                return TensorError.OutOfMemory;
            };
        }

        var loader = DataLoader{
            .data = data,
            .targets = targets,
            .batch_size = batch_size,
            .shuffle = shuffle,
            .current_index = 0,
            .indices = indices,
            .allocator = allocator,
        };

        if (shuffle) {
            loader.shuffleIndices();
        }

        return loader;
    }

    pub fn deinit(self: *DataLoader) void {
        self.indices.deinit(self.allocator);
    }

    pub fn nextBatch(self: *DataLoader) ?struct { data: []const Tensor, targets: []const Tensor } {
        if (self.current_index >= self.indices.items.len) {
            return null;
        }

        const end_index = @min(self.current_index + self.batch_size, self.indices.items.len);
        const batch_indices = self.indices.items[self.current_index..end_index];

        const batch_data = batch_indices;
        const batch_targets = batch_indices;

        self.current_index = end_index;

        return .{
            .data = self.data[batch_data[0]..@min(batch_data[0] + batch_data.len, self.data.len)],
            .targets = self.targets[batch_targets[0]..@min(batch_targets[0] + batch_targets.len, self.targets.len)],
        };
    }

    pub fn reset(self: *DataLoader) void {
        self.current_index = 0;
        if (self.shuffle) {
            self.shuffleIndices();
        }
    }

    fn shuffleIndices(self: *DataLoader) void {
        for (self.indices.items, 0..) |_, i| {
            const j = i + (@as(usize, @intCast(std.time.timestamp())) + i) % (self.indices.items.len - i);
            if (j < self.indices.items.len) {
                const temp = self.indices.items[i];
                self.indices.items[i] = self.indices.items[j];
                self.indices.items[j] = temp;
            }
        }
    }

    pub fn len(self: *const DataLoader) usize {
        return (self.indices.items.len + self.batch_size - 1) / self.batch_size;
    }
};

pub const DataAugmentation = struct {
    pub fn addNoise(tensor: *const Tensor, noise_level: f32, allocator: std.mem.Allocator) TensorError!Tensor {
        var result = if (tensor.pool) |pool|
            try Tensor.initWithPool(allocator, tensor.shape, pool)
        else
            try Tensor.init(allocator, tensor.shape);

        for (0..tensor.data.len) |i| {
            const noise = ((@as(f32, @floatFromInt(i)) / @as(f32, @floatFromInt(tensor.data.len))) - 0.5) * 2.0 * noise_level;
            result.data[i] = tensor.data[i] + noise;
        }

        return result;
    }

    pub fn normalize(tensor: *const Tensor, mean: f32, std_dev: f32, allocator: std.mem.Allocator) TensorError!Tensor {
        var result = if (tensor.pool) |pool|
            try Tensor.initWithPool(allocator, tensor.shape, pool)
        else
            try Tensor.init(allocator, tensor.shape);

        for (0..tensor.data.len) |i| {
            result.data[i] = (tensor.data[i] - mean) / std_dev;
        }

        return result;
    }

    pub fn scale(tensor: *const Tensor, min_val: f32, max_val: f32, allocator: std.mem.Allocator) TensorError!Tensor {
        var result = if (tensor.pool) |pool|
            try Tensor.initWithPool(allocator, tensor.shape, pool)
        else
            try Tensor.init(allocator, tensor.shape);

        var tensor_min = tensor.data[0];
        var tensor_max = tensor.data[0];
        for (tensor.data) |val| {
            tensor_min = @min(tensor_min, val);
            tensor_max = @max(tensor_max, val);
        }

        const scale_factor = (max_val - min_val) / (tensor_max - tensor_min);
        for (0..tensor.data.len) |i| {
            result.data[i] = min_val + (tensor.data[i] - tensor_min) * scale_factor;
        }

        return result;
    }
};

pub const Dataset = struct {
    data: std.ArrayList(Tensor),
    targets: std.ArrayList(Tensor),
    allocator: std.mem.Allocator,

    pub fn init(allocator: std.mem.Allocator) Dataset {
        return Dataset{
            .data = std.ArrayList(Tensor){},
            .targets = std.ArrayList(Tensor){},
            .allocator = allocator,
        };
    }

    pub fn deinit(self: *Dataset) void {
        for (self.data.items) |*tensor| {
            tensor.deinit();
        }
        for (self.targets.items) |*tensor| {
            tensor.deinit();
        }
        self.data.deinit(self.allocator);
        self.targets.deinit(self.allocator);
    }

    pub fn addSample(self: *Dataset, data: Tensor, target: Tensor) TensorError!void {
        var owned_data = data;
        var owned_target = target;

        self.data.append(self.allocator, owned_data) catch {
            owned_data.deinit();
            owned_target.deinit();
            return TensorError.OutOfMemory;
        };

        self.targets.append(self.allocator, owned_target) catch {
            const removed = self.data.pop();
            removed.deinit();
            owned_target.deinit();
            return TensorError.OutOfMemory;
        };
    }

    pub fn len(self: *const Dataset) usize {
        return self.data.items.len;
    }

    pub fn getDataLoader(self: *const Dataset, batch_size: usize, shuffle: bool, allocator: std.mem.Allocator) !DataLoader {
        return try DataLoader.init(allocator, self.data.items, self.targets.items, batch_size, shuffle);
    }
};

pub const Benchmark = struct {
    name: []const u8,
    setup_fn: *const fn (std.mem.Allocator) anyerror!void,
    bench_fn: *const fn (std.mem.Allocator) anyerror!u64,
    teardown_fn: *const fn (std.mem.Allocator) anyerror!void,
};

pub fn runBenchmarks(allocator: std.mem.Allocator, benchmarks: []const Benchmark) !void {
    std.debug.print("Running {} benchmarks...\n", .{benchmarks.len});

    for (benchmarks) |bench| {
        std.debug.print("Benchmark: {s}... ", .{bench.name});

        try bench.setup_fn(allocator);
        const start_time = std.time.nanoTimestamp();
        const ops = try bench.bench_fn(allocator);
        const end_time = std.time.nanoTimestamp();
        try bench.teardown_fn(allocator);

        const elapsed_ns = @as(u64, @intCast(end_time - start_time));
        const ops_per_sec = if (elapsed_ns > 0) (ops * 1_000_000_000) / elapsed_ns else 0;

        std.debug.print("{d:.2} ns/op, {d} ops/sec\n", .{
            @as(f64, @floatFromInt(elapsed_ns)) / @as(f64, @floatFromInt(ops)),
            ops_per_sec
        });
    }
}

fn benchTensorAdd(allocator: std.mem.Allocator) !u64 {
    const size = 10000;
    var a = try Tensor.init(allocator, &[_]usize{size});
    defer a.deinit();
    var b = try Tensor.init(allocator, &[_]usize{size});
    defer b.deinit();

    for (0..size) |i| {
        a.data[i] = @as(f32, @floatFromInt(i));
        b.data[i] = @as(f32, @floatFromInt(i)) * 2.0;
    }

    const iterations = 1000;
    for (0..iterations) |_| {
        var c = try a.add(&b);
        c.deinit();
    }

    return iterations;
}

fn benchTensorAddSimd(allocator: std.mem.Allocator) !u64 {
    const size = 10000;
    var a = try Tensor.init(allocator, &[_]usize{size});
    defer a.deinit();
    var b = try Tensor.init(allocator, &[_]usize{size});
    defer b.deinit();

    for (0..size) |i| {
        a.data[i] = @as(f32, @floatFromInt(i));
        b.data[i] = @as(f32, @floatFromInt(i)) * 2.0;
    }

    const iterations = 1000;
    for (0..iterations) |_| {
        var c = try a.addSimd(&b);
        c.deinit();
    }

    return iterations;
}

fn benchMatmul(allocator: std.mem.Allocator) !u64 {
    const size = 100;
    var a = try Tensor.init(allocator, &[_]usize{size, size});
    defer a.deinit();
    var b = try Tensor.init(allocator, &[_]usize{size, size});
    defer b.deinit();

    for (0..size*size) |i| {
        a.data[i] = @as(f32, @floatFromInt(i));
        b.data[i] = @as(f32, @floatFromInt(i)) * 0.1;
    }

    const iterations = 10;
    for (0..iterations) |_| {
        var c = try a.matmul(&b);
        c.deinit();
    }

    return iterations;
}

fn noOpSetup(_: std.mem.Allocator) !void {}
fn noOpTeardown(_: std.mem.Allocator) !void {}

pub const default_benchmarks = [_]Benchmark{
    .{ .name = "tensor_add", .setup_fn = noOpSetup, .bench_fn = benchTensorAdd, .teardown_fn = noOpTeardown },
    .{ .name = "tensor_add_simd", .setup_fn = noOpSetup, .bench_fn = benchTensorAddSimd, .teardown_fn = noOpTeardown },
    .{ .name = "matmul", .setup_fn = noOpSetup, .bench_fn = benchMatmul, .teardown_fn = noOpTeardown },
};

pub fn bufferedPrint() !void {
    var stdout_buffer: [1024]u8 = undefined;
    var stdout_writer = std.fs.File.stdout().writer(&stdout_buffer);
    const stdout = &stdout_writer.interface;

    try stdout.print("Run `zig build test` to run the tests.\n", .{});

    try stdout.flush();
}

pub fn add(a: i32, b: i32) i32 {
    return a + b;
}

// =============================================================================
// AUTOMATIC DIFFERENTIATION ENGINE
// =============================================================================

pub const ComputationNodeType = enum {
    input,     // Leaf nodes (parameters/inputs)
    add,       // Addition operation
    mul,       // Multiplication operation
    matmul,    // Matrix multiplication
    relu,      // ReLU activation
    sigmoid,   // Sigmoid activation
    tanh,      // Tanh activation
    mse_loss,  // Mean squared error loss
    cross_entropy_loss, // Cross entropy loss
    sum,       // Sum reduction
    transpose, // Matrix transpose
    reshape,   // Tensor reshape
};

pub const ComputationNode = struct {
    id: u32,
    node_type: ComputationNodeType,
    tensor: *Tensor,
    gradient: ?*Tensor,
    parents: std.ArrayList(*ComputationNode),
    children: std.ArrayList(*ComputationNode),
    grad_fn: ?*const fn(node: *ComputationNode, grad: *const Tensor) TensorError!void,
    allocator: std.mem.Allocator,
    requires_grad: bool,
    owns_tensor: bool,

    pub fn init(allocator: std.mem.Allocator, tensor: *Tensor, node_type: ComputationNodeType) TensorError!ComputationNode {
        return ComputationNode{
            .id = 0,
            .node_type = node_type,
            .tensor = tensor,
            .gradient = null,
            .parents = std.ArrayList(*ComputationNode){},
            .children = std.ArrayList(*ComputationNode){},
            .grad_fn = null,
            .allocator = allocator,
            .requires_grad = true,
            .owns_tensor = false,
        };
    }

    pub fn deinit(self: *ComputationNode) void {
        if (self.gradient) |grad| {
            grad.deinit();
            self.allocator.destroy(grad);
        }
        if (self.owns_tensor) {
            self.tensor.deinit();
            self.allocator.destroy(self.tensor);
        }
        self.parents.deinit(self.allocator);
        self.children.deinit(self.allocator);
    }

    pub fn addParent(self: *ComputationNode, parent: *ComputationNode) TensorError!void {
        self.parents.append(self.allocator, parent) catch return TensorError.OutOfMemory;
        parent.children.append(parent.allocator, self) catch return TensorError.OutOfMemory;
    }

    pub fn backward(self: *ComputationNode, grad: ?*const Tensor) TensorError!void {
        // Initialize gradient if not present
        if (self.gradient == null) {
            if (grad) |g| {
                var g_clone = g.clone();
                const grad_ptr = self.allocator.create(Tensor) catch {
                    g_clone.deinit();
                    return TensorError.OutOfMemory;
                };
                grad_ptr.* = g_clone;
                self.gradient = grad_ptr;
            } else {
                // Create gradient of ones for final output
                const grad_ptr = self.allocator.create(Tensor) catch return TensorError.OutOfMemory;
                const grad_tensor = Tensor.init(self.allocator, self.tensor.shape) catch |err| {
                    self.allocator.destroy(grad_ptr);
                    return err;
                };
                grad_ptr.* = grad_tensor;
                self.gradient = grad_ptr;
                for (self.gradient.?.data) |*val| {
                    val.* = 1.0;
                }
            }
        } else if (grad) |g| {
            // Accumulate gradient
            var new_grad = try self.gradient.?.add(g);
            defer new_grad.deinit();
            @memcpy(self.gradient.?.data, new_grad.data);
        }

        // Call gradient function if available
        if (self.grad_fn) |grad_fn| {
            try grad_fn(self, self.gradient.?);
        }

    }
};

pub const ComputationGraph = struct {
    nodes: std.ArrayList(*ComputationNode),
    next_id: u32,
    allocator: std.mem.Allocator,
    debug_mode: bool,

    pub fn init(allocator: std.mem.Allocator) ComputationGraph {
        return ComputationGraph{
            .nodes = std.ArrayList(*ComputationNode){},
            .next_id = 0,
            .allocator = allocator,
            .debug_mode = false,
        };
    }

    pub fn deinit(self: *ComputationGraph) void {
        for (self.nodes.items) |node| {
            node.deinit();
            self.allocator.destroy(node);
        }
        self.nodes.deinit(self.allocator);
    }

    pub fn createNode(self: *ComputationGraph, tensor: *Tensor, node_type: ComputationNodeType, owns_tensor: bool) TensorError!*ComputationNode {
        const node = self.allocator.create(ComputationNode) catch return TensorError.OutOfMemory;
        node.* = try ComputationNode.init(self.allocator, tensor, node_type);
        node.owns_tensor = owns_tensor;
        node.id = self.next_id;
        self.next_id += 1;
        self.nodes.append(self.allocator, node) catch {
            self.next_id -= 1;
            self.allocator.destroy(node);
            return TensorError.OutOfMemory;
        };

        if (self.debug_mode) {
            std.debug.print("Created node {d} ({s}) with shape: ", .{ node.id, @tagName(node_type) });
            for (tensor.shape) |dim| {
                std.debug.print("{d} ", .{dim});
            }
            std.debug.print("\n", .{});
        }

        return node;
    }

    pub fn backward(self: *ComputationGraph, output_node: *ComputationNode) TensorError!void {
        if (self.debug_mode) {
            std.debug.print("Starting backpropagation from node {d}\n", .{output_node.id});
        }
        try output_node.backward(null);
    }

    pub fn zeroGrad(self: *ComputationGraph) void {
        for (self.nodes.items) |node| {
            if (node.gradient) |grad| {
                for (grad.data) |*val| {
                    val.* = 0.0;
                }
            }
        }
    }
};

// Gradient functions for different operations
fn addGradFn(node: *ComputationNode, grad: *const Tensor) TensorError!void {
    // For addition: gradient flows unchanged to both parents
    for (node.parents.items) |parent| {
        try parent.backward(grad);
    }
}

fn mulGradFn(node: *ComputationNode, grad: *const Tensor) TensorError!void {
    // For element-wise multiplication: grad * other_tensor
    if (node.parents.items.len != 2) return TensorError.InvalidInput;

    const left = node.parents.items[0];
    const right = node.parents.items[1];

    // Gradient w.r.t. left parent: grad * right.tensor
    var left_grad = try grad.mul(right.tensor);
    defer left_grad.deinit();
    try left.backward(&left_grad);

    // Gradient w.r.t. right parent: grad * left.tensor
    var right_grad = try grad.mul(left.tensor);
    defer right_grad.deinit();
    try right.backward(&right_grad);
}

fn matmulGradFn(node: *ComputationNode, grad: *const Tensor) TensorError!void {
    // For matrix multiplication: complex gradient computation
    if (node.parents.items.len != 2) return TensorError.InvalidInput;

    const left = node.parents.items[0];  // A
    const right = node.parents.items[1]; // B

    // Gradient w.r.t. A: grad @ B^T
    const right_t = try right.tensor.transpose();
    defer right_t.deinit();
    const left_grad = try grad.matmul(&right_t);
    defer left_grad.deinit();
    try left.backward(&left_grad);

    // Gradient w.r.t. B: A^T @ grad
    const left_t = try left.tensor.transpose();
    defer left_t.deinit();
    const right_grad = try left_t.matmul(grad);
    defer right_grad.deinit();
    try right.backward(&right_grad);
}

fn reluGradFn(node: *ComputationNode, grad: *const Tensor) TensorError!void {
    // For ReLU: gradient is grad where input > 0, else 0
    if (node.parents.items.len != 1) return TensorError.InvalidInput;

    const parent = node.parents.items[0];
    const input = parent.tensor;

    const masked_grad = try Tensor.init(grad.allocator, grad.shape);
    defer masked_grad.deinit();

    for (0..input.data.len) |i| {
        masked_grad.data[i] = if (input.data[i] > 0.0) grad.data[i] else 0.0;
    }

    try parent.backward(&masked_grad);
}

fn sigmoidGradFn(node: *ComputationNode, grad: *const Tensor) TensorError!void {
    // For sigmoid: gradient is grad * sigmoid * (1 - sigmoid)
    if (node.parents.items.len != 1) return TensorError.InvalidInput;

    const parent = node.parents.items[0];
    const output = node.tensor; // sigmoid output

    const sig_grad = try Tensor.init(grad.allocator, grad.shape);
    defer sig_grad.deinit();

    for (0..output.data.len) |i| {
        const sig = output.data[i];
        sig_grad.data[i] = grad.data[i] * sig * (1.0 - sig);
    }

    try parent.backward(&sig_grad);
}

fn mseGradFn(node: *ComputationNode, grad: *const Tensor) TensorError!void {
    // For MSE loss: gradient is 2 * (pred - target) / n
    if (node.parents.items.len != 2) return TensorError.InvalidInput;

    const pred = node.parents.items[0];
    const target = node.parents.items[1];

    var diff = try pred.tensor.sub(target.tensor);
    defer diff.deinit();

    const n = @as(f32, @floatFromInt(pred.tensor.data.len));
    var pred_grad = try Tensor.init(grad.allocator, pred.tensor.shape);
    defer pred_grad.deinit();

    for (0..diff.data.len) |i| {
        pred_grad.data[i] = grad.data[i] * 2.0 * diff.data[i] / n;
    }

    try pred.backward(&pred_grad);
    // Target typically doesn't require gradients
}

// Autograd-enabled Tensor wrapper
pub const Variable = struct {
    tensor: *Tensor,
    node: *ComputationNode,
    graph: *ComputationGraph,
    requires_grad: bool,

    pub fn init(graph: *ComputationGraph, tensor: *Tensor, requires_grad: bool) TensorError!Variable {
    const node = try graph.createNode(tensor, .input, false);
        node.requires_grad = requires_grad;
        return Variable{
            .tensor = tensor,
            .node = node,
            .graph = graph,
            .requires_grad = requires_grad,
        };
    }

    fn createTensorPtr(self: *const Variable, tensor_val: Tensor) TensorError!*Tensor {
        var owned_tensor = tensor_val;
        const tensor_ptr = self.graph.allocator.create(Tensor) catch {
            owned_tensor.deinit();
            return TensorError.OutOfMemory;
        };
        tensor_ptr.* = owned_tensor;
        return tensor_ptr;
    }

    pub fn add(self: *const Variable, other: *const Variable) TensorError!Variable {
        const result_tensor_val = try self.tensor.add(other.tensor);
        const result_tensor = try self.createTensorPtr(result_tensor_val);
        const result_node = self.graph.createNode(result_tensor, .add, true) catch |err| {
            result_tensor.deinit();
            self.graph.allocator.destroy(result_tensor);
            return err;
        };
        errdefer {
            self.graph.next_id -= 1;
            _ = self.graph.nodes.pop();
            result_node.deinit();
            self.graph.allocator.destroy(result_node);
        }
        result_node.grad_fn = addGradFn;

        try result_node.addParent(self.node);
        try result_node.addParent(other.node);

        return Variable{
            .tensor = result_tensor,
            .node = result_node,
            .graph = self.graph,
            .requires_grad = self.requires_grad or other.requires_grad,
        };
    }

    pub fn mul(self: *const Variable, other: *const Variable) TensorError!Variable {
        const result_tensor_val = try self.tensor.mul(other.tensor);
        const result_tensor = try self.createTensorPtr(result_tensor_val);
        const result_node = self.graph.createNode(result_tensor, .mul, true) catch |err| {
            result_tensor.deinit();
            self.graph.allocator.destroy(result_tensor);
            return err;
        };
        errdefer {
            self.graph.next_id -= 1;
            _ = self.graph.nodes.pop();
            result_node.deinit();
            self.graph.allocator.destroy(result_node);
        }
        result_node.grad_fn = mulGradFn;

        try result_node.addParent(self.node);
        try result_node.addParent(other.node);

        return Variable{
            .tensor = result_tensor,
            .node = result_node,
            .graph = self.graph,
            .requires_grad = self.requires_grad or other.requires_grad,
        };
    }

    pub fn matmul(self: *const Variable, other: *const Variable) TensorError!Variable {
        const result_tensor_val = try self.tensor.matmul(other.tensor);
        const result_tensor = try self.createTensorPtr(result_tensor_val);
        const result_node = self.graph.createNode(result_tensor, .matmul, true) catch |err| {
            result_tensor.deinit();
            self.graph.allocator.destroy(result_tensor);
            return err;
        };
        errdefer {
            self.graph.next_id -= 1;
            _ = self.graph.nodes.pop();
            result_node.deinit();
            self.graph.allocator.destroy(result_node);
        }
        result_node.grad_fn = matmulGradFn;

        try result_node.addParent(self.node);
        try result_node.addParent(other.node);

        return Variable{
            .tensor = result_tensor,
            .node = result_node,
            .graph = self.graph,
            .requires_grad = self.requires_grad or other.requires_grad,
        };
    }

    pub fn relu(self: *const Variable) TensorError!Variable {
        const result_tensor_val = try self.tensor.relu();
        const result_tensor = try self.createTensorPtr(result_tensor_val);
        const result_node = self.graph.createNode(result_tensor, .relu, true) catch |err| {
            result_tensor.deinit();
            self.graph.allocator.destroy(result_tensor);
            return err;
        };
        errdefer {
            self.graph.next_id -= 1;
            _ = self.graph.nodes.pop();
            result_node.deinit();
            self.graph.allocator.destroy(result_node);
        }
        result_node.grad_fn = reluGradFn;

        try result_node.addParent(self.node);

        return Variable{
            .tensor = result_tensor,
            .node = result_node,
            .graph = self.graph,
            .requires_grad = self.requires_grad,
        };
    }

    pub fn sigmoid(self: *const Variable) TensorError!Variable {
        const result_tensor_val = try self.tensor.sigmoid();
        const result_tensor = try self.createTensorPtr(result_tensor_val);
        const result_node = self.graph.createNode(result_tensor, .sigmoid, true) catch |err| {
            result_tensor.deinit();
            self.graph.allocator.destroy(result_tensor);
            return err;
        };
        errdefer {
            self.graph.next_id -= 1;
            _ = self.graph.nodes.pop();
            result_node.deinit();
            self.graph.allocator.destroy(result_node);
        }
        result_node.grad_fn = sigmoidGradFn;

        try result_node.addParent(self.node);

        return Variable{
            .tensor = result_tensor,
            .node = result_node,
            .graph = self.graph,
            .requires_grad = self.requires_grad,
        };
    }

    pub fn mseLoss(self: *const Variable, target: *const Variable) TensorError!Variable {
        var diff = try self.tensor.sub(target.tensor);
        defer diff.deinit();

        var squared = try diff.square();
        defer squared.deinit();

        const loss_value = squared.sum() / @as(f32, @floatFromInt(squared.data.len));
        var loss_tensor_val = try Tensor.init(self.tensor.allocator, &[_]usize{1});
        loss_tensor_val.data[0] = loss_value;
        const loss_tensor = try self.createTensorPtr(loss_tensor_val);

        const loss_node = self.graph.createNode(loss_tensor, .mse_loss, true) catch |err| {
            loss_tensor.deinit();
            self.graph.allocator.destroy(loss_tensor);
            return err;
        };
        errdefer {
            self.graph.next_id -= 1;
            _ = self.graph.nodes.pop();
            loss_node.deinit();
            self.graph.allocator.destroy(loss_node);
        }
        loss_node.grad_fn = mseGradFn;

        try loss_node.addParent(self.node);
        try loss_node.addParent(target.node);

        return Variable{
            .tensor = loss_tensor,
            .node = loss_node,
            .graph = self.graph,
            .requires_grad = true,
        };
    }

    pub fn backward(self: *const Variable) TensorError!void {
        try self.graph.backward(self.node);
    }

    pub fn getGradient(self: *const Variable) ?*const Tensor {
        return self.node.gradient;
    }
};

// Enhanced loss functions with regularization
pub const LossFunctions = struct {
    pub fn mseLoss(pred: *const Tensor, target: *const Tensor) TensorError!f32 {
        var diff = try pred.sub(target);
        defer diff.deinit();
        var squared = try diff.square();
        defer squared.deinit();
        return squared.sum() / @as(f32, @floatFromInt(squared.data.len));
    }

    pub fn mseLossWithL1(pred: *const Tensor, target: *const Tensor, weights: *const Tensor, l1_lambda: f32) TensorError!f32 {
        const mse = try LossFunctions.mseLoss(pred, target);
        var l1_penalty: f32 = 0.0;
        for (weights.data) |w| {
            l1_penalty += @abs(w);
        }
        return mse + l1_lambda * l1_penalty;
    }

    pub fn mseLossWithL2(pred: *const Tensor, target: *const Tensor, weights: *const Tensor, l2_lambda: f32) TensorError!f32 {
        const mse = try LossFunctions.mseLoss(pred, target);
        var l2_penalty: f32 = 0.0;
        for (weights.data) |w| {
            l2_penalty += w * w;
        }
        return mse + l2_lambda * l2_penalty * 0.5;
    }

    pub fn crossEntropyLoss(logits: *const Tensor, targets: *const Tensor) TensorError!f32 {
        // Softmax + Cross entropy
        var loss: f32 = 0.0;
        var batch_size: usize = 1;
        const num_classes = logits.shape[logits.shape.len - 1];

        if (logits.shape.len > 1) {
            batch_size = logits.data.len / num_classes;
        }

        for (0..batch_size) |b| {
            const offset = b * num_classes;

            // Find max for numerical stability
            var max_val: f32 = logits.data[offset];
            for (1..num_classes) |i| {
                max_val = @max(max_val, logits.data[offset + i]);
            }

            // Compute softmax
            var sum_exp: f32 = 0.0;
            for (0..num_classes) |i| {
                sum_exp += @exp(logits.data[offset + i] - max_val);
            }

            // Compute cross entropy
            const target_class = @as(usize, @intFromFloat(targets.data[b]));
            loss += -(logits.data[offset + target_class] - max_val) + @log(sum_exp);
        }

        return loss / @as(f32, @floatFromInt(batch_size));
    }

    pub const LossType = enum {
        mse,
        mse_l1,
        mse_l2,
        cross_entropy,
        custom,
    };

    pub const CustomLossFn = *const fn(pred: *const Tensor, target: *const Tensor, context: ?*anyopaque) TensorError!f32;

    pub const LossConfig = struct {
        loss_type: LossType,
        l1_lambda: f32 = 0.0,
        l2_lambda: f32 = 0.0,
        custom_fn: ?CustomLossFn = null,
        context: ?*anyopaque = null,

        pub fn compute(self: *const LossConfig, pred: *const Tensor, target: *const Tensor, weights: ?*const Tensor) TensorError!f32 {
            switch (self.loss_type) {
                .mse => return LossFunctions.mseLoss(pred, target),
                .mse_l1 => {
                    if (weights == null) return TensorError.InvalidInput;
                    return LossFunctions.mseLossWithL1(pred, target, weights.?, self.l1_lambda);
                },
                .mse_l2 => {
                    if (weights == null) return TensorError.InvalidInput;
                    return LossFunctions.mseLossWithL2(pred, target, weights.?, self.l2_lambda);
                },
                .cross_entropy => return LossFunctions.crossEntropyLoss(pred, target),
                .custom => {
                    if (self.custom_fn == null) return TensorError.InvalidInput;
                    return self.custom_fn.?(pred, target, self.context);
                },
            }
        }
    };
};

test "basic add functionality" {
    try std.testing.expect(add(3, 7) == 10);
}

test "tensor init and deinit" {
    const allocator = std.testing.allocator;
    var tensor = try Tensor.init(allocator, &[_]usize{ 2, 3 });
    defer tensor.deinit();
    try std.testing.expectEqual(@as(usize, 6), tensor.data.len);
    try std.testing.expect(std.mem.eql(usize, tensor.shape, &[_]usize{ 2, 3 }));
}

test "tensor add" {
    const allocator = std.testing.allocator;
    var a = try Tensor.init(allocator, &[_]usize{2});
    defer a.deinit();
    var b = try Tensor.init(allocator, &[_]usize{2});
    defer b.deinit();
    a.data[0] = 1.0;
    a.data[1] = 2.0;
    b.data[0] = 3.0;
    b.data[1] = 4.0;
    var c = try a.add(&b);
    defer c.deinit();
    try std.testing.expectEqual(@as(f32, 4.0), c.data[0]);
    try std.testing.expectEqual(@as(f32, 6.0), c.data[1]);
}

test "tensor add with broadcasting" {
    const allocator = std.testing.allocator;
    var a = try Tensor.init(allocator, &[_]usize{ 3, 1 });
    defer a.deinit();
    a.data[0] = 1.0;
    a.data[1] = 2.0;
    a.data[2] = 3.0;
    var b = try Tensor.init(allocator, &[_]usize{1});
    defer b.deinit();
    b.data[0] = 10.0;
    var c = try a.add(&b);
    defer c.deinit();
    try std.testing.expect(std.mem.eql(usize, c.shape, &[_]usize{ 3, 1 }));
    try std.testing.expectEqual(@as(f32, 11.0), c.data[0]);
    try std.testing.expectEqual(@as(f32, 12.0), c.data[1]);
    try std.testing.expectEqual(@as(f32, 13.0), c.data[2]);
}

test "tensor square and sum" {
    const allocator = std.testing.allocator;
    var a = try Tensor.init(allocator, &[_]usize{3});
    defer a.deinit();
    a.data[0] = 1.0;
    a.data[1] = 2.0;
    a.data[2] = 3.0;
    var sq = try a.square();
    defer sq.deinit();
    try std.testing.expectEqual(@as(f32, 1.0), sq.data[0]);
    try std.testing.expectEqual(@as(f32, 4.0), sq.data[1]);
    try std.testing.expectEqual(@as(f32, 9.0), sq.data[2]);
    const total = a.sum();
    try std.testing.expectEqual(@as(f32, 6.0), total);
}

test "tensor transpose" {
    const allocator = std.testing.allocator;
    var a = try Tensor.init(allocator, &[_]usize{ 2, 3 });
    defer a.deinit();
    a.data[0] = 1.0;
    a.data[1] = 2.0;
    a.data[2] = 3.0;
    a.data[3] = 4.0;
    a.data[4] = 5.0;
    a.data[5] = 6.0;
    var t = try a.transpose();
    defer t.deinit();
    try std.testing.expect(std.mem.eql(usize, t.shape, &[_]usize{ 3, 2 }));
    try std.testing.expectEqual(@as(f32, 1.0), t.data[0]);
    try std.testing.expectEqual(@as(f32, 4.0), t.data[1]);
    try std.testing.expectEqual(@as(f32, 2.0), t.data[2]);
    try std.testing.expectEqual(@as(f32, 5.0), t.data[3]);
    try std.testing.expectEqual(@as(f32, 3.0), t.data[4]);
    try std.testing.expectEqual(@as(f32, 6.0), t.data[5]);
}

test "tensor matmul" {
    const allocator = std.testing.allocator;
    var a = try Tensor.init(allocator, &[_]usize{ 2, 2 });
    defer a.deinit();
    a.data[0] = 1.0;
    a.data[1] = 2.0;
    a.data[2] = 3.0;
    a.data[3] = 4.0;
    var b = try Tensor.init(allocator, &[_]usize{ 2, 2 });
    defer b.deinit();
    b.data[0] = 5.0;
    b.data[1] = 6.0;
    b.data[2] = 7.0;
    b.data[3] = 8.0;
    var c = try a.matmul(&b);
    defer c.deinit();
    try std.testing.expect(std.mem.eql(usize, c.shape, &[_]usize{ 2, 2 }));
    try std.testing.expectEqual(@as(f32, 19.0), c.data[0]); // 1*5 + 2*7
    try std.testing.expectEqual(@as(f32, 22.0), c.data[1]); // 1*6 + 2*8
    try std.testing.expectEqual(@as(f32, 43.0), c.data[2]); // 3*5 + 4*7
    try std.testing.expectEqual(@as(f32, 50.0), c.data[3]); // 3*6 + 4*8
}

test "linear layer init and forward" {
    const allocator = std.testing.allocator;
    var layer = try Linear.init(allocator, 3, 2);
    defer layer.deinit();
    var input = try Tensor.init(allocator, &[_]usize{ 1, 3 });
    defer input.deinit();
    input.data[0] = 1.0;
    input.data[1] = 0.0;
    input.data[2] = -1.0;
    var output = try layer.forward(&input);
    defer output.deinit();
    try std.testing.expectEqual(@as(f32, 0.1), output.data[0]);
    try std.testing.expectEqual(@as(f32, 0.1), output.data[1]);
}

test "automatic differentiation - basic operations" {
    const allocator = std.testing.allocator;

    var graph = ComputationGraph.init(allocator);
    defer graph.deinit();

    // Test data
    var x_tensor = try Tensor.init(allocator, &[_]usize{1});
    defer x_tensor.deinit();
    x_tensor.data[0] = 2.0;

    var y_tensor = try Tensor.init(allocator, &[_]usize{1});
    defer y_tensor.deinit();
    y_tensor.data[0] = 3.0;

    // Create variables
    const x = try Variable.init(&graph, &x_tensor, true);
    const y = try Variable.init(&graph, &y_tensor, true);

    // Test addition: z = x + y
    const z = try x.add(&y);
    try std.testing.expectEqual(@as(f32, 5.0), z.tensor.data[0]);

    // Backward pass
    try z.backward();

    // Check gradients (gradient of addition is 1 for both inputs)
    try std.testing.expectEqual(@as(f32, 1.0), x.getGradient().?.data[0]);
    try std.testing.expectEqual(@as(f32, 1.0), y.getGradient().?.data[0]);
}

test "automatic differentiation - multiplication chain rule" {
    const allocator = std.testing.allocator;

    var graph = ComputationGraph.init(allocator);
    defer graph.deinit();

    // Test multiplication chain rule
    var a_tensor = try Tensor.init(allocator, &[_]usize{1});
    defer a_tensor.deinit();
    a_tensor.data[0] = 3.0;

    var b_tensor = try Tensor.init(allocator, &[_]usize{1});
    defer b_tensor.deinit();
    b_tensor.data[0] = 4.0;

    const a = try Variable.init(&graph, &a_tensor, true);
    const b = try Variable.init(&graph, &b_tensor, true);

    // z = a * b = 3 * 4 = 12
    const z = try a.mul(&b);
    try std.testing.expectEqual(@as(f32, 12.0), z.tensor.data[0]);

    try z.backward();

    // dz/da = b = 4, dz/db = a = 3
    try std.testing.expectEqual(@as(f32, 4.0), a.getGradient().?.data[0]);
    try std.testing.expectEqual(@as(f32, 3.0), b.getGradient().?.data[0]);
}

test "automatic differentiation - complex expression" {
    const allocator = std.testing.allocator;

    var graph = ComputationGraph.init(allocator);
    defer graph.deinit();

    // Test f(x, y) = x^2 + 2*x*y + y^2 at x=2, y=3
    var x_tensor = try Tensor.init(allocator, &[_]usize{1});
    defer x_tensor.deinit();
    x_tensor.data[0] = 2.0;

    var y_tensor = try Tensor.init(allocator, &[_]usize{1});
    defer y_tensor.deinit();
    y_tensor.data[0] = 3.0;

    var two_tensor = try Tensor.init(allocator, &[_]usize{1});
    defer two_tensor.deinit();
    two_tensor.data[0] = 2.0;

    const x = try Variable.init(&graph, &x_tensor, true);
    const y = try Variable.init(&graph, &y_tensor, true);
    const two = try Variable.init(&graph, &two_tensor, false);

    // f = x^2 + 2*x*y + y^2
    const x_squared = try x.mul(&x);  // x^2
    const y_squared = try y.mul(&y);  // y^2
    const xy = try x.mul(&y);         // x*y
    const two_xy = try two.mul(&xy);  // 2*x*y
    const temp = try x_squared.add(&two_xy); // x^2 + 2*x*y
    const f = try temp.add(&y_squared);       // x^2 + 2*x*y + y^2

    // f(2,3) = 4 + 12 + 9 = 25
    try std.testing.expectEqual(@as(f32, 25.0), f.tensor.data[0]);

    try f.backward();

    // df/dx = 2*x + 2*y = 2*2 + 2*3 = 10
    // df/dy = 2*x + 2*y = 2*2 + 2*3 = 10
    const x_grad = x.getGradient().?.data[0];
    const y_grad = y.getGradient().?.data[0];

    try std.testing.expect(@abs(x_grad - 10.0) < 1e-5);
    try std.testing.expect(@abs(y_grad - 10.0) < 1e-5);
}

test "loss functions with regularization" {
    const allocator = std.testing.allocator;

    var pred = try Tensor.init(allocator, &[_]usize{3});
    defer pred.deinit();
    pred.data[0] = 1.0;
    pred.data[1] = 2.0;
    pred.data[2] = 3.0;

    var target = try Tensor.init(allocator, &[_]usize{3});
    defer target.deinit();
    target.data[0] = 1.1;
    target.data[1] = 1.9;
    target.data[2] = 3.1;

    var weights = try Tensor.init(allocator, &[_]usize{2});
    defer weights.deinit();
    weights.data[0] = 0.5;
    weights.data[1] = -0.3;

    // Test MSE loss
    const mse = try LossFunctions.mseLoss(&pred, &target);
    try std.testing.expect(mse > 0.0 and mse < 0.1);

    // Test MSE + L1 regularization
    const l1_config = LossFunctions.LossConfig{
        .loss_type = .mse_l1,
        .l1_lambda = 0.1,
    };
    const mse_l1 = try l1_config.compute(&pred, &target, &weights);
    try std.testing.expect(mse_l1 > mse); // Should be higher due to L1 penalty

    // Test MSE + L2 regularization
    const l2_config = LossFunctions.LossConfig{
        .loss_type = .mse_l2,
        .l2_lambda = 0.1,
    };
    const mse_l2 = try l2_config.compute(&pred, &target, &weights);
    try std.testing.expect(mse_l2 > mse); // Should be higher due to L2 penalty
}

test "cross entropy loss" {
    const allocator = std.testing.allocator;

    // Test with simple 2-class case
    var logits = try Tensor.init(allocator, &[_]usize{2});
    defer logits.deinit();
    logits.data[0] = 2.0;  // High confidence for class 0
    logits.data[1] = 0.1;  // Low confidence for class 1

    var targets = try Tensor.init(allocator, &[_]usize{1});
    defer targets.deinit();
    targets.data[0] = 0.0; // True class is 0

    const loss = try LossFunctions.crossEntropyLoss(&logits, &targets);
    try std.testing.expect(loss >= 0.0); // Cross entropy is always non-negative
    try std.testing.expect(loss < 1.0);  // Should be low since prediction is correct
}

test "sequential forward applies layers" {
    const allocator = std.testing.allocator;

    var seq = Sequential.init(allocator);

    const linear_ptr = try allocator.create(Linear);
    linear_ptr.* = try Linear.init(allocator, 2, 1);
    linear_ptr.weights.data[0] = 0.5;
    linear_ptr.weights.data[1] = -0.25;
    linear_ptr.biases.data[0] = 0.75;

    try seq.addLinear(linear_ptr);
    try seq.addActivation(relu);

    defer allocator.destroy(linear_ptr);
    defer seq.deinit();

    var input = try Tensor.init(allocator, &[_]usize{ 1, 2 });
    defer input.deinit();
    input.data[0] = 2.0;
    input.data[1] = 1.0;

    var output = try seq.forward(&input);
    defer output.deinit();

    try std.testing.expect(std.mem.eql(usize, output.shape, &[_]usize{ 1, 1 }));
    try std.testing.expect(@abs(output.data[0] - 1.5) < 1e-6);
}

test "optimizer steps update parameters" {
    const allocator = std.testing.allocator;

    var params = try Tensor.init(allocator, &[_]usize{2});
    defer params.deinit();
    params.data[0] = 1.0;
    params.data[1] = -1.0;

    var grads = try Tensor.init(allocator, &[_]usize{2});
    defer grads.deinit();
    grads.data[0] = 0.2;
    grads.data[1] = -0.4;

    var optimizer = Optimizer.init(.sgd, .{ .learning_rate = 0.5 });
    optimizer.step(&params, &grads, null, null);

    try std.testing.expect(@abs(params.data[0] - 0.9) < 1e-6);
    try std.testing.expect(@abs(params.data[1] - (-0.8)) < 1e-6);

    // Adam step
    var adam_params = try Tensor.init(allocator, &[_]usize{1});
    defer adam_params.deinit();
    adam_params.data[0] = 1.0;

    var adam_grads = try Tensor.init(allocator, &[_]usize{1});
    defer adam_grads.deinit();
    adam_grads.data[0] = 0.1;

    var momentum = try Tensor.init(allocator, &[_]usize{1});
    defer momentum.deinit();
    momentum.data[0] = 0.0;

    var velocity = try Tensor.init(allocator, &[_]usize{1});
    defer velocity.deinit();
    velocity.data[0] = 0.0;

    optimizer = Optimizer.init(.adam, .{ .learning_rate = 0.1 });
    optimizer.step(&adam_params, &adam_grads, &momentum, &velocity);

    try std.testing.expect(@abs(adam_params.data[0] - 0.9) < 1e-5);
    try std.testing.expect(momentum.data[0] > 0.0);
    try std.testing.expect(velocity.data[0] > 0.0);
}

test "learning rate scheduler progression" {
    var scheduler = LearningRateScheduler.init(.step_lr, 0.1, 2, 0.5);
    try std.testing.expect(@abs(scheduler.getLearningRate() - 0.1) < 1e-6);
    scheduler.step();
    try std.testing.expect(@abs(scheduler.getLearningRate() - 0.1) < 1e-6);
    scheduler.step();
    try std.testing.expect(@abs(scheduler.getLearningRate() - 0.05) < 1e-6);

    scheduler = LearningRateScheduler.init(.exponential, 0.2, 1, 0.9);
    try std.testing.expect(@abs(scheduler.getLearningRate() - 0.2) < 1e-6);
    scheduler.step();
    try std.testing.expect(@abs(scheduler.getLearningRate() - 0.18) < 1e-6);
}

test "data augmentation helpers" {
    const allocator = std.testing.allocator;

    var tensor = try Tensor.init(allocator, &[_]usize{4});
    defer tensor.deinit();
    tensor.data[0] = -1.0;
    tensor.data[1] = -0.5;
    tensor.data[2] = 0.5;
    tensor.data[3] = 1.0;

    const noise_level = 0.5;
    var noisy = try DataAugmentation.addNoise(&tensor, noise_level, allocator);
    defer noisy.deinit();
    for (0..tensor.data.len) |i| {
        const frac = @as(f32, @floatFromInt(i)) / @as(f32, @floatFromInt(tensor.data.len));
        const expected = tensor.data[i] + (frac - 0.5) * 2.0 * noise_level;
        try std.testing.expect(@abs(noisy.data[i] - expected) < 1e-6);
    }

    var normalized = try DataAugmentation.normalize(&tensor, 0.0, 1.0, allocator);
    defer normalized.deinit();
    for (0..tensor.data.len) |i| {
        try std.testing.expect(@abs(normalized.data[i] - tensor.data[i]) < 1e-6);
    }

    var scaled = try DataAugmentation.scale(&tensor, -1.0, 1.0, allocator);
    defer scaled.deinit();
    try std.testing.expect(@abs(scaled.data[0] - -1.0) < 1e-6);
    try std.testing.expect(@abs(scaled.data[3] - 1.0) < 1e-6);
}
