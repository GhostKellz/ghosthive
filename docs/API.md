# GhostHive API Documentation

## Tensor

### Initialization
```zig
pub fn init(allocator: std.mem.Allocator, shape: []const usize) !Tensor
```
Creates a new tensor with the given shape, initialized to zeros.

### Operations
```zig
pub fn add(self: *const Tensor, other: *const Tensor) !Tensor
pub fn sub(self: *const Tensor, other: *const Tensor) !Tensor
pub fn mul(self: *const Tensor, other: *const Tensor) !Tensor
pub fn square(self: *const Tensor) !Tensor
pub fn sum(self: *const Tensor) f32
pub fn matmul(self: *const Tensor, other: *const Tensor) !Tensor
pub fn transpose(self: *const Tensor) !Tensor
```

### Utilities
```zig
pub fn randomize(self: *Tensor) void
```

## Linear Layer
```zig
pub fn init(allocator: std.mem.Allocator, in_features: usize, out_features: usize) !Linear
pub fn forward(self: *Linear, input: *const Tensor) !Tensor
pub fn deinit(self: *Linear) void
```

## Activations
```zig
pub fn relu(tensor: *const Tensor) !Tensor
pub fn sigmoid(tensor: *const Tensor) !Tensor
```

## Loss
```zig
pub fn mseLoss(pred: *const Tensor, target: *const Tensor) !f32
```

## Training
```zig
pub fn trainLinear(allocator: std.mem.Allocator, layer: *Linear, x: *const Tensor, y: *const Tensor, lr: f32, epochs: usize) !void
```