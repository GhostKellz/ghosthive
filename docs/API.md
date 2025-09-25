# GhostHive API Documentation

Quick links:

- 📘 [Quickstart guide](./quickstart.md) — bootstrap a project and see GhostHive in action.
- 📄 [Examples](./examples.md) — fully worked demos that build on this API surface.

> **Error Handling**
>
> Unless otherwise stated, fallible functions return `TensorError!T`. Check documentation in `src/root.zig` for the extended error enum and helper context.

## Tensor Core

### Construction

```zig
pub fn init(allocator: std.mem.Allocator, shape: []const usize) TensorError!Tensor
pub fn initWithPool(allocator: std.mem.Allocator, shape: []const usize, pool: *TensorPool) TensorError!Tensor
pub fn initOnDevice(allocator: std.mem.Allocator, shape: []const usize, device: *const Device) TensorError!Tensor
```

### Element-wise operations

```zig
pub fn add(self: *const Tensor, other: *const Tensor) TensorError!Tensor
pub fn sub(self: *const Tensor, other: *const Tensor) TensorError!Tensor
pub fn mul(self: *const Tensor, other: *const Tensor) TensorError!Tensor
pub fn square(self: *const Tensor) TensorError!Tensor
pub fn sum(self: *const Tensor) f32
pub fn transpose(self: *const Tensor) TensorError!Tensor
pub fn matmul(self: *const Tensor, other: *const Tensor) TensorError!Tensor
```

### Utilities

```zig
pub fn randomize(self: *Tensor) void
pub fn deinit(self: *Tensor) void
pub fn clone(self: *const Tensor) Tensor
```

## Sequential Models

```zig
pub fn init(allocator: std.mem.Allocator) Sequential
pub fn addLinear(self: *Sequential, layer: *Linear) !void
pub fn addActivation(self: *Sequential, activation_fn: ActivationFn) !void
pub fn addDropout(self: *Sequential, layer: *Dropout) !void
pub fn addBatchNorm1D(self: *Sequential, layer: *BatchNorm1D) !void
pub fn forward(self: *Sequential, input: *const Tensor) TensorError!Tensor
pub fn setTraining(self: *Sequential, training: bool) void
pub fn deinit(self: *Sequential) void
```

> The [Quickstart](./quickstart.md) demonstrates how to build a `Sequential` with linear + activation layers.

## Layers & Activations

```zig
pub fn Linear.init(allocator: std.mem.Allocator, in_features: usize, out_features: usize) TensorError!Linear
pub fn Linear.forward(self: *Linear, input: *const Tensor) TensorError!Tensor
pub fn Linear.deinit(self: *Linear) void

pub fn relu(tensor: *const Tensor) TensorError!Tensor
pub fn sigmoid(tensor: *const Tensor) TensorError!Tensor
pub fn tanh(tensor: *const Tensor) TensorError!Tensor
pub fn gelu(tensor: *const Tensor) TensorError!Tensor
pub fn swish(tensor: *const Tensor, beta: f32) TensorError!Tensor
```

## Optimizers & Scheduling

```zig
pub fn Optimizer.init(optimizer_type: OptimizerType, config: OptimizerConfig) Optimizer
pub fn Optimizer.step(self: *Optimizer, params: *Tensor, gradients: *const Tensor, momentum: ?*Tensor, velocity: ?*Tensor) void
pub fn Optimizer.clipGradients(gradients: *Tensor, max_norm: f32) void

pub fn LearningRateScheduler.init(scheduler_type: SchedulerType, initial_lr: f32, step_size: usize, gamma: f32) LearningRateScheduler
pub fn LearningRateScheduler.getLearningRate(self: *LearningRateScheduler) f32
pub fn LearningRateScheduler.step(self: *LearningRateScheduler) void
```

> See the new unit tests in `src/root.zig` for concrete optimizer and scheduler usage patterns.

## Data Utilities

```zig
pub fn DataAugmentation.addNoise(tensor: *const Tensor, noise_level: f32, allocator: std.mem.Allocator) TensorError!Tensor
pub fn DataAugmentation.normalize(tensor: *const Tensor, mean: f32, std_dev: f32, allocator: std.mem.Allocator) TensorError!Tensor
pub fn DataAugmentation.scale(tensor: *const Tensor, min_val: f32, max_val: f32, allocator: std.mem.Allocator) TensorError!Tensor

pub fn Dataset.init(allocator: std.mem.Allocator) Dataset
pub fn Dataset.addSample(self: *Dataset, data: Tensor, target: Tensor) TensorError!void
pub fn Dataset.deinit(self: *Dataset) void

pub fn DataLoader.init(allocator: std.mem.Allocator, data: []const Tensor, targets: []const Tensor, batch_size: usize, shuffle: bool) TensorError!DataLoader
pub fn DataLoader.nextBatch(self: *DataLoader) ?struct { data: []const Tensor, targets: []const Tensor }
pub fn DataLoader.reset(self: *DataLoader) void
```

## Losses

```zig
pub fn mseLoss(pred: *const Tensor, target: *const Tensor) TensorError!f32
pub fn LossFunctions.mseLoss(pred: *const Tensor, target: *const Tensor) TensorError!f32
pub fn LossFunctions.crossEntropyLoss(logits: *const Tensor, targets: *const Tensor) TensorError!f32
pub fn LossFunctions.LossConfig.compute(self: *const LossConfig, pred: *const Tensor, target: *const Tensor, weights: ?*const Tensor) TensorError!f32
```

---

📥 **Want more?**

- Browse the [examples](./examples.md) for end-to-end scripts.
- Track upcoming milestones in the repository [TODO roadmap](../TODO.md).