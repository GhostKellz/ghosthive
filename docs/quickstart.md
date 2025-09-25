# GhostHive Quickstart

Kick the tires on GhostHive with a minimal Zig project that trains a tiny model and inspects a forward pass. For a deeper dive into every public function, hop over to the [API reference](./API.md).

## Prerequisites

- Zig 0.16.0-dev (the same toolchain GhostHive is developed against)
- A recent Linux/macOS/Windows environment with a working C toolchain (needed for Zig's libc shims)

## 1. Fetch GhostHive

Inside your project directory, pull GhostHive as a dependency using Zig's package manager:

```bash
zig fetch --save https://github.com/ghostkellz/ghostmind/archive/refs/heads/main.tar.gz
```

This records the package in `build.zig.zon` so it can be imported by name (`@import("ghosthive")`).

## 2. Wire GhostHive into `build.zig`

Add the module to your executable configuration so Zig knows how to find it:

```zig
const ghosthive_mod = b.dependency("ghostmind", .{}).module("ghosthive");
exe.root_module.addImport("ghosthive", ghosthive_mod);
```

## 3. Build a tiny model

The snippet below sets up a single linear layer wrapped in a `Sequential`, runs a forward pass, and performs an optimization step with the SGD optimizer. Save it as `src/main.zig` inside your project.

```zig
const std = @import("std");
const ghosthive = @import("ghosthive");

pub fn main() !void {
    var gpa = std.heap.GeneralPurposeAllocator(.{}){};
    defer _ = gpa.deinit();
    const allocator = gpa.allocator();

    // Build a Sequential model with one Linear layer and a ReLU activation.
    var model = ghosthive.Sequential.init(allocator);
    defer model.deinit();

    const linear_ptr = try allocator.create(ghosthive.Linear);
    defer allocator.destroy(linear_ptr);
    linear_ptr.* = try ghosthive.Linear.init(allocator, 2, 1);
    try model.addLinear(linear_ptr);
    try model.addActivation(ghosthive.relu);

    // Toy input batch (1 x 2) and target.
    var input = try ghosthive.Tensor.init(allocator, &[_]usize{ 1, 2 });
    defer input.deinit();
    input.data[0] = 1.0;
    input.data[1] = -1.0;

    var target = try ghosthive.Tensor.init(allocator, &[_]usize{ 1, 1 });
    defer target.deinit();
    target.data[0] = 0.5;

    // Forward pass.
    var prediction = try model.forward(&input);
    defer prediction.deinit();
    std.debug.print("Prediction before step: {d}\n", .{prediction.data[0]});

    // Optimizer update using vanilla SGD.
    var optimizer = ghosthive.Optimizer.init(.sgd, .{ .learning_rate = 0.1 });

    var gradients = try prediction.sub(&target);
    defer gradients.deinit();

    optimizer.step(&linear_ptr.weights, &gradients, null, null);
    optimizer.step(&linear_ptr.biases, &gradients, null, null);

    std.debug.print("Updated weight: {d}, bias: {d}\n", .{
        linear_ptr.weights.data[0],
        linear_ptr.biases.data[0],
    });
}
```

## 4. Run it

```bash
zig build run
```

You should see the model's initial prediction and the adjusted weights/biases printed to stdout.

## 5. Explore next steps

- Inspect the [API reference](./API.md) for every tensor, optimizer, and scheduler entry point.
- Browse [docs/examples.md](./examples.md) for more end-to-end samples, including regression training.
- Check the repository `TODO.md` for the roadmap and upcoming milestones.

Need richer API docs or a walkthrough of autograd? Let us know by filing an issue on GitHub!
