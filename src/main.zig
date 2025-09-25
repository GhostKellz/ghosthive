const std = @import("std");
const ghosthive = @import("ghosthive");

pub fn main() !void {
    var gpa = std.heap.GeneralPurposeAllocator(.{}){};
    defer _ = gpa.deinit();
    const allocator = gpa.allocator();

    // Data: y = 2x + 1
    var x = try ghosthive.Tensor.init(allocator, &[_]usize{ 4, 1 });
    defer x.deinit();
    x.data[0] = 1.0;
    x.data[1] = 2.0;
    x.data[2] = 3.0;
    x.data[3] = 4.0;

    var y = try ghosthive.Tensor.init(allocator, &[_]usize{ 4, 1 });
    defer y.deinit();
    y.data[0] = 3.0;
    y.data[1] = 5.0;
    y.data[2] = 7.0;
    y.data[3] = 9.0;

    var layer = try ghosthive.Linear.init(allocator, 1, 1);
    defer layer.deinit();

    try ghosthive.trainLinear(allocator, &layer, &x, &y, 0.01, 100);

    std.debug.print("Trained weights: {d}, bias: {d}\n", .{ layer.weights.data[0], layer.biases.data[0] });

    // Test new activation functions
    std.debug.print("\nTesting new activation functions...\n", .{});
    var relu_result = try ghosthive.relu(&x);
    defer relu_result.deinit();

    var tanh_result = try ghosthive.tanh(&x);
    defer tanh_result.deinit();

    var gelu_result = try ghosthive.gelu(&x);
    defer gelu_result.deinit();

    std.debug.print("RELU(x[0]): {d}\n", .{relu_result.data[0]});
    std.debug.print("Tanh(x[0]): {d}\n", .{tanh_result.data[0]});
    std.debug.print("GELU(x[0]): {d}\n", .{gelu_result.data[0]});

    // Test new BETA features
    std.debug.print("\nTesting BETA features...\n", .{});

    // Sequential model test
    var sequential = ghosthive.Sequential.init(allocator);
    defer sequential.deinit();

    // Test multiple optimizers
    const sgd_config = ghosthive.OptimizerConfig{ .learning_rate = 0.01, .momentum = 0.9 };
    const sgd_optimizer = ghosthive.Optimizer.init(.sgd, sgd_config);

    const adamw_config = ghosthive.OptimizerConfig{ .learning_rate = 0.001, .weight_decay = 0.01 };
    const adamw_optimizer = ghosthive.Optimizer.init(.adamw, adamw_config);

    std.debug.print("SGD step count: {d}\n", .{sgd_optimizer.step_count});
    std.debug.print("AdamW step count: {d}\n", .{adamw_optimizer.step_count});

    // Test learning rate scheduler
    var lr_scheduler = ghosthive.LearningRateScheduler.init(.step_lr, 0.1, 10, 0.5);
    std.debug.print("Initial LR: {d:.4}\n", .{lr_scheduler.getLearningRate()});
    lr_scheduler.step();
    std.debug.print("After 1 step LR: {d:.4}\n", .{lr_scheduler.getLearningRate()});

    // Test data augmentation
    var noisy_x = try ghosthive.DataAugmentation.addNoise(&x, 0.1, allocator);
    defer noisy_x.deinit();

    var normalized_x = try ghosthive.DataAugmentation.normalize(&x, 2.5, 1.0, allocator);
    defer normalized_x.deinit();

    std.debug.print("Original x[0]: {d}, Noisy: {d}, Normalized: {d}\n", .{ x.data[0], noisy_x.data[0], normalized_x.data[0] });

    // Test GPU infrastructure
    std.debug.print("\nTesting GPU infrastructure...\n", .{});

    var device_manager = ghosthive.DeviceManager.init(allocator);
    defer device_manager.deinit();

    try device_manager.detectDevices();
    const devices = device_manager.listDevices();

    std.debug.print("Available devices:\n", .{});
    for (devices) |device| {
        std.debug.print("  - {s} ({s})\n", .{ device.name, @tagName(device.device_type) });
    }

    // Test CPU device operations
    if (device_manager.getCurrentDevice()) |cpu_device| {
        std.debug.print("Using device: {s}\n", .{cpu_device.name});

        var gpu_x = try ghosthive.Tensor.initOnDevice(allocator, &[_]usize{4}, cpu_device);
        defer gpu_x.deinit();
        var gpu_y = try ghosthive.Tensor.initOnDevice(allocator, &[_]usize{4}, cpu_device);
        defer gpu_y.deinit();

        // Copy data to GPU tensors
        @memcpy(gpu_x.data, x.data);
        @memcpy(gpu_y.data, y.data);

        // Test GPU operations (will fallback to optimized CPU for now)
        var gpu_add_result = try gpu_x.addGpu(&gpu_y);
        defer gpu_add_result.deinit();

        var gpu_mul_result = try gpu_x.mulGpu(&gpu_y);
        defer gpu_mul_result.deinit();

        std.debug.print("GPU Add result[0]: {d}\n", .{gpu_add_result.data[0]});
        std.debug.print("GPU Mul result[0]: {d}\n", .{gpu_mul_result.data[0]});
    }

    // Run basic benchmarks
    std.debug.print("\nRunning GhostHive benchmarks...\n", .{});
    try ghosthive.runBenchmarks(allocator, &ghosthive.default_benchmarks);

    // Test automatic differentiation system
    std.debug.print("\nTesting Automatic Differentiation (RC1)...\n", .{});

    var graph = ghosthive.ComputationGraph.init(allocator);
    defer graph.deinit();

    // Create tensors for autograd
    var w_tensor = try ghosthive.Tensor.init(allocator, &[_]usize{1});
    defer w_tensor.deinit();
    w_tensor.data[0] = 2.0;

    var b_tensor = try ghosthive.Tensor.init(allocator, &[_]usize{1});
    defer b_tensor.deinit();
    b_tensor.data[0] = 1.0;

    var x_auto_tensor = try ghosthive.Tensor.init(allocator, &[_]usize{1});
    defer x_auto_tensor.deinit();
    x_auto_tensor.data[0] = 3.0;

    var target_auto_tensor = try ghosthive.Tensor.init(allocator, &[_]usize{1});
    defer target_auto_tensor.deinit();
    target_auto_tensor.data[0] = 7.0; // Target: 2*3 + 1 = 7

    // Create Variables for autograd
    const w_auto = try ghosthive.Variable.init(&graph, &w_tensor, true);
    const b_auto = try ghosthive.Variable.init(&graph, &b_tensor, true);
    const x_auto = try ghosthive.Variable.init(&graph, &x_auto_tensor, false);
    const target_auto = try ghosthive.Variable.init(&graph, &target_auto_tensor, false);

    // Forward pass: y = w * x + b
    const wx = try w_auto.mul(&x_auto);
    const y_auto = try wx.add(&b_auto);

    // Compute loss: loss = (y - target)^2 / 2
    const loss = try y_auto.mseLoss(&target_auto);

    std.debug.print("Forward pass: w={d:.3}, x={d:.3}, b={d:.3} -> y={d:.3}\n", .{
        w_auto.tensor.data[0], x_auto.tensor.data[0], b_auto.tensor.data[0], y_auto.tensor.data[0]
    });
    std.debug.print("Target: {d:.3}, Loss: {d:.6}\n", .{ target_auto.tensor.data[0], loss.tensor.data[0] });

    // Backward pass
    try loss.backward();

    std.debug.print("Gradients after backprop:\n", .{});
    if (w_auto.getGradient()) |w_grad| {
        std.debug.print("  dw = {d:.6}\n", .{w_grad.data[0]});
    }
    if (b_auto.getGradient()) |b_grad| {
        std.debug.print("  db = {d:.6}\n", .{b_grad.data[0]});
    }

    // Test multiple loss functions
    std.debug.print("\nTesting Loss Functions with Regularization...\n", .{});

    const loss_config_l1 = ghosthive.LossFunctions.LossConfig{
        .loss_type = .mse_l1,
        .l1_lambda = 0.01,
    };
    const l1_loss = try loss_config_l1.compute(y_auto.tensor, target_auto.tensor, w_auto.tensor);
    std.debug.print("MSE + L1 loss: {d:.6}\n", .{l1_loss});

    const loss_config_l2 = ghosthive.LossFunctions.LossConfig{
        .loss_type = .mse_l2,
        .l2_lambda = 0.01,
    };
    const l2_loss = try loss_config_l2.compute(y_auto.tensor, target_auto.tensor, w_auto.tensor);
    std.debug.print("MSE + L2 loss: {d:.6}\n", .{l2_loss});

    // Test fluent API design
    std.debug.print("\nTesting Fluent API Design...\n", .{});

    // Fluent tensor creation
    var fluent_tensor = try ghosthive.TensorBuilder.init(allocator)
        .withShape(&[_]usize{2, 2})
        .zeros()
        .build();
    defer fluent_tensor.deinit();

    fluent_tensor.data[0] = 1.0;
    fluent_tensor.data[1] = 2.0;
    fluent_tensor.data[2] = 3.0;
    fluent_tensor.data[3] = 4.0;

    std.debug.print("Original tensor: [{d:.1}, {d:.1}, {d:.1}, {d:.1}]\n",
        .{ fluent_tensor.data[0], fluent_tensor.data[1], fluent_tensor.data[2], fluent_tensor.data[3] });

    // Fluent chained operations (in-place modifications)
    _ = try (try (try fluent_tensor.scale_(2.0)).relu_()).clamp_(0.0, 5.0);

    std.debug.print("After scale(2.0).relu().clamp(0,5): [{d:.1}, {d:.1}, {d:.1}, {d:.1}]\n",
        .{ fluent_tensor.data[0], fluent_tensor.data[1], fluent_tensor.data[2], fluent_tensor.data[3] });

    // Create another tensor using fluent builder with random values
    var random_tensor = try ghosthive.TensorBuilder.init(allocator)
        .withShape(&[_]usize{3})
        .random(42)
        .build();
    defer random_tensor.deinit();

    std.debug.print("Random tensor: [{d:.3}, {d:.3}, {d:.3}]\n",
        .{ random_tensor.data[0], random_tensor.data[1], random_tensor.data[2] });

    // Test enhanced error messages
    std.debug.print("\nTesting Enhanced Error Messages...\n", .{});

    // Create tensors with incompatible shapes to demonstrate error messages
    var incompatible_tensor = try ghosthive.TensorBuilder.init(allocator)
        .withShape(&[_]usize{3, 2})
        .ones()
        .build();
    defer incompatible_tensor.deinit();

    std.debug.print("Attempting to add tensors with incompatible shapes:\n", .{});
    std.debug.print("  Tensor A: [2, 2], Tensor B: [3, 2]\n", .{});

    // This should produce a detailed error message
    const add_result = fluent_tensor.add(&incompatible_tensor);
    if (add_result) |result| {
        var res = result;
        defer res.deinit();
        std.debug.print("  Unexpectedly succeeded!\n", .{});
    } else |err| {
        std.debug.print("  Caught error: {}\n", .{err});
    }

    // Test type-safe tensor operations
    std.debug.print("\nTesting Type-Safe Tensor Operations...\n", .{});

    // Create type-safe tensors with compile-time shape verification
    const Matrix2x3 = ghosthive.TypedTensor(&[_]usize{2, 3});
    const Vector3 = ghosthive.TypedTensor(&[_]usize{3});

    var typed_matrix = try Matrix2x3.init(allocator);
    defer typed_matrix.deinit();

    var typed_vector = try Vector3.init(allocator);
    defer typed_vector.deinit();

    // Type-safe element access with bounds checking
    try typed_matrix.set([_]usize{0, 0}, 1.0);
    try typed_matrix.set([_]usize{0, 1}, 2.0);
    try typed_matrix.set([_]usize{0, 2}, 3.0);
    try typed_matrix.set([_]usize{1, 0}, 4.0);
    try typed_matrix.set([_]usize{1, 1}, 5.0);
    try typed_matrix.set([_]usize{1, 2}, 6.0);

    typed_vector.fill(2.0);

    const val_00 = try typed_matrix.get([_]usize{0, 0});
    const val_11 = try typed_matrix.get([_]usize{1, 1});

    std.debug.print("Type-safe matrix[0,0]: {d:.1}\n", .{val_00});
    std.debug.print("Type-safe matrix[1,1]: {d:.1}\n", .{val_11});

    // Type-safe operations - can only add tensors of the same type
    const AnotherMatrix2x3 = ghosthive.TypedTensor(&[_]usize{2, 3});
    var another_matrix = try AnotherMatrix2x3.init(allocator);
    defer another_matrix.deinit();
    another_matrix.ones();

    var sum_result = try typed_matrix.add(&another_matrix);
    defer sum_result.deinit();

    const sum_val = try sum_result.get([_]usize{0, 0});
    std.debug.print("Type-safe addition result[0,0]: {d:.1}\n", .{sum_val});

    // Demonstrate compile-time bounds checking would prevent this:
    // const invalid_access = try typed_matrix.get([_]usize{5, 5}); // This would be caught at runtime

    // Test debug mode with shape checking
    std.debug.print("\nTesting Debug Mode with Shape Checking...\n", .{});

    // Enable debug mode for development
    ghosthive.setDebugMode(ghosthive.DebugConfig.development);

    std.debug.print("Debug mode enabled with full development features.\n", .{});

    // Create debug tensors with tracking
    var debug_tensor1 = try ghosthive.DebugTensor.init(allocator, &[_]usize{2, 2}, @src());
    defer debug_tensor1.deinit();

    var debug_tensor2 = try ghosthive.DebugTensor.init(allocator, &[_]usize{2, 2}, @src());
    defer debug_tensor2.deinit();

    // Test debug operations with logging
    try debug_tensor1.debugSet(0, 1.5, @src());
    try debug_tensor1.debugSet(1, 2.5, @src());

    const value = try debug_tensor1.debugGet(0, @src());
    std.debug.print("Retrieved value from debug tensor: {d:.1}\n", .{value});

    // Test debug addition with shape checking
    var debug_result = try debug_tensor1.debugAdd(&debug_tensor2, @src());
    defer debug_result.deinit();

    // Get and display statistics
    const stats1 = debug_tensor1.getStats();
    const stats_result = debug_result.getStats();

    std.debug.print("Debug tensor 1 stats: {d} operations, last: {s}\n", .{stats1.operations, stats1.last_op orelse "none"});
    std.debug.print("Debug result stats: {d} operations, created at line {d}\n", .{stats_result.operations, stats_result.created_at.line});

    // Disable debug mode
    ghosthive.setDebugMode(ghosthive.DebugConfig.default);
    std.debug.print("Debug mode disabled.\n", .{});

    std.debug.print("\nGhostHive RC1 Developer Experience Features Complete!\n", .{});
    std.debug.print("✅ Fluent API Design\n", .{});
    std.debug.print("✅ Enhanced Error Messages\n", .{});
    std.debug.print("✅ Type-Safe Tensor Operations\n", .{});
    std.debug.print("✅ Debug Mode with Shape Checking\n", .{});

    std.debug.print("\nGhostHive RC1 with Complete Developer Experience successfully implemented!\n", .{});
}

test "simple test" {
    const gpa = std.testing.allocator;
    var list: std.ArrayList(i32) = .empty;
    defer list.deinit(gpa); // Try commenting this out and see if zig detects the memory leak!
    try list.append(gpa, 42);
    try std.testing.expectEqual(@as(i32, 42), list.pop());
}

test "fuzz example" {
    const Context = struct {
        fn testOne(context: @This(), input: []const u8) anyerror!void {
            _ = context;
            // Try passing `--fuzz` to `zig build test` and see if it manages to fail this test case!
            try std.testing.expect(!std.mem.eql(u8, "canyoufindme", input));
        }
    };
    try std.testing.fuzz(Context{}, Context.testOne, .{});
}
