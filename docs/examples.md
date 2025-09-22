# GhostHive Examples

## Linear Regression

```zig
const std = @import("std");
const ghosthive = @import("ghosthive");

pub fn main() !void {
    var gpa = std.heap.GeneralPurposeAllocator(.{}){};
    defer _ = gpa.deinit();
    const allocator = gpa.allocator();

    // Data: y = 2x + 1
    var x = try ghosthive.Tensor.init(allocator, &[_]usize{4, 1});
    defer x.deinit();
    x.data[0] = 1.0; x.data[1] = 2.0; x.data[2] = 3.0; x.data[3] = 4.0;

    var y = try ghosthive.Tensor.init(allocator, &[_]usize{4, 1});
    defer y.deinit();
    y.data[0] = 3.0; y.data[1] = 5.0; y.data[2] = 7.0; y.data[3] = 9.0;

    var layer = try ghosthive.Linear.init(allocator, 1, 1);
    defer layer.deinit();

    try ghosthive.trainLinear(allocator, &layer, &x, &y, 0.01, 100);

    std.debug.print("Trained weights: {d}, bias: {d}\n", .{layer.weights.data[0], layer.biases.data[0]});
}
```

## Tensor Operations

```zig
// Broadcasting example
var a = try Tensor.init(allocator, &[_]usize{3, 1});
a.data[0] = 1; a.data[1] = 2; a.data[2] = 3;
var b = try Tensor.init(allocator, &[_]usize{1});
b.data[0] = 10;
var c = try a.add(&b); // [11, 12, 13]
```