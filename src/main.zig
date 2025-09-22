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
