//! By convention, root.zig is the root source file when making a library.
const std = @import("std");

pub const Tensor = struct {
    data: []f32,
    shape: []const usize,
    allocator: std.mem.Allocator,

    fn computeSize(shape: []const usize) usize {
        var size: usize = 1;
        for (shape) |dim| {
            size *= dim;
        }
        return size;
    }

    pub fn init(allocator: std.mem.Allocator, shape: []const usize) !Tensor {
        const size = computeSize(shape);
        const data = try allocator.alloc(f32, size);
        @memset(data, 0.0);
        return Tensor{
            .data = data,
            .shape = try allocator.dupe(usize, shape),
            .allocator = allocator,
        };
    }

    pub fn deinit(self: *Tensor) void {
        self.allocator.free(self.data);
        self.allocator.free(self.shape);
    }

    pub fn add(self: *const Tensor, other: *const Tensor) !Tensor {
        const bs = broadcastShape(self.shape, other.shape) orelse return error.IncompatibleShapes;
        const new_shape = bs.shape[0..bs.len];
        var result = try Tensor.init(self.allocator, new_shape);
        for (0..result.data.len) |i| {
            const a_idx = broadcastIndex(i, self.shape, new_shape);
            const b_idx = broadcastIndex(i, other.shape, new_shape);
            result.data[i] = self.data[a_idx] + other.data[b_idx];
        }
        return result;
    }

    pub fn mul(self: *const Tensor, other: *const Tensor) !Tensor {
        const bs = broadcastShape(self.shape, other.shape) orelse return error.IncompatibleShapes;
        const new_shape = bs.shape[0..bs.len];
        var result = try Tensor.init(self.allocator, new_shape);
        for (0..result.data.len) |i| {
            const a_idx = broadcastIndex(i, self.shape, new_shape);
            const b_idx = broadcastIndex(i, other.shape, new_shape);
            result.data[i] = self.data[a_idx] * other.data[b_idx];
        }
        return result;
    }

    pub fn sub(self: *const Tensor, other: *const Tensor) !Tensor {
        const bs = broadcastShape(self.shape, other.shape) orelse return error.IncompatibleShapes;
        const new_shape = bs.shape[0..bs.len];
        var result = try Tensor.init(self.allocator, new_shape);
        for (0..result.data.len) |i| {
            const a_idx = broadcastIndex(i, self.shape, new_shape);
            const b_idx = broadcastIndex(i, other.shape, new_shape);
            result.data[i] = self.data[a_idx] - other.data[b_idx];
        }
        return result;
    }

    pub fn square(self: *const Tensor) !Tensor {
        var result = try Tensor.init(self.allocator, self.shape);
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

    pub fn transpose(self: *const Tensor) !Tensor {
        if (self.shape.len != 2) return error.Not2D;
        const rows = self.shape[0];
        const cols = self.shape[1];
        var result = try Tensor.init(self.allocator, &[_]usize{ cols, rows });
        for (0..rows) |i| {
            for (0..cols) |j| {
                result.data[j * rows + i] = self.data[i * cols + j];
            }
        }
        return result;
    }

    pub fn matmul(self: *const Tensor, other: *const Tensor) !Tensor {
        if (self.shape.len != 2 or other.shape.len != 2) return error.Not2D;
        if (self.shape[1] != other.shape[0]) return error.ShapeMismatch;
        const m = self.shape[0];
        const k = self.shape[1];
        const n = other.shape[1];
        var result = try Tensor.init(self.allocator, &[_]usize{ m, n });

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

    pub fn forward(self: *Linear, input: *const Tensor) !Tensor {
        if (input.shape.len != 2 or input.shape[1] != self.weights.shape[0]) return error.ShapeMismatch;
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

pub fn relu(tensor: *const Tensor) !Tensor {
    var result = try Tensor.init(tensor.allocator, tensor.shape);
    for (0..tensor.data.len) |i| {
        result.data[i] = if (tensor.data[i] > 0) tensor.data[i] else 0;
    }
    return result;
}

pub fn sigmoid(tensor: *const Tensor) !Tensor {
    var result = try Tensor.init(tensor.allocator, tensor.shape);
    for (0..tensor.data.len) |i| {
        result.data[i] = 1.0 / (1.0 + std.math.exp(-tensor.data[i]));
    }
    return result;
}

pub fn mseLoss(pred: *const Tensor, target: *const Tensor) !f32 {
    if (!std.mem.eql(usize, pred.shape, target.shape)) return error.ShapeMismatch;
    var diff = try pred.sub(target);
    defer diff.deinit();
    var sq = try diff.square();
    defer sq.deinit();
    const loss = sq.sum() / @as(f32, @floatFromInt(pred.data.len));
    return loss;
}

pub fn trainLinear(allocator: std.mem.Allocator, layer: *Linear, x: *const Tensor, y: *const Tensor, lr: f32, epochs: usize) !void {
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

pub fn bufferedPrint() !void {
    // Stdout is for the actual output of your application, for example if you
    // are implementing gzip, then only the compressed bytes should be sent to
    // stdout, not any debugging messages.
    var stdout_buffer: [1024]u8 = undefined;
    var stdout_writer = std.fs.File.stdout().writer(&stdout_buffer);
    const stdout = &stdout_writer.interface;

    try stdout.print("Run `zig build test` to run the tests.\n", .{});

    try stdout.flush(); // Don't forget to flush!
}

pub fn add(a: i32, b: i32) i32 {
    return a + b;
}

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
