import mindspore.nn as nn
import mindspore.ops as ops
from mindspore import Tensor, Parameter, context
import mindspore.common.dtype as mstype

"""
For AMP white list
"""
amp_white_list = (
    nn.Dense
)

amp_black_list = (
    nn.BatchNorm1d,
    nn.BatchNorm2d
)

class _OutputTo32(nn.Cell):
    "Wrap cell for amp. Cast network output back to float32"

    def __init__(self, op):
        super().__init__(auto_prefix=False)
        self._op = op

    def construct(self, *x):
        return ops.cast(self._op(*x), mstype.float32)

class _OutputTo16(nn.Cell):
    "Wrap cell for amp. Cast network output back to float32"

    def __init__(self, op):
        super().__init__(auto_prefix=False)
        self._op = op

    def construct(self, *x):
        return ops.cast(self._op(*x), mstype.float16)

def auto_mixed_precision(network, amp_level='O1'):
    if amp_level == 'O0':
        pass
    elif amp_level == 'O1':
        auto_white_list(network)
    elif amp_level == 'O2':
        auto_black_list(network)
    elif amp_level == 'O3':
        network.to_float(mstype.float16)
    else:
        raise ValueError(f"the amp_level '{amp_level}' is not supported.")

def auto_white_list(network, white_list=None):
    if white_list is None:
        white_list = amp_white_list
    cells = network.name_cells()
    change = False
    for name in cells:
        subcell = cells[name]
        if subcell == network:
            continue
        elif isinstance(subcell, white_list):
            network._cells[name] = _OutputTo32(subcell.to_float(mstype.float16))
            change = True
        else:
            auto_white_list(subcell, white_list)

    if isinstance(network, nn.SequentialCell) and change:
        network.cell_list = list(network.cells())

def auto_black_list(network, black_list=None):
    if black_list is None:
        black_list = amp_black_list
    network.to_float(mstype.float16)
    cells = network.name_cells()
    change = False
    for name in cells:
        subcell = cells[name]
        if subcell == network:
            continue
        elif isinstance(subcell, black_list):
            network._cells[name] = _OutputTo16(subcell.to_float(mstype.float32))
            change = True
        else:
            auto_black_list(subcell, black_list)

    if isinstance(network, nn.SequentialCell) and change:
        network.cell_list = list(network.cells())

"""
For Loss Scale
"""
ascend_target = (context.get_context("device_target") == "Ascend")
gpu_target = (context.get_context("device_target") == "GPU")
reciprocal = ops.Reciprocal()

gpu_float_status = ops.FloatStatus()
npu_alloc_float_status = ops.NPUAllocFloatStatus()
npu_clear_float_status = ops.NPUClearFloatStatus()
npu_get_float_status = ops.NPUGetFloatStatus()
if ascend_target:
    status = npu_alloc_float_status()
    _ = npu_clear_float_status(status)
else:
    status = None

hypermap = ops.HyperMap()
partial = ops.Partial()


def grad_unscale(scale, grad):
    return grad * reciprocal(scale).astype(grad.dtype)

def grad_scale(scale, grad):
    return grad * scale.astype(grad.dtype)

def is_finite(inputs):
    if gpu_target:
        return gpu_float_status(inputs)[0] == 0
    status = ops.isfinite(inputs)
    return status.all()

def all_finite(inputs):
    if ascend_target:
        status = ops.depend(status, inputs)
        get_status = npu_get_float_status(status)
        status = ops.depend(status, get_status)
        status_finite = status.sum() == 0
        _ = npu_clear_float_status(status)
        return status_finite
    outputs = hypermap(partial(is_finite), inputs) 
    return ops.stack(outputs).all()


class LossScale(nn.Cell):
    def __init__(self, scale_value, scale_factor, scale_window):
        super().__init__()
        self.scale_factor = scale_factor
        self.scale_window = scale_window

        self.scale_value = Parameter(Tensor(scale_value, dtype=mstype.float32), name="scale_value")
        self.counter = Parameter(Tensor(0, dtype=mstype.int32), name="counter")

    def scale(self, inputs):
        raise NotImplementedError

    def unscale(self, inputs):
        raise NotImplementedError

    def adjust(self, grads_finite):
        raise NotImplementedError

class NoLossScale(LossScale):
    def __init__(self):
        super().__init__(0, 0, 0)

    def scale(self, inputs):
        return inputs

    def unscale(self, inputs):
        return inputs

    def adjust(self, grads_finite):
        return

class StaticLossScale(LossScale):
    def __init__(self, scale_value):
        super().__init__(scale_value, 1, 1)

    def scale(self, inputs):
        return hypermap(partial(grad_scale, self.scale_value), inputs)

    def unscale(self, inputs):
        return hypermap(partial(grad_unscale, self.scale_value), inputs)

    def adjust(self, grads_finite):
        return

class DynamicLossScale(LossScale):
    def __init__(self, scale_value, scale_factor, scale_window):
        super().__init__(scale_value, scale_factor, scale_window)

    def scale(self, inputs):
        return hypermap(partial(grad_scale, self.scale_value), inputs)

    def unscale(self, inputs):
        return hypermap(partial(grad_unscale, self.scale_value), inputs)

    def adjust(self, grads_finite):
        one = ops.ones((), self.scale_value.dtype)
        scale_mul_factor = self.scale_value * self.scale_factor
        scale_value = ops.select(
            grads_finite,
            # When grads are finite increase loss scale periodically.
            ops.select(
                self.counter == (self.scale_window - 1),
                ops.select(is_finite(scale_mul_factor),
                           scale_mul_factor,
                           self.scale_value),
                self.scale_value),
            # If grads are non finite reduce loss scale.
            ops.maximum(one, self.scale_value / self.scale_factor))
        ops.assign(self.scale_value, scale_value)

        counter = ((self.counter + 1) % self.scale_window) * grads_finite
        ops.assign(self.counter, counter)
        return