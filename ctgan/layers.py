import math
import mindspore.nn as nn
import mindspore.ops as ops
import mindspore.numpy as mnp
from mindspore.common.initializer import initializer, Normal, Uniform, HeUniform, _calculate_fan_in_and_fan_out

class Dense(nn.Dense):
    def __init__(self, in_channels, out_channels, has_bias=True, activation=None):
        super().__init__(in_channels, out_channels, weight_init='normal', bias_init='zeros', has_bias=has_bias, activation=activation)
        self.reset_parameters()
        
    def reset_parameters(self):
        self.weight.set_data(initializer(HeUniform(math.sqrt(5)), self.weight.shape))
        if self.has_bias:
            fan_in, _ = _calculate_fan_in_and_fan_out(self.weight.shape)
            bound = 1 / math.sqrt(fan_in)
            self.bias.set_data(initializer(Uniform(bound), [self.out_channels]))

class Residual(nn.Cell):
    """Residual layer for the CTGANSynthesizer."""

    def __init__(self, i, o):
        super(Residual, self).__init__()
        self.fc = Dense(i, o)
        self.bn = nn.BatchNorm1d(o)
        self.relu = nn.ReLU()

    def construct(self, inputs):
        """Apply the Residual layer to the `input_`."""
        out = self.fc(inputs)
        out = self.bn(out)
        out = self.relu(out)
        return ops.Concat(1)([out, inputs])

class ApplyActivation(nn.Cell):
    def __init__(self, input_dim, output_dim, transformer_info, tau):
        super().__init__()
        self.input_dim = input_dim
        self.output_dim = output_dim
        self.transformer_info = transformer_info
        self.tau = tau

        self.fc = Dense(input_dim, output_dim)
        self.tanh = nn.Tanh()

    def construct(self, inputs):
        outputs = self.fc(inputs)
        # data_t = ops.zeros_like(outputs)
        data_list = ()
        for idx in self.transformer_info:
            if idx[2] == 0:
                act = self.tanh(outputs[:, idx[0]:idx[1]])
            else:
                act = gumbel_softmax(outputs[:, idx[0]:idx[1]], temperature=self.tau)
            # data_t = ops.Concat(1)([data_t[:, :idx[0]], act, data_t[:, idx[1]:]])
            data_list += (act,)
        data_t = mnp.concatenate(data_list, 1)
        return outputs, data_t

def gumbel_softmax(logits, temperature, hard=False, axis=-1, eps=1e-10):
    uniform_samples = ops.UniformReal()(logits.shape)
    gumbels = -ops.log(-ops.log(uniform_samples + eps) + eps) # ~Gumbel(0, 1)
    gumbels = (logits + gumbels) / temperature
    y_soft = ops.Softmax(axis)(gumbels)

    if hard:
        # Straight through
        index = y_soft.argmax(axis)
        y_hard = ops.OneHot(axis)(index, y_soft.shape[axis], ops.scalar_to_array(1.0), ops.scalar_to_array(0.0))
        ret = ops.stop_gradient(y_hard - y_soft) + y_soft
    else:
        # Reparametrization trick.
        ret = y_soft
    return ret