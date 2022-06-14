import mindspore.nn as nn
import mindspore.ops as ops
import mindspore.numpy as mnp
from mindspore import ms_function
from .layers import Dense, Residual, ApplyActivation

class Discriminator(nn.Cell):
    """Discriminator for the CTGANSynthesizer."""

    def __init__(self, input_dim, discriminator_dim, pac=10):
        super(Discriminator, self).__init__()
        dim = input_dim * pac
        self.pac = pac
        self.pacdim = dim
        seq = []
        for item in list(discriminator_dim):
            seq += [Dense(dim, item), nn.LeakyReLU(0.2), nn.Dropout(0.5)]
            dim = item

        seq += [Dense(dim, 1)]
        self.seq = nn.SequentialCell(*seq)

    def construct(self, inputs):
        """Apply the Discriminator to the `input_`."""
        return self.seq(inputs.view(-1, self.pacdim))

class Generator(nn.Cell):
    """Generator for the CTGANSynthesizer."""

    def __init__(self, embedding_dim, generator_dim, data_dim, transformer_info, tau):
        super(Generator, self).__init__()
        dim = embedding_dim
        seq = []
        for item in generator_dim:
            seq += [Residual(dim, item)]
            dim += item
        seq.append(ApplyActivation(dim, data_dim, transformer_info, tau))
        self.seq = nn.SequentialCell(*seq)

    def construct(self, inputs):
        """Apply the Generator to the `input_`."""
        data = self.seq(inputs)
        return data

def conditional_loss(cond_info, data, cond, mask):
    # c_loss = ops.zeros_like(mask)
    c_loss_list = ()
    for item in cond_info:
        data_logsoftmax = data[:, item[0]:item[1]]
        cond_vec = cond[:, item[2]: item[3]].argmax(1)
        loss = ops.SparseSoftmaxCrossEntropyWithLogits()(cond_vec, data_logsoftmax)
        loss = loss.reshape(-1, 1)
        # c_loss = mnp.concatenate([c_loss[:, :item[-1]], loss, c_loss[:, item[-1]+1:]], 1)
        c_loss_list += (loss,)
    c_loss = mnp.concatenate(c_loss_list, 1)
    return ops.reduce_sum(c_loss * mask) / mask.shape[0]

