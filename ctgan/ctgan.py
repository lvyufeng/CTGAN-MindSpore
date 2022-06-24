import time
import pandas as pd
import numpy as np
import mindspore
import mindspore.nn as nn
import mindspore.ops as ops
import mindspore.numpy as mnp
from mindspore import Tensor, ms_function
from .data_modules import DataTransformer, DataSampler
from .modules import Generator, Discriminator, conditional_loss
from .grad import value_and_grad, grad
from .amp import DynamicLossScale, NoLossScale, all_finite, auto_mixed_precision

class CTGANSynthesizer(nn.Cell):
    def __init__(self, embedding_dim=128, generator_dim=(256, 256), discriminator_dim=(256, 256),
                 generator_lr=2e-4, generator_decay=1e-6, discriminator_lr=2e-4,
                 discriminator_decay=1e-6, batch_size=500, discriminator_steps=1,
                 log_frequency=True, verbose=False, epochs=300, pac=10, amp=False):
        super().__init__()
        assert batch_size % 2 == 0

        self._embedding_dim = embedding_dim
        self._generator_dim = generator_dim
        self._discriminator_dim = discriminator_dim

        self._generator_lr = generator_lr
        self._generator_decay = generator_decay
        self._discriminator_lr = discriminator_lr
        self._discriminator_decay = discriminator_decay

        self._batch_size = batch_size
        self._discriminator_steps = discriminator_steps
        self._log_frequency = log_frequency
        self._verbose = verbose
        self._epochs = epochs
        self.pac = pac
        self.amp = amp

        self._transformer = None
        self._data_sampler = None
        self._generator = None

    def discriminator_with_loss_scale(self, inputs):
        outputs = self.discriminator(inputs)
        return self.loss_scale_D.scale(outputs)

    def calc_gradient_penalty(self, real_data, fake_data, pac=10, lambda_=10):
        """Compute the gradient penalty."""
        alpha = ops.StandardNormal()((real_data.shape[0] // pac, 1, 1))
        alpha = ops.tile(alpha, (1, pac, real_data.shape[1]))
        alpha = alpha.view(-1, real_data.shape[1])

        interpolates = alpha * real_data + ((1 - alpha) * fake_data)

        (gradients,) = self.grad_fn(interpolates)
        if self.amp:
            gradients = self.loss_scale_D.unscale(gradients)

        gradients_view = gradients.view(-1, pac * real_data.shape[1])
        gradients_view = mnp.norm(gradients_view, 2, 1) - 1
        gradient_penalty = ((gradients_view) ** 2).mean() * lambda_

        return gradient_penalty

    def discriminator_forward(self, fakez, real, c1, c2):
        _, fakeact = self.generator(fakez)
        if c1 is not None:
            fake_cat = mnp.concatenate([fakeact, c1], axis=1)
            real_cat = mnp.concatenate([real, c2], axis=1)
        else:
            real_cat = real
            fake_cat = fakeact

        y_fake = self.discriminator(fake_cat)
        y_real = self.discriminator(real_cat)

        g_p = self.calc_gradient_penalty(real_cat, fake_cat, self.pac)
        loss_d = -(ops.reduce_mean(y_real) - ops.reduce_mean(y_fake))            

        c_loss = loss_d + g_p
        if self.amp:
            c_loss = self.loss_scale_D.scale(c_loss)
        return c_loss, loss_d

    def generator_forward(self, fake_z, c1, m1, cond_info):
        fake, fakeact = self.generator(fake_z)

        if c1 is not None:
            y_fake = self.discriminator(mnp.concatenate([fakeact, c1], axis=1))
        else:
            y_fake = self.discriminator(fakeact)

        if c1 is None:
            cross_entropy = 0
        else:
            cross_entropy = conditional_loss(cond_info, fake, c1, m1)

        loss_g = -ops.reduce_mean(y_fake) + cross_entropy
        if self.amp:
            loss_g = self.loss_scale_G.scale(loss_g)
        return loss_g

    def fit(self, train_data, discrete_columns=()):
        self._validate_discrete_columns(train_data, discrete_columns)
        self._transformer = DataTransformer()
        self._transformer.fit(train_data, discrete_columns)
        train_data = self._transformer.transform(train_data)
        self._transformer.generate_tensors()

        self._data_sampler = DataSampler(train_data,
                                         self._transformer.output_info_list,
                                         self._log_frequency)

        data_dim = int(self._transformer.output_dimensions)

        self.generator = Generator(self._embedding_dim + self._data_sampler.dim_cond_vec(),
                                   self._generator_dim,
                                   data_dim,
                                   self._transformer.output_tensor,
                                   tau=0.2)
        self.discriminator = Discriminator(data_dim + self._data_sampler.dim_cond_vec(),
                                      self._discriminator_dim,
                                      pac=self.pac)


        if self.amp:
            self.loss_scale_G = DynamicLossScale(1024, 2, 2000)
            self.loss_scale_D = DynamicLossScale(1024, 2, 2000)
            auto_mixed_precision(self.generator)
            auto_mixed_precision(self.discriminator)

        self.generator.set_train(True)
        self.discriminator.set_train(True)

        self.optimizer_G = nn.Adam(self.generator.trainable_params(),
                              learning_rate=self._generator_lr,
                              beta1=0.5, beta2=0.9,
                              weight_decay=self._generator_decay)

        self.optimizer_D = nn.Adam(self.discriminator.trainable_params(),
                              learning_rate=self._discriminator_lr,
                              beta1=0.5, beta2=0.9,
                              weight_decay=self._discriminator_decay)

        self.discriminator_grad_fn = value_and_grad(self.discriminator_forward, self.discriminator.trainable_params(), has_aux=True)
        self.generator_grad_fn = value_and_grad(self.generator_forward, self.generator.trainable_params())
        if self.amp:
            self.grad_fn = grad(self.discriminator_with_loss_scale)
        else:
            self.grad_fn = grad(self.discriminator)

        steps_per_epoch = max(len(train_data) // self._batch_size, 1)
        for i in range(self._epochs):
            for id_ in range(steps_per_epoch):
                cost_d = 0
                for n in range(self._discriminator_steps):
                    s_d = time.time()
                    fakez = np.random.normal(size=(self._batch_size, self._embedding_dim))

                    condvec = self._data_sampler.sample_condvec(self._batch_size)
                    if condvec is None:
                        c1, m1, col, opt = None, None, None, None
                        real = self._data_sampler.sample_data(self._batch_size, col, opt)
                        c2 = None
                    else:
                        c1, m1, col, opt = condvec
                        fakez = np.concatenate([fakez, c1], axis=1)

                        perm = np.arange(self._batch_size)
                        np.random.shuffle(perm)
                        real = self._data_sampler.sample_data(
                            self._batch_size, col[perm], opt[perm])
                        c2 = c1[perm]
                        c1 = Tensor(c1)
                        c2 = Tensor(c2)

                    fakez = Tensor(fakez, mindspore.float32)
                    real = Tensor(real, mindspore.float32)

                    loss_d = self.train_discriminator(fakez, real, c1, c2)
                    t_d = time.time()
                    cost_d += (t_d - s_d)
                
                cost_d /= self._discriminator_steps
                
                fakez = np.random.normal(size=(self._batch_size, self._embedding_dim))
                condvec = self._data_sampler.sample_condvec(self._batch_size)

                if condvec is None:
                    c1, m1, col, opt = None, None, None, None
                else:
                    c1, m1, col, opt = condvec
                    fakez = np.concatenate([fakez, c1], axis=1)
                    c1 = Tensor(c1)
                    m1 = Tensor(m1)
                
                fakez = Tensor(fakez, mindspore.float32)

                s_g = time.time()
                loss_g = self.train_generator(fakez, c1, m1, self._transformer.cond_tensor)
                t_g = time.time()
                cost_g = t_g - s_g

                if self._verbose:
                    print(f'Epoch {i+1}, Loss G: {loss_g.asnumpy(): .4f},'  # noqa: T001
                        f'Loss D: {loss_d.asnumpy(): .4f}, '
                        f'Cost D: {cost_d: .6f}, Cost G: {cost_g: .6f}',
                        flush=True)

    @ms_function
    def train_discriminator(self, fakez, real, c1, c2):
        (_, (loss_d,)), grads = self.discriminator_grad_fn(fakez, real, c1, c2)
        if self.amp:
            grads_finite = all_finite(grads)
            self.loss_scale_D.adjust(grads_finite)
            if grads_finite:
                grads = self.loss_scale_D.unscale(grads)
                loss_d = ops.depend(loss_d, self.optimizer_D(grads))
        else:
            self.optimizer_D(grads)
        return loss_d

    @ms_function
    def train_generator(self, fake_z, c1, m1, cond_info):
        loss_g, grads = self.generator_grad_fn(fake_z, c1, m1, cond_info)
        if self.amp:
            loss_g = self.loss_scale_G.unscale(loss_g)
            grads_finite = all_finite(grads)
            self.loss_scale_G.adjust(grads_finite)
            if grads_finite:
                grads = self.loss_scale_G.unscale(grads)
                loss_g = ops.depend(loss_g, self.optimizer_G(grads))
        else:
            self.optimizer_G(grads)
        return loss_g

    def sample(self, n, condiction_column, condition_value=None):
        pass

    def _validate_discrete_columns(self, train_data, discrete_columns):
        """Check whether ``discrete_columns`` exists in ``train_data``.
        Args:
            train_data (numpy.ndarray or pandas.DataFrame):
                Training Data. It must be a 2-dimensional numpy array or a pandas.DataFrame.
            discrete_columns (list-like):
                List of discrete columns to be used to generate the Conditional
                Vector. If ``train_data`` is a Numpy array, this list should
                contain the integer indices of the columns. Otherwise, if it is
                a ``pandas.DataFrame``, this list should contain the column names.
        """
        if isinstance(train_data, pd.DataFrame):
            invalid_columns = set(discrete_columns) - set(train_data.columns)
        elif isinstance(train_data, np.ndarray):
            invalid_columns = []
            for column in discrete_columns:
                if column < 0 or column >= train_data.shape[1]:
                    invalid_columns.append(column)
        else:
            raise TypeError('``train_data`` should be either pd.DataFrame or np.array.')

        if invalid_columns:
            raise ValueError(f'Invalid columns found: {invalid_columns}')