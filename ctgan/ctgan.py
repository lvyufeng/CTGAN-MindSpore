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

class CTGANSynthesizer():
    def __init__(self, embedding_dim=128, generator_dim=(256, 256), discriminator_dim=(256, 256),
                 generator_lr=2e-4, generator_decay=1e-6, discriminator_lr=2e-4,
                 discriminator_decay=1e-6, batch_size=500, discriminator_steps=1,
                 log_frequency=True, verbose=False, epochs=300, pac=10):

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

        self._transformer = None
        self._data_sampler = None
        self._generator = None

    def fit(self, train_data, discrete_columns=()):
        self._validate_discrete_columns(train_data, discrete_columns)
        self._transformer = DataTransformer()
        self._transformer.fit(train_data, discrete_columns)
        train_data = self._transformer.transform(train_data)
        self._transformer.generate_tensors()

        pac = self.pac

        self._data_sampler = DataSampler(train_data,
                                         self._transformer.output_info_list,
                                         self._log_frequency)

        data_dim = int(self._transformer.output_dimensions)

        generator = Generator(self._embedding_dim + self._data_sampler.dim_cond_vec(),
                                    self._generator_dim,
                                    data_dim,
                                    self._transformer.output_tensor,
                                    tau=0.2)
        generator.update_parameters_name('generator')
        generator.set_train(True)

        discriminator = Discriminator(data_dim + self._data_sampler.dim_cond_vec(),
                                      self._discriminator_dim,
                                      pac=self.pac)
        discriminator.update_parameters_name('discriminator')
        discriminator.set_train(True)

        optimizer_G = nn.Adam(generator.trainable_params(),
                              learning_rate=self._generator_lr,
                              beta1=0.5, beta2=0.9,
                              weight_decay=self._generator_decay)
        optimizer_G.update_parameters_name('optimizer_G')

        optimizer_D = nn.Adam(discriminator.trainable_params(),
                              learning_rate=self._discriminator_lr,
                              beta1=0.5, beta2=0.9,
                              weight_decay=self._discriminator_decay)
        optimizer_D.update_parameters_name('optimizer_D')

        def calc_gradient_penalty(real_data, fake_data, pac=10, lambda_=10):
            """Compute the gradient penalty."""
            alpha = ops.StandardNormal()((real_data.shape[0] // pac, 1, 1))
            alpha = ops.tile(alpha, (1, pac, real_data.shape[1]))
            alpha = alpha.view(-1, real_data.shape[1])

            interpolates = alpha * real_data + ((1 - alpha) * fake_data)

            grad_fn = grad(discriminator)

            (gradients,) = grad_fn(interpolates)

            gradients_view = gradients.view(-1, pac * real_data.shape[1])
            gradients_view = mnp.norm(gradients_view, 2, 1) - 1
            gradient_penalty = ((gradients_view) ** 2).mean() * lambda_

            return gradient_penalty

        def discriminator_forward(fakez, real, c1, c2):
            _, fakeact = generator(fakez)
            if c1 is not None:
                fake_cat = mnp.concatenate([fakeact, c1], axis=1)
                real_cat = mnp.concatenate([real, c2], axis=1)
            else:
                real_cat = real
                fake_cat = fakeact

            y_fake = discriminator(fake_cat)
            y_real = discriminator(real_cat)

            g_p = calc_gradient_penalty(real_cat, fake_cat, pac)
            loss_d = -(ops.reduce_mean(y_real) - ops.reduce_mean(y_fake))            

            c_loss = loss_d + g_p
            return c_loss, loss_d

        def generator_forward(fake_z, c1, m1, cond_info):
            fake, fakeact = generator(fake_z)

            if c1 is not None:
                y_fake = discriminator(mnp.concatenate([fakeact, c1], axis=1))
            else:
                y_fake = discriminator(fakeact)

            if c1 is None:
                cross_entropy = 0
            else:
                cross_entropy = conditional_loss(cond_info, fake, c1, m1)

            loss_g = -ops.reduce_mean(y_fake) + cross_entropy
            return loss_g

        discriminator_grad_fn = value_and_grad(discriminator_forward, discriminator.trainable_params(), has_aux=True)
        generator_grad_fn = value_and_grad(generator_forward, generator.trainable_params())

        @ms_function
        def train_discriminator(fakez, real, c1, c2):
            (_, (loss_d,)), grads = discriminator_grad_fn(fakez, real, c1, c2)
            optimizer_D(grads)
            return loss_d

        @ms_function
        def train_generator(fake_z, c1, m1, cond_info):
            loss_g, grads = generator_grad_fn(fake_z, c1, m1, cond_info)
            optimizer_G(grads)
            return loss_g

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

                    loss_d = train_discriminator(fakez, real, c1, c2)
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
                loss_g = train_generator(fakez, c1, m1, self._transformer.cond_tensor)
                t_g = time.time()
                cost_g = t_g - s_g

            if self._verbose:
                print(f'Epoch {i+1}, Loss G: {loss_g.asnumpy(): .4f},'  # noqa: T001
                      f'Loss D: {loss_d.asnumpy(): .4f}, '
                      f'Cost D: {cost_d: .6f}, Cost G: {cost_g: .6f}',
                      flush=True)

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