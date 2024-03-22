# Copyright 2024 Google LLC.
#
# Redistribution and use in source and binary forms, with or without
# modification, are permitted provided that the following conditions are met:
#
# 1. Redistributions of source code must retain the above copyright notice, this
#    list of conditions and the following disclaimer.
#
# 2. Redistributions in binary form must reproduce the above copyright notice,
#    this list of conditions and the following disclaimer in the documentation
#    and/or other materials provided with the distribution.
#
# 3. Neither the name of the copyright holder nor the names of its contributors
#    may be used to endorse or promote products derived from this software
#    without specific prior written permission.
#
# THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS"
# AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE
# IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE ARE
# DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT HOLDER OR CONTRIBUTORS BE LIABLE
# FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR CONSEQUENTIAL
# DAMAGES (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR
# SERVICES; LOSS OF USE, DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER
# CAUSED AND ON ANY THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY,
# OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE
# OF THIS SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.
"""Multimodal representation learning models."""
import dataclasses
from typing import Optional
import tensorflow as tf


@dataclasses.dataclass(eq=False)
class VAE1dConfig:
  """Config for building 1d convolutional autoencoder model.

  Attributes:
    waveform_length: The length of input waveforms.
    waveform_channels_num: The channel num of input waveforms.
    latent_dim: The latent dimension of the autoencoder model.
    conv_kernel_size: The kernel size of the convolutional layers.
    encoder_conv_num_filters: The filter nums for the encoder's conv layers.
    pool_size: The pool size for pooling layers.
    leaky_relu_alpha: The parameter for leaky relu.
    hidden_dense_units: The dense unit nums for hidden dense layers. (All dense
      layers except the last layer, which is equal to latent dim for encoder,
      and will be calculated for decoder.)
    beta: Beta in the VAE model, which is the weight for KL loss.
    learning_rate: Learning rate.
    dense_activation: The activation function's name for dense layers.
    model_name: The name of the model.
    channel_name: The channel name of waveform that the model is build for.
      Optional.
  """

  waveform_length: int
  waveform_channels_num: int
  latent_dim: int
  conv_kernel_size: int
  encoder_conv_num_filters: list[int]
  pool_size: int
  leaky_relu_alpha: float
  hidden_dense_units: list[int]
  beta: float
  learning_rate: float
  dense_activation: str
  model_name: str
  channel_name: Optional[str]


class GaussianSampling(tf.keras.layers.Layer):
  """Sampling layer used in variational autoencoders.

  This layer uses the mean and log(variance) to sample from a Gaussian.
  It is used in variational autoencoders for example
  ("reparameterization trick").
  """

  def call(self: ..., inputs: ...) -> ...:
    z_mean, z_log_var = inputs
    return (
        tf.keras.backend.random_normal(tf.shape(z_log_var))
        * tf.keras.backend.exp(z_log_var / 2)
        + z_mean
    )


def vae_kl_loss_mean_scaled(
    z_mean: ..., z_log_var: ..., scale_factor: float
) -> ...:
  """Returns a scaled standard KL loss for VAE.

  This comes from KL-divergence KL(N(mu, sigma^2) || N(0, 1)),
  which has a closed-form formula:
  - 1/2 * (1 + log(sigma^2) - sigma^2 - mu^2)

  Args:
    z_mean: The mean vector.
    z_log_var: The log(variance) vector.
    scale_factor: The scale factor to divide by.
  """
  kl_loss = -0.5 * tf.keras.backend.sum(
      1
      + z_log_var
      - tf.keras.backend.exp(z_log_var)
      - tf.keras.backend.square(z_mean),
      axis=-1,
  )
  return tf.keras.backend.mean(kl_loss) / scale_factor


def get_conv_1d_block(
    conv_kernel_size: int,
    conv_num_filters: list[int],
    pool_size: int,
    leaky_relu_alpha: float,
    model_name: str,
) -> list[tf.keras.layers.Layer]:
  """Get the a list of 1d convolutional layers based on the input parameters.

  Args:
    conv_kernel_size: The kernel size for the conv layers.
    conv_num_filters: A list of output filter numbers for creating the conv
      layers. The length of the list is the number of conv layers returned.
    pool_size: The pool size for pooling layers.
    leaky_relu_alpha: The leaky relu parameter.
    model_name: The name of the model that uses this block.

  Returns:
    A list of covolutional layers.
  """
  layers = []
  for i, num_filters in enumerate(conv_num_filters, start=1):
    layers.extend([
        tf.keras.layers.Conv1D(
            filters=num_filters,
            kernel_size=conv_kernel_size,
            padding='same',
            name=f'{model_name}_conv{i}',
        ),
        tf.keras.layers.LeakyReLU(leaky_relu_alpha),
        tf.keras.layers.MaxPooling1D(
            pool_size=pool_size,
            padding='same',
            name=f'{model_name}_pooling{i}',
        ),
    ])
  return layers


def get_dense_block(
    dense_units: list[int],
    activation: str,
    model_name: str,
) -> list[tf.keras.layers.Layer]:
  """Get the a list of dense layers based on the input parameters.

  Args:
    dense_units: A list of unit numbers for creating dense layers. The length of
      the list is the number of dense layers returned.
    activation: The activation function name for the dense layers.
    model_name: The name of the model that uses this block.

  Returns:
    A list of dense layers.
  """
  layers = []
  for i, units in enumerate(dense_units, start=1):
    layers.append(
        tf.keras.layers.Dense(
            units, activation=activation, name=f'{model_name}_dense{i}'
        )
    )
  return layers


def get_transposed_conv_1d_block(
    conv_kernel_size: int,
    conv_num_filters: list[int],
    pool_size: int,
    leaky_relu_alpha: float,
    model_name: str,
) -> list[tf.keras.layers.Layer]:
  """Get the a list of 1d transposed conv layers based on the input parameters.

  Args:
    conv_kernel_size: The kernel size for the transposed conv layers.
    conv_num_filters: A list of output filter numbers for creating the
      transposed conv layers. The length of the list is the number of transposed
      conv layers returned.
    pool_size: The pool size for upsampling layers.
    leaky_relu_alpha: The leaky relu parameter.
    model_name: The name of the model that uses this block.

  Returns:
    A list of transposed convolutional layers.
  """
  layers = []
  for i, num_filters in enumerate(conv_num_filters, start=1):
    layers.extend([
        tf.keras.layers.UpSampling1D(
            size=pool_size,
            name=f'{model_name}_upsample{i}',
        ),
        tf.keras.layers.Conv1DTranspose(
            filters=num_filters,
            kernel_size=conv_kernel_size,
            padding='same',
            name=f'{model_name}_trans_conv{i}',
        ),
        tf.keras.layers.LeakyReLU(leaky_relu_alpha),
    ])

  return layers


def run_tf_layer_blocks(
    model_blocks: list[list[tf.keras.layers.Layer]],
    inputs: tf.Tensor,
) -> tf.Tensor:
  """Performs inference a list of tf layers blocks.

  Each block is a list of tf layers.

  Args:
    model_blocks: A list of list of tf layers.
    inputs: A tf tensor input.

  Returns:
    The result of the blocks.
  """
  all_layers = []
  for block in model_blocks:
    all_layers.extend(block)

  x = inputs
  for layer in all_layers:
    x = layer(x)
  return x


def vae_encoder_model(
    latent_dim: int,
    input_shape: tuple[int, int],
    conv_kernel_size: int,
    conv_num_filters: list[int],
    pool_size: int,
    leaky_relu_alpha: float,
    dense_units: list[int],
    dense_activation: Optional[str] = 'relu',
    beta: float = 1.0,
    name: str = 'vae_encoder',
) -> tf.keras.Model:
  """Returns a VAE encoder model.

  Args:
    latent_dim: The dimension of the learned latent variables.
    input_shape: The shape of the encoder's input. In the case of waveform data,
      the shape should be (waveform_length, waveform_channels_num).
    conv_kernel_size: The kernel size of the convolutional layers.
    conv_num_filters: A list of filter nums for conv layers of the encoder.
    pool_size: The pool size for pooling layers.
    leaky_relu_alpha: The parameter for leaky relu.
    dense_units: A list of dense unit nums for dense layers of the encoder. The
      list except for the last item is shared with decoder (in reverse order).
      The last item in this list is equal to the latent dimension nums of the
      encoder.
    dense_activation: The activation function's name for dense layers.
    beta: The beta value to scale up the KL loss. beta = 1 returns a regular
      VAE.
    name: The name of the model.

  Returns:
    A VAE encoder model.
  """
  inputs = tf.keras.Input(shape=input_shape, name=f'{name}_input')

  conv_block = get_conv_1d_block(
      conv_kernel_size=conv_kernel_size,
      conv_num_filters=conv_num_filters,
      pool_size=pool_size,
      leaky_relu_alpha=leaky_relu_alpha,
      model_name=name,
  )

  flatten = [tf.keras.layers.Flatten(name=f'{name}_flatten')]
  dense_block = get_dense_block(dense_units, dense_activation, name)

  x = run_tf_layer_blocks([conv_block, flatten, dense_block], inputs)

  z_mean = tf.keras.layers.Dense(latent_dim, name='z_mean')(x)
  z_log_var = tf.keras.layers.Dense(latent_dim, name='z_log_var')(x)
  z = GaussianSampling()([z_mean, z_log_var])

  model = tf.keras.Model(
      inputs=inputs, outputs=[z, z_mean, z_log_var], name=name
  )

  kl_loss_mean_scaled = vae_kl_loss_mean_scaled(
      z_mean, z_log_var, scale_factor=input_shape[0] * input_shape[1] / beta
  )

  # Add KL-loss. MSE loss should be added later in the full VAE model.
  model.add_loss(kl_loss_mean_scaled)
  model.add_metric(kl_loss_mean_scaled, name='vae_kl_loss', aggregation='mean')

  return model


def conv_1d_decoder_model(
    latent_dim: int,
    waveform_length: int,
    encoder_last_num_filters: int,
    encoder_last_conv_output_length: int,
    decoder_last_conv_output_length: int,
    conv_kernel_size: int,
    conv_num_filters: list[int],
    pool_size: int,
    leaky_relu_alpha: float,
    dense_units: list[int],
    dense_activation: Optional[str] = 'relu',
    model_name: Optional[str] = 'decoder',
) -> tf.keras.Model:
  """Returns a decoder model.

  Args:
    latent_dim: The latent dimension number of the autoencoder model. It is the
      decoder's input length.
    waveform_length: The length of input waveforms of the autoencoder.
    encoder_last_num_filters: The filter num of the last conv layer of the
      encoder model. This is for reshape layer of the decoder.
    encoder_last_conv_output_length: The output length of the last conv layer of
      the encoder model.
    decoder_last_conv_output_length: The output length of the last transposed
      conv layer of the decoder model.
    conv_kernel_size: The kernel size of the convolutional layers.
    conv_num_filters: A list of filter nums for the decoder's conv layers.
    pool_size: The pool size for pooling layers.
    leaky_relu_alpha: The parameter for leaky relu.
    dense_units: A list of dense unit nums for dense layers of the decoder. The
      list except for the last item is shared with encoder (in reverse order).
      The last item in this list is equal to the flattened shape of the output
      of the encoder model's last conv layer.
    dense_activation: The activation function's name for dense layers.
    model_name: The model's name.

  Returns:
    A decoder model.
  """
  inputs = tf.keras.Input(shape=latent_dim, name=f'{model_name}_input')
  dense_block = get_dense_block(dense_units, dense_activation, model_name)
  reshape = [
      tf.keras.layers.Reshape(
          target_shape=(
              encoder_last_conv_output_length,
              encoder_last_num_filters,
          )
      )
  ]
  transposed_conv_block = get_transposed_conv_1d_block(
      conv_kernel_size,
      conv_num_filters,
      pool_size,
      leaky_relu_alpha,
      model_name,
  )
  crop = [
      tf.keras.layers.Cropping1D(
          (0, decoder_last_conv_output_length - waveform_length)
      )
  ]
  outputs = run_tf_layer_blocks(
      [dense_block, reshape, transposed_conv_block, crop], inputs
  )
  return tf.keras.Model(inputs=inputs, outputs=outputs, name=model_name)


def _get_encoder_decoder_last_conv_output_len(
    waveform_length: int,
    num_layers: int,
    pool_size: int,
) -> tuple[int, int]:
  """Get output lengths of the last 1D conv layers for encoder and decoder.

  This function is only correct when using 'padding=same' and not setting
  `stride` in the Conv1D, Conv1DTranspose layers, MaxPooling1D and UpSampling1D
  layers.

  Args:
    waveform_length: The length of input waveform of the autoencoder.
    num_layers: The number of convolutional layers of the encoder/decoder.
    pool_size: The pool size of pooling layer.

  Returns:
    The output tensors' lengths of the last conv layers of encoder and decoder.
  """

  # In the modeling, we use `padding='same'` and the default `stride=1` in the
  # Conv1D layer which results in the output of the exact same size as input.
  # Then, the pooling layer reduces the size to
  # `ceiling(encoder_conv_output_length / pool_size)`, because MaxPool1D uses
  # `pool_size` as `stride` when it is not set and `padding='same'` pads input
  # with zeros when there is partial inputs left.
  encoder_conv_output_length = waveform_length
  for _ in range(num_layers):
    encoder_conv_output_length = (
        encoder_conv_output_length + pool_size - 1
    ) // pool_size
  decoder_conv_output_length = encoder_conv_output_length

  # For each transposed convolutional layer in the decoder model, we first
  # upsampling the input by 'pool_size', this leads to a output length of
  # `decoder_conv_output_length * pool_size`. Since we use `padding='same'` and
  # the default `stride=1` in the Conv1DTranspose layer which results in the
  # output of the exact same size as input. Therefore the final output length of
  # one transposed convolutional layer is `decoder_conv_output_length *
  # pool_size`.
  for _ in range(num_layers):
    decoder_conv_output_length *= pool_size
  return encoder_conv_output_length, decoder_conv_output_length


def build_and_compile_vae_model(
    model_config: VAE1dConfig,
) -> tf.keras.Model:
  """Builds and compiles a VAE model using the config.

  Args:
    model_config: A dataclass containing structural information and parameters
      for building a 1D VAE/CAE model.

  Returns:
    A 1d VAE model compiled with the MSE and KL loss and Adam optimizer.
  """
  encoder_last_conv_output_length, decoder_last_conv_output_length = (
      _get_encoder_decoder_last_conv_output_len(
          model_config.waveform_length,
          len(model_config.encoder_conv_num_filters),
          model_config.pool_size,
      )
  )
  # Encoder, decoder models share part of the encoder_conv_num_filters and
  # hidden_dense_units. Decoder use them in a reverse order.
  decoder_conv_num_filters = model_config.encoder_conv_num_filters[::-1]
  encoder_last_num_filters = decoder_conv_num_filters.pop(0)
  decoder_conv_num_filters.append(model_config.waveform_channels_num)

  decoder_dense_units = model_config.hidden_dense_units[::-1]
  decoder_dense_units = decoder_dense_units + [
      encoder_last_conv_output_length * encoder_last_num_filters
  ]

  encoder = vae_encoder_model(
      latent_dim=model_config.latent_dim,
      input_shape=(
          model_config.waveform_length,
          model_config.waveform_channels_num,
      ),
      conv_kernel_size=model_config.conv_kernel_size,
      conv_num_filters=model_config.encoder_conv_num_filters,
      pool_size=model_config.pool_size,
      leaky_relu_alpha=model_config.leaky_relu_alpha,
      dense_units=model_config.hidden_dense_units,
      dense_activation=model_config.dense_activation,
      beta=model_config.beta,
      name='vae_encoder',
  )

  decoder = conv_1d_decoder_model(
      latent_dim=model_config.latent_dim,
      waveform_length=model_config.waveform_length,
      encoder_last_num_filters=encoder_last_num_filters,
      encoder_last_conv_output_length=encoder_last_conv_output_length,
      decoder_last_conv_output_length=decoder_last_conv_output_length,
      conv_kernel_size=model_config.conv_kernel_size,
      conv_num_filters=decoder_conv_num_filters,
      pool_size=model_config.pool_size,
      leaky_relu_alpha=model_config.leaky_relu_alpha,
      dense_units=decoder_dense_units,
      dense_activation=model_config.dense_activation,
      model_name='decoder',
  )

  inputs = tf.keras.Input(
      shape=(model_config.waveform_length, model_config.waveform_channels_num)
  )
  encoded, _, _ = encoder(inputs)

  model = tf.keras.Model(
      inputs=inputs,
      outputs=[decoder(encoded)],
      name='vae',
  )

  optimizer = tf.keras.optimizers.Adam(learning_rate=model_config.learning_rate)
  model.compile(optimizer=optimizer, loss='mse', metrics=['mse'])
  return model
