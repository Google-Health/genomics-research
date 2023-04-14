# Copyright 2023 Google LLC.
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
"""Spirometry representation learning models."""
from typing import Sequence, Tuple
import tensorflow as tf

ENCODER_CONV_FILTERS1 = (8, 16, 32)
ENCODER_CONV_FILTERS2 = (4, 8, 16)


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


def _vae_kl_loss_mean_scaled(
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


def get_vae_encoder_model(
    latent_dim: int = 8,
    input_shape: Tuple[int, int] = (1000, 2),
    kernel_size: int = 10,
    encoder_conv_filters: Sequence[int] = ENCODER_CONV_FILTERS1,
    dense_size: int = 64,
    beta: float = 1.0,
    name: str = 'vae_encoder',
) -> tf.keras.Model:
  """Encoder model for a (beta-) variational autoencoder.

  Args:
    latent_dim: The latent dimension.
    input_shape: The input shape. (1000, 2) for time-flow and time-volume
      spirograms.
    kernel_size: The kernel size.
    encoder_conv_filters: The number of convolutional filters in each layer,
      starting from the closest layer from the input.
    dense_size: The number of neurons in each dense layer before the encoding.
      There will be 3 dense layers of this size.
    beta: The beta value to scale up the KL loss. beta = 1 returns a regular
      VAE.
    name: The name of the model.

  Returns:
    Keras model (functional API) representing a VAE encoder.
  """
  if input_shape[0] % 2 ** len(encoder_conv_filters) > 0:
    raise ValueError(
        f'2^(number of convolution layers) must divide {input_shape[0]}.'
    )
  encoder_conv_filters = list(encoder_conv_filters)
  inputs = tf.keras.Input(shape=input_shape, name=f'{name}_input')
  x = inputs
  for num_filter in encoder_conv_filters:
    x = tf.keras.layers.Conv1D(
        filters=num_filter,
        kernel_size=kernel_size,
        activation='relu',
        padding='same',
    )(x)
    x = tf.keras.layers.MaxPooling1D(2, padding='same')(x)
  x = tf.keras.layers.Flatten()(x)
  x = tf.keras.layers.Dense(dense_size, activation='relu')(x)
  x = tf.keras.layers.Dense(dense_size, activation='relu')(x)
  x = tf.keras.layers.Dense(dense_size, activation='relu')(x)

  z_mean = tf.keras.layers.Dense(latent_dim, name='z_mean')(x)
  z_log_var = tf.keras.layers.Dense(latent_dim, name='z_log_var')(x)
  z = GaussianSampling()([z_mean, z_log_var])
  model = tf.keras.Model(
      inputs=inputs, outputs=[z, z_mean, z_log_var], name=name
  )

  # KL loss scale factor = the total dimension of the input / "beta".
  # beta = 1 for standard VAE. If beta > 1, this becomes a "beta-VAE" and
  # increases the penalty on the KL term relative to the reconstruction loss.

  kl_loss_mean_scaled = _vae_kl_loss_mean_scaled(
      z_mean, z_log_var, scale_factor=input_shape[0] * input_shape[1] / beta
  )

  # Add KL-loss. MSE loss should be added later in the full VAE model.
  model.add_loss(kl_loss_mean_scaled)
  model.add_metric(kl_loss_mean_scaled, name='vae_kl_loss', aggregation='mean')

  return model


def get_decoder_model(
    latent_dim: int = 4,
    output_shape: Tuple[int, int] = (1000, 2),
    kernel_size: int = 10,
    encoder_conv_filters: Sequence[int] = ENCODER_CONV_FILTERS1,
    dense_size: int = 64,
    name: str = 'decoder',
) -> tf.keras.Model:
  """Decoder model for (variational/denoising) autoencoder.

  Args:
    latent_dim: The latent dimension.
    output_shape: The output shape. (1000, 2) for time-flow and time-volume
      spirograms.
    kernel_size: The kernel size.
    encoder_conv_filters: The number of convolutional filters in each layer in
      the matching *encoder*, starting from the closest layer from the input.
      The numbers will be flipped to create a mirror image of the encoder.
    dense_size: The number of neurons in each dense layer before the encoding.
      There will be 3 dense layers of this size.
    name: The name of the model.

  Returns:
    A Keras model (functional API) representing the decoder.
  """
  if output_shape[0] % 2 ** len(encoder_conv_filters) > 0:
    raise ValueError(
        f'2^(number of convolution layers) must divide {output_shape[0]}.'
    )
  encoder_conv_filters = list(encoder_conv_filters)
  last_conv_output_length = output_shape[0] // 2 ** len(encoder_conv_filters)
  inputs = tf.keras.Input(shape=(latent_dim,), name=f'{name}_input')
  x = inputs

  # Three large dense layers connected to the encodings.
  x = tf.keras.layers.Dense(dense_size, activation='relu')(x)
  x = tf.keras.layers.Dense(dense_size, activation='relu')(x)
  x = tf.keras.layers.Dense(dense_size, activation='relu')(x)

  x = tf.keras.layers.Dense(
      last_conv_output_length * encoder_conv_filters[-1], activation='relu'
  )(x)
  x = tf.keras.layers.Reshape(
      target_shape=(last_conv_output_length, encoder_conv_filters[-1])
  )(x)

  # We want to build a mirror image of the encoder here.
  # Remove the last element (already used above at the dense layer), reverse it,
  # and add the number of channels of the input.
  decoder_conv_filters = encoder_conv_filters[:-1][::-1] + [output_shape[1]]
  for num_filter in decoder_conv_filters:
    x = tf.keras.layers.UpSampling1D(2)(x)
    x = tf.keras.layers.Conv1DTranspose(
        filters=num_filter,
        kernel_size=kernel_size,
        activation='relu',
        padding='same',
    )(x)
  return tf.keras.Model(inputs=inputs, outputs=x, name=name)


def get_vae_with_feature_injection(
    latent_dim: int = 4,
    injected_latent_dim: int = 0,
    input_shape: Tuple[int, int] = (1000, 2),
    kernel_size: int = 10,
    encoder_conv_filters: Sequence[int] = ENCODER_CONV_FILTERS1,
    dense_size: int = 64,
    beta: float = 1.0,
    learning_rate: float = 1e-3,
    name: str = 'vae',
) -> tf.keras.Model:
  """(Beta-) variational autoencoder model.

  This returns a *compiled* Keras model for a variational autoencoder.
  Note that the returned model is compiled to make sure the losses are
  correctly added.

  Args:
    latent_dim: The dimension of the learned latent variables.
    injected_latent_dim: The injected latent dimension.
    input_shape: The input shape. (1000, 2) for time-flow and time-volume
      spirograms.
    kernel_size: The kernel size.
    encoder_conv_filters: The number of convolutional filters in each layer,
      starting from the closest layer from the input.
    dense_size: The number of neurons in each dense layer before the encoding.
      There will be 3 dense layers of this size.
    beta: The beta value to scale up the KL loss. beta = 1 returns a regular
      VAE.
    learning_rate: The learning rate for the Adam optimizer (for compilation).
    name: The name of the model.

  Returns:
    Compiled Keras model for a variational autoencoder.
  """
  assert latent_dim > 0

  encoder = get_vae_encoder_model(
      latent_dim=latent_dim,
      input_shape=input_shape,
      kernel_size=kernel_size,
      encoder_conv_filters=encoder_conv_filters,
      dense_size=dense_size,
      beta=beta,
      name=f'{name}_encoder',
  )
  decoder = get_decoder_model(
      latent_dim=latent_dim + injected_latent_dim,
      output_shape=input_shape,
      kernel_size=kernel_size,
      encoder_conv_filters=encoder_conv_filters,
      dense_size=dense_size,
      name=f'{name}_decoder',
  )

  curve_inputs = tf.keras.Input(shape=input_shape, name='vae_curve_input')

  # Encoder returns (sample, mean, log_var).
  encoded, _, _ = encoder(curve_inputs)

  if injected_latent_dim > 0:
    # We have injected variables in this case as input.
    injected_feature_inputs = tf.keras.Input(
        shape=(injected_latent_dim,), name='vae_feature_input'
    )
    all_inputs = [curve_inputs, injected_feature_inputs]
    aug_encoded = tf.keras.layers.Concatenate()(
        [encoded, injected_feature_inputs]
    )
  else:
    # We don't have injected variables in this case, just use the encodings.
    all_inputs = curve_inputs
    aug_encoded = encoded

  decoded = decoder(aug_encoded)

  vae = tf.keras.Model(inputs=all_inputs, outputs=decoded, name=name)

  # Note that KL-loss is already added in the encoder.
  # Here we just add the MSE loss.
  vae.compile(
      loss='mse',
      optimizer=tf.keras.optimizers.Adam(learning_rate),
      metrics=['mse'],
  )

  return vae
