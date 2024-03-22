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
r"""Trains a vae model on multimodal ecg/ecg+ppg data.

Examples:
Train a multimodal 12-lead ECG model:
$ python3 train.py \
  --logging_dir=log \
  --data_setting=ecg12 \
  --train_data_path=demo_train/ecg_ml_data.npy \
  --validation_data_path=demo_val/ecg_ml_data.npy \
  --latent_dim=96

Train a multimodal ECG+PPG model:
$ python3 train.py \
  --logging_dir=log \
  --data_setting=ecgppg \
  --train_data_path=demo_train/ecgppg_ml_data.npy \
  --validation_data_path=demo_val/ecgppg_ml_data.npy \
  --latent_dim=12
"""
import json
import os
from typing import Any, ContextManager, Dict, Optional, Sequence

from absl import app
from absl import flags
from absl import logging
from lib import models
import numpy as np
import tensorflow as tf


# Training params.
_LOGGING_DIR = flags.DEFINE_string(
    'logging_dir', None, 'The dir to save the training output.'
)
_DATA_SETTING = flags.DEFINE_enum(
    'data_setting',
    'ecg12',
    ['ecg12', 'ecgppg'],
    'ecg12 or ecgppg.',
)
_TRAIN_DATA_PATH = flags.DEFINE_string(
    'train_data_path',
    'data/ecg_ml_data.npy',
    'The npy file path of training data.',
)
_VALIDATION_DATA_DIR = flags.DEFINE_string(
    'validation_data_path',
    'data/ecg_ml_data.npy',
    'The npy file path of validation data.',
)
_MODALITY_TYPE = flags.DEFINE_enum(
    'modality_type',
    'multimodal',
    ['single_modal', 'multimodal'],
    'single_modal or multimodal.',
)
_DATA_CHANNEL_NAME = flags.DEFINE_string(
    'data_channel_name',
    None,
    'The channel name that we train a single modal CAE on.',
)
_BATCH_SIZE = flags.DEFINE_integer(
    'batch_size', 32, 'Batch size for model training and evaluation.'
)
_RANDOM_SEED = flags.DEFINE_integer(
    'random_seed', 42, 'Random seed for reproducibility.'
)
_LATENT_DIM = flags.DEFINE_integer(
    'latent_dim', 8, 'Latent dimension number of the CAE model.'
)
_CONV_KERNEL_SIZE = flags.DEFINE_integer(
    'conv_kernel_size', 7, 'Size of the 1d convolutional filter.'
)
_NUM_CONV_FILTER = flags.DEFINE_integer(
    'num_conv_filter', 32, 'The convolutional layer filters num.'
)
_NUM_CONV_LAYER = flags.DEFINE_integer(
    'num_conv_layer', 3, 'The number of convolutional layers.'
)
_NUM_DENSE_UNIT = flags.DEFINE_integer(
    'num_dense_unit', 128, 'The number of units in the dense layer.'
)
_LEARNING_RATE = flags.DEFINE_float(
    'learning_rate', 0.0001, 'Learning rate for an optimizer'
)
_DENSE_ACTIVATION = flags.DEFINE_string(
    'dense_activation',
    'relu',
    'The activation function for the dense layer.',
)
_BETA = flags.DEFINE_float('beta', 1.0, 'The beta parameter of VAE model.')
_NUM_EPOCHS = flags.DEFINE_integer(
    'num_epochs', 100, 'The number of epochs for training the model.'
)

PPG_LENGTH = 100
ECG_LENGTH = 600
ECG_CHANNELNUM = 12

ECG12_CHANNEL_LEN_DICT = {
    'I': ECG_LENGTH,
    'II': ECG_LENGTH,
    'III': ECG_LENGTH,
    'V1': ECG_LENGTH,
    'V2': ECG_LENGTH,
    'V3': ECG_LENGTH,
    'V4': ECG_LENGTH,
    'V5': ECG_LENGTH,
    'V6': ECG_LENGTH,
    'aVF': ECG_LENGTH,
    'aVR': ECG_LENGTH,
    'avL': ECG_LENGTH,
}

ECGPPG_CHANNEL_LEN_DICT = {
    'ecg': ECG_LENGTH,
    'ppg': PPG_LENGTH,
}


def _get_train_valid_datasets_and_lengths(
    training_data_path: str,
    validation_data_path: str,
    data_setting: str,
) -> tuple[tf.data.Dataset, tf.data.Dataset, Dict[str, int]]:
  """Gets training and validation datasets and waveforms' lengths.

  Args:
    training_data_path: The path of training npy file.
    validation_data_path: The path of validation npy file.
    data_setting: Whether to use multichannel ECG or ECG + PPG.

  Returns:
    A training dataset, a validation dataset and a dictionary of the lengths of
    the waveform data in the datasets.

  Raises:
    ValueError: When it is single_modal but channel name is not specified.
  """
  training_data = np.load(training_data_path)
  validation_data = np.load(validation_data_path)
  channel_length_dict = {}
  if data_setting == 'ecg12':
    if len(training_data.shape) == 3:
      assert training_data.shape[1] == 600
      assert training_data.shape[2] == 12
      assert len(validation_data.shape) == 3
      assert validation_data.shape[1] == 600
      assert validation_data.shape[2] == 12
    elif len(training_data.shape) == 2:
      assert training_data.shape[0] == 600
      assert training_data.shape[1] == 12
      assert len(validation_data.shape) == 2
      assert validation_data.shape[0] == 600
      assert validation_data.shape[1] == 12
      training_data = np.expand_dims(training_data, axis=0)
      validation_data = np.expand_dims(validation_data, axis=0)
    channel_length_dict = ECG12_CHANNEL_LEN_DICT
  elif data_setting == 'ecgppg':
    assert len(training_data.shape) == 2
    assert training_data.shape[1] == 700
    assert len(validation_data.shape) == 2
    assert validation_data.shape[1] == 700
    channel_length_dict = ECGPPG_CHANNEL_LEN_DICT

  return training_data, validation_data, channel_length_dict


def _get_model(
    data_setting: str,
    modality_type: str,
    channel_length_dict: Dict[str, int],
    latent_dim: int,
    conv_kernel_size: int,
    num_conv_filter: int,
    num_conv_layer: int,
    num_dense_unit: int,
    learning_rate: float,
    dense_activation: str,
    beta: float,
    channel_name: Optional[str] = None,
) -> tf.keras.Model:
  """Builds and compiles the CAE model given the structural information.

  Args:
    data_setting: ecg12 or ecgppg.
    modality_type: single_modal or multimodal.
    channel_length_dict: A dictionary of data lengths in the dataset.
    latent_dim: The latent dimension number.
    conv_kernel_size: The length of the 1d kernel of convolutional layers.
    num_conv_filter: The number of convolutional filters.
    num_conv_layer: The number of convolutional layers.
    num_dense_unit: The number of units of the dense layer.
    learning_rate: The learning rate of optimizer.
    dense_activation: The activation function name of the dense layer.
    beta: Beta in the VAE model, which is the weight for KL loss.
    channel_name: The channel name when training single modal.

  Returns:
    A compiled tf model.
  """
  if modality_type == 'single_modal':
    input_length = channel_length_dict[channel_name]
    channel_num = 1
  else:
    if data_setting == 'ecg12':
      input_length = ECG_LENGTH
      channel_num = ECG_CHANNELNUM
    elif data_setting == 'ecgppg':
      input_length = sum(channel_length_dict.values())
      channel_num = 1
    else:
      raise ValueError('Data setting can only be "ecg12" or "ecgppg".')

  if modality_type == 'multimodal' and data_setting == 'ecg12':
    hidden_dense_units = [num_dense_unit, 256]
  else:
    hidden_dense_units = [num_dense_unit, 32]

  model_config = models.VAE1dConfig(
      waveform_length=input_length,
      waveform_channels_num=channel_num,
      latent_dim=latent_dim,
      conv_kernel_size=conv_kernel_size,
      encoder_conv_num_filters=[num_conv_filter] * num_conv_layer,
      pool_size=2,
      leaky_relu_alpha=0.1,
      hidden_dense_units=hidden_dense_units,
      beta=beta,
      learning_rate=learning_rate,
      dense_activation=dense_activation,
      model_name='vae',
      channel_name=channel_name,
  )
  model = models.build_and_compile_vae_model(model_config)
  return model


def _get_callbacks(
    base_dir: str,
) -> Sequence[tf.keras.callbacks.Callback]:
  """Gets the callbacks for model.fit.

  Args:
    base_dir: The base directory of the experiment for logging.

  Returns:
    A list of callbacks.
  """
  callbacks = [
      # Checkpoint callback.
      tf.keras.callbacks.ModelCheckpoint(
          filepath=os.path.join(base_dir, 'checkpoint', 'best-cp.ckpt'),
          save_weights_only=False,
          monitor='val_loss',
          mode='min',
          save_freq='epoch',
          save_best_only=True,
      ),
  ]

  return callbacks


def _get_distribution_strategy_scope() -> ContextManager[Any]:
  """Returns the device-specific TensorFlow strategy scope used for training."""
  return tf.distribute.MirroredStrategy().scope()


def main(argv: Sequence[str]) -> None:
  if len(argv) > 1:
    raise app.UsageError('Too many command-line arguments.')

  tf.keras.utils.set_random_seed(_RANDOM_SEED.value)

  data_setting = _DATA_SETTING.value
  modality_type = _MODALITY_TYPE.value
  channel_name = _DATA_CHANNEL_NAME.value
  base_dir = _LOGGING_DIR.value

  # Loads the datasets based on the data setting and modality type.
  train_dataset, valid_dataset, channel_length_dict = (
      _get_train_valid_datasets_and_lengths(
          _TRAIN_DATA_PATH.value,
          _VALIDATION_DATA_DIR.value,
          data_setting,
      )
  )
  logging.info('Train and validation datasets loaded.')

  model = _get_model(
      data_setting,
      modality_type,
      channel_length_dict,
      _LATENT_DIM.value,
      _CONV_KERNEL_SIZE.value,
      _NUM_CONV_FILTER.value,
      _NUM_CONV_LAYER.value,
      _NUM_DENSE_UNIT.value,
      _LEARNING_RATE.value,
      _DENSE_ACTIVATION.value,
      _BETA.value,
      channel_name,
  )
  logging.info('Model is created.')

  # Create the base dir to store the information of model training.
  os.makedirs(base_dir, exist_ok=True)

  # Gets callback functions
  callbacks = _get_callbacks(
      base_dir=base_dir,
  )

  # logging.info('Beginning of training (attempt #%s)', num_restarts)
  history = model.fit(
      train_dataset,
      train_dataset,
      batch_size=_BATCH_SIZE.value,
      validation_data=(valid_dataset, valid_dataset),
      epochs=_NUM_EPOCHS.value,
      verbose=1,
      callbacks=callbacks,
  )

  # Store the training history as pickle file.
  with open(os.path.join(base_dir, 'training_history.json'), 'wt') as f:
    json.dump(history.history, f, sort_keys=True, indent=2)
    f.write('\n')
  logging.info('End of training.')


if __name__ == '__main__':
  app.run(main)
