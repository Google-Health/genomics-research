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
r"""Train (R)SPINCs model using given training/validation spirograms.

SPINCs model takes a numpy array of shape (num_individuals, 1000, 2), where
two types of spirometry curves, namely flow-time and volume-time curves, are
encoded in two channels (last dimension) with the same length (=1000).

RSPINCs model takes a numpy array of shape (num_individuals, 1000, 1), where
the flow-volume curves are encoded in a single channel with the same length
(=1000), and another numpy array of "expert-defined features (EDFs)", e.g.
FEV1, FVC, PEF, FEV1/FVC ratio, and FEF25-75%.
When using RSPINCs model, `edf_dim` argument must be specified, and the
numpy array of EDFs is assumed to have shape: (num_individuals, edf_dim).

Example (SPINCs):

$ python3 train_model.py \
    --input_train=/path/to/input_train.npy \
    --input_validation=/path/to/input_validation.npy \
    --latent_dim=5 \
    --output_dir=/path/to/train_output

Example (RSPINCs):

$ python3 train_model.py \
    --input_train=/path/to/input_train.npy \
    --input_train_edfs=/path/to/input_train_edfs.npy \
    --input_validation=/path/to/input_validation.npy \
    --input_validation_edfs=/path/to/input_validation_edfs.npy \
    --latent_dim=2 \
    --edf_dim=5 \
    --output_dir=/path/to/train_output \
    --rspincs
"""
import json
import os
from typing import Sequence
from absl import app
from absl import flags
from lib import models
import numpy as np
import tensorflow as tf

_RSPINCS = flags.DEFINE_bool('rspincs', False, 'Use RSPINCs model.')
_INPUT_TRAIN = flags.DEFINE_string(
    'input_train', None, 'Path to input numpy file for training data.'
)
_INPUT_VALIDATION = flags.DEFINE_string(
    'input_validation',
    None,
    'Path to input numpy file for validation data.',
)
_INPUT_TRAIN_EDFS = flags.DEFINE_string(
    'input_train_edfs',
    None,
    (
        'Path to input numpy file for training data contaning expert-defined '
        'features. Used only if "rspincs" is True.'
    ),
)
_INPUT_VALIDATION_EDFS = flags.DEFINE_string(
    'input_validation_edfs',
    None,
    (
        'Path to input numpy file for validation data contaning expert-defined '
        'features (EDFs). Used only if "rspincs" is True.'
    ),
)
_LATENT_DIM = flags.DEFINE_integer('latent_dim', None, 'Latent dimension.')
_EDF_DIM = flags.DEFINE_integer(
    'edf_dim',
    0,
    (
        'The dimension of expert-defined features (EDFs). '
        'This must be set only if "rspincs" is True.'
    ),
)
_OUTPUT_DIR = flags.DEFINE_string(
    'output_dir', None, 'Output directory to save training results.'
)
_RANDOM_SEED = flags.DEFINE_integer('random_seed', 42, 'Random seed.')
_LEARNING_RATE = flags.DEFINE_float('learning_rate', 0.0001, 'Learning rate.')
_BATCH_SIZE = flags.DEFINE_integer('batch_size', 16, 'Batch size.')
_NUM_EPOCHS = flags.DEFINE_integer('num_epochs', 1, 'Number of epochs.')


def get_model(
    rspincs: bool, latent_dim: int, edf_dim: int, learning_rate: float
) -> tf.keras.Model:
  """Returns SPINCs or RSPINCs model."""
  if rspincs:
    model = models.get_vae_with_feature_injection(
        latent_dim=latent_dim,
        injected_latent_dim=edf_dim,
        input_shape=(1000, 1),
        beta=1,
        learning_rate=learning_rate,
    )
  else:
    assert edf_dim == 0
    model = models.get_vae_with_feature_injection(
        latent_dim=latent_dim,
        injected_latent_dim=0,
        input_shape=(1000, 2),
        beta=1,
        learning_rate=learning_rate,
    )
  return model


def main(unused_argv: Sequence[str]) -> None:
  train_np = np.load(_INPUT_TRAIN.value)
  validation_np = np.load(_INPUT_VALIDATION.value)
  num_train = train_np.shape[0]
  num_validation = validation_np.shape[0]
  if _RSPINCS.value:
    if _INPUT_TRAIN_EDFS.value is None:
      raise ValueError('Must provied "input_train_edfs" for RSPINCs.')
    if _INPUT_VALIDATION_EDFS.value is None:
      raise ValueError('Must provied "input_validation_edfs" for RSPINCs.')
    if _EDF_DIM.value <= 0:
      raise ValueError('Must provide a positive "edf_dim" for RSPINCs.')
    train_edfs_np = np.load(_INPUT_TRAIN_EDFS.value)
    validation_edfs_np = np.load(_INPUT_VALIDATION_EDFS.value)
    assert train_np.shape == (num_train, 1000, 1)
    assert validation_np.shape == (num_validation, 1000, 1)
    assert train_edfs_np.shape == (num_train, _EDF_DIM.value)
    assert validation_edfs_np.shape == (num_validation, _EDF_DIM.value)
    train_input = [train_np, train_edfs_np]
    train_label = train_np
    validation_input = [validation_np, validation_edfs_np]
    validation_label = validation_np
  else:
    if _INPUT_TRAIN_EDFS.value is not None:
      raise ValueError('Must not use "input_train_edfs" for SPINCs.')
    if _INPUT_VALIDATION_EDFS.value is not None:
      raise ValueError('Must not use "input_validation_edfs" for SPINCs.')
    if _EDF_DIM.value != 0:
      raise ValueError('Must not use "edf_dim" for SPINCs.')
    assert train_np.shape == (num_train, 1000, 2)
    assert validation_np.shape == (num_validation, 1000, 2)
    train_input = train_np
    train_label = train_np
    validation_input = validation_np
    validation_label = validation_np

  model = get_model(
      rspincs=_RSPINCS.value,
      latent_dim=_LATENT_DIM.value,
      edf_dim=_EDF_DIM.value,
      learning_rate=_LEARNING_RATE.value,
  )
  output_dir = _OUTPUT_DIR.value

  # Create output directories.
  os.makedirs(os.path.join(output_dir, 'checkpoint'), exist_ok=True)

  callbacks = [
      # Checkpoint callback.
      tf.keras.callbacks.ModelCheckpoint(
          filepath=os.path.join(output_dir, 'checkpoint', 'best-cp.ckpt'),
          save_weights_only=False,
          monitor='val_loss',
          mode='min',
          save_freq='epoch',
          save_best_only=True,
      ),
  ]
  tf.random.set_seed(_RANDOM_SEED.value)
  history = model.fit(
      train_input,
      train_label,
      batch_size=_BATCH_SIZE.value,
      validation_data=(validation_input, validation_label),
      epochs=_NUM_EPOCHS.value,
      shuffle=True,
      verbose=1,
      callbacks=callbacks,
  )
  # Write training history as JSON.
  with open(os.path.join(output_dir, 'training_history.json'), 'wt') as f:
    json.dump(history.history, f, sort_keys=True, indent=2)
    f.write('\n')


if __name__ == '__main__':
  flags.mark_flags_as_required(
      ['input_train', 'input_validation', 'rspincs', 'latent_dim', 'output_dir']
  )
  app.run(main)
