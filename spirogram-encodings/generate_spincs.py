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
r"""Generates spirometry encodings.

Generates spirometry encodings (SPINCs) and residual spirometry encodings
(RSPINCs) from the given input spirogram. The input values are assumed to be
normalized to have exactly 1000 values. The inputs must be in numpy array
format (.npy) and the output will also be saved in the same format.

For SPINCs model, the input numpy array must have the following shape:
(number_of_individuals, 1000, 2)
where the flow-time and volume-time curves are encoded in two channels.

For RSPINCs model, the input numpy array must have the following shape:
(number_of_individuals, 1000, 1)
where the flow-volume curves are encoded in a single channel.

Example:

$ python3 generate_spincs.py \
    --input_path=/path/to/input.npy \
    --output_path=/path/to/output.npy \
    --model_type=spincs
"""
import os
from typing import Sequence

from absl import app
from absl import flags
import numpy as np
import tensorflow as tf

_INPUT_PATH = flags.DEFINE_string(
    'input_path', None, 'Path to input numpy file containing spirogram values.'
)
_OUTPUT_PATH = flags.DEFINE_string(
    'output_path', None, 'Path to output numpy file.'
)

_MODEL_TYPE = flags.DEFINE_enum(
    'model_type',
    None,
    ['spincs', 'rspincs'],
    'The model type.',
)

_PRINT_OUTPUT = flags.DEFINE_boolean(
    'print_output',
    False,
    'Print outputs to command line. Only for debugging a small file only.',
)


def get_vae_encodings(data: np.ndarray, model: tf.keras.Model) -> np.ndarray:
  """Generates encodings using a trained VAE model's encoder."""
  encoder_layer_name = f'{model.name}_encoder'
  encoder = model.get_layer(encoder_layer_name)
  # VAE's encoder returns (sample, mean, log_variance).
  _, encoded_mean, _ = encoder.predict(data)
  return encoded_mean


def main(argv: Sequence[str]) -> None:
  if len(argv) > 1:
    raise app.UsageError('Too many command-line arguments.')

  model_type = _MODEL_TYPE.value
  checkpoint_path = os.path.join('saved_models', model_type)
  input_nparray = np.load(_INPUT_PATH.value)

  # Argument validations.
  if len(input_nparray.shape) != 3:
    raise ValueError(
        'The input numpy array should have three dimensions: '
        '(individuals, spirogram values, channels)'
    )
  if input_nparray.shape[1] != 1000:
    raise ValueError(
        'Expect 1000 values in the second dimension of the numpy array.'
    )
  if model_type == 'spincs':
    if input_nparray.shape[2] != 2:
      raise ValueError('SPINCs model needs exactly two channels.')
  else:
    if input_nparray.shape[2] != 1:
      raise ValueError('RSPINCs model needs exactly one channel.')

  sample_count = input_nparray.shape[0]
  vae_model = tf.keras.models.load_model(checkpoint_path)
  output_nparray = get_vae_encodings(input_nparray, vae_model)
  if model_type == 'spincs':
    assert output_nparray.shape == (sample_count, 5)
  else:
    assert output_nparray.shape == (sample_count, 2)

  np.save(_OUTPUT_PATH.value, output_nparray)
  if _PRINT_OUTPUT.value:
    print('========== (R)SPINCs VALUES ==========')
    print(output_nparray)


if __name__ == '__main__':
  flags.mark_flags_as_required(['input_path', 'output_path', 'model_type'])
  app.run(main)
