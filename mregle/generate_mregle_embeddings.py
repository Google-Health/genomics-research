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
r"""Generates M-REGLE embeddings.

Example:
$ python3 generate_mregle_embeddings.py \
  --output_dir=/path/to/output \
  --dataset=ecgppg
"""

from collections.abc import Sequence
import os
from absl import app
from absl import flags
import numpy as np
import tensorflow as tf

_INPUT_PATH = flags.DEFINE_string(
    'input_path', None, 'Path to input waveform tsv file.'
)
_OUTPUT_DIR = flags.DEFINE_string(
    'output_dir', None, 'Path to output embeddings.'
)
_DATASET = flags.DEFINE_enum(
    'dataset',
    None,
    ['ecg12', 'ecgppg'],
    'The model type.',
)

flags.mark_flags_as_required(['output_dir', 'dataset'])

ECG = 'ecg'
PPG = 'ppg'


def get_vae_encoding(
    data: np.ndarray,
    model: tf.keras.Model,
    encoder_name: str = 'vae_encoder',
) -> np.ndarray:
  """Returns the encoding of data by a VAE model."""
  encoder = model.get_layer(encoder_name)
  # VAE's encoder returns (sample, mean, log_variance).
  _, encoded_mean, _ = encoder.predict(data, verbose=0)
  return np.squeeze(encoded_mean)


def get_joint_embeddings_ecg12(npy_path: str, model_ckpt: str) -> np.ndarray:
  """Gets the joint representation of the input 12-lead ECG.

  Args:
    npy_path: The path of a npy file containing the preprocessed 12-lead ECG.
    model_ckpt: The trained model checkpoint.

  Returns:
    The joint representation of the input 12-lead ECG.
  """
  # load the array, and make multimodal input data.
  multimodal_arr = np.load(npy_path)

  # load model ckpt.
  model = tf.keras.models.load_model(model_ckpt)

  # get the embeddings.
  joint_embeddings = get_vae_encoding(multimodal_arr, model)
  return joint_embeddings


def get_joint_embeddings_ecgppg(npy_path: str, model_ckpt: str) -> np.ndarray:
  """Gets the joint representation of the input ECG+PPG data.

  Args:
    npy_path: The path of a npy file containing the preprocessed ECG+PPG.
    model_ckpt: The trained model checkpoint.

  Returns:
    The joint representation of the input ECG+PPG.
  """
  # load the array, and make multimodal input data.
  multimodal_arr = np.load(npy_path)

  # load model ckpt.
  model = tf.keras.models.load_model(model_ckpt)

  # get the embeddings.
  joint_embeddings = get_vae_encoding(multimodal_arr, model)
  return joint_embeddings


def main(unused_argv: Sequence[str]) -> None:

  if _DATASET.value == 'ecg12':
    input_file = 'data/ecg_ml_data.npy'
    if _INPUT_PATH.value:
      input_file = _INPUT_PATH.value
    if not os.path.isfile(input_file):
      raise ValueError(
          f'Input file {input_file} does not exist. Please generate the input'
          ' first.'
      )

    model_ckpt = 'model_ckpts/ecg12/multimodal'
    ecg12_joint_reps = get_joint_embeddings_ecg12(input_file, model_ckpt)

    output_file = _OUTPUT_DIR.value + '/ecg12_joint_reps.npy'
    np.save(output_file, ecg12_joint_reps)

  elif _DATASET.value == 'ecgppg':
    input_file = 'data/ecgppg_ml_data.npy'
    if _INPUT_PATH.value:
      input_file = _INPUT_PATH.value
    if not os.path.isfile(input_file):
      raise ValueError(
          f'Input file {input_file} does not exist. Please generate the input'
          ' first.'
      )

    model_ckpt = 'model_ckpts/ecg_leadI_ppg/multimodal'
    ecgppg_joint_reps = get_joint_embeddings_ecgppg(input_file, model_ckpt)

    output_file = _OUTPUT_DIR.value + '/ecgppg_joint_reps.npy'
    np.save(output_file, ecgppg_joint_reps)


if __name__ == '__main__':
  app.run(main)
