# Copyright 2022 Google LLC
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#      http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
"""Custom `tf.keras.callbacks.Callback` implementations and loaders."""
import pathlib
from typing import List

import ml_collections
import tensorflow as tf

CHECKPOINT_SUBDIR = 'checkpoints'
_LOG_SUBDIR = 'logs'
_CHECKPOINT_FILENAME = 'weights.best.ckpt'


def get_callbacks(
    work_dir: pathlib.Path, callbacks_config: ml_collections.ConfigDict
) -> List[tf.keras.callbacks.Callback]:
  """Returns a list of callbacks for use in model training and evaluation.

  The `callbacks_config` `ConfigDict` must match the config schema defined in
  the README.md.

  The returned list includes some specific callbacks. By default, a TensorBoard
  callback is included and writes logs to the '{work_dir}/logs' subdirectory.
  Checkpointing and early stopping are enabled and configured by the
  `callbacks_config`. Checkpoints are written to the '{work_dir}/checkpoints'
  subdirectory.

  Args:
    work_dir: The base work directory in which to write logs, checkpoints, etc.
    callbacks_config: A ConfigDict matching the `callbacks_config` schema.

  Returns:
    A list of callback instances.
  """
  log_dir = work_dir / _LOG_SUBDIR
  callbacks = [tf.keras.callbacks.TensorBoard(log_dir=log_dir)]

  if callbacks_config['checkpoint_best']:
    checkpoint_file = work_dir / CHECKPOINT_SUBDIR / _CHECKPOINT_FILENAME
    checkpoint_callback = tf.keras.callbacks.ModelCheckpoint(
        filepath=checkpoint_file, save_best_only=True, save_weights_only=True)
    callbacks.append(checkpoint_callback)

  if callbacks_config['use_early_stopping']:
    early_stopping_callback = tf.keras.callbacks.EarlyStopping(
        patience=callbacks_config['early_stopping_patience'])
    callbacks.append(early_stopping_callback)

  return callbacks
