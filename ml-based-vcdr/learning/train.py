# Copyright 2021 Google LLC.
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
"""Pipeline for training models."""
import pathlib
from typing import List
from typing import Tuple

from absl import app
from absl import flags
import input_pipeline
import ml_collections
from ml_collections import config_flags
import model_utils
import tensorflow as tf
import tensorflow_addons as tfa

FLAGS = flags.FLAGS

config_flags.DEFINE_config_file(
    'config',
    None,
    'A resource path to a ConfigDict training configuration. Also allows for '
    'command-line config overrides of the form `--config.seed=43.`',
    lock_config=True)
flags.DEFINE_string('workdir', None, 'Working directory for the experiment.')


def get_callbacks(
    workdir: pathlib.Path,
    config: ml_collections.ConfigDict) -> List[tf.keras.callbacks.Callback]:
  """Builds and returns a list of tensorflow callbacks for use in training.

  Args:
    workdir: The directory where logs and checkpoints are saved.
    config: The experimental configuration.

  Returns:
    A list of tensorflow callbacks for TensorBoard and checkpointing.
  """
  log_dir = workdir / 'logs'
  checkpoint_dir = workdir / 'checkpoints'
  checkpoint_file = checkpoint_dir / 'weights.best.ckpt'
  checkpoint_file_avg = checkpoint_dir / 'weights.avg.best.ckpt'

  callbacks = [tf.keras.callbacks.TensorBoard(log_dir=log_dir)]

  if config.opt.use_model_averaging:
    update_weights = config.opt.get('update_model_averaging_weights', True)
    callbacks.append(
        tfa.callbacks.AverageModelCheckpoint(
            filepath=str(checkpoint_file_avg),
            update_weights=update_weights,
            save_best_only=True,
            save_weights_only=True,
            monitor='val_vertical_cup_to_disc_loss',
            mode='min',
            save_freq='epoch'))

    # If not update model weights to include variable averages, checkpoint both
    # the standard and averaged model.
    if not update_weights:
      callbacks.append(
          tf.keras.callbacks.ModelCheckpoint(
              filepath=checkpoint_file,
              save_best_only=True,
              save_weights_only=True,
              monitor='val_vertical_cup_to_disc_loss',
              mode='min',
              save_freq='epoch'))

  else:
    callbacks.append(
        tf.keras.callbacks.ModelCheckpoint(
            filepath=checkpoint_file,
            save_best_only=True,
            save_weights_only=True,
            monitor='val_vertical_cup_to_disc_loss',
            mode='min',
            save_freq='epoch'))

  return callbacks


def get_datasets(
    config: ml_collections.ConfigDict,
) -> Tuple[tf.data.Dataset, tf.data.Dataset]:
  """Fetches and return tensorflow datasets."""
  dataset_config = config.get('dataset', None)
  if dataset_config is None:
    raise ValueError(
        f'Provided "config" missing "dataset" sub-config: {config}')

  cache = dataset_config.get('use_cache', False)
  return input_pipeline.build_datasets(
      dataset_config, config.outcomes, cache=cache)


def train(
    workdir: pathlib.Path,
    config: ml_collections.ConfigDict,
) -> tf.keras.callbacks.History:
  """Trains the model and returns training and evaluation history."""

  # Set seed for reproducibility.
  tf.random.set_seed(config.seed)

  if config.train.get('use_mixed_precision', False):
    tf.keras.mixed_precision.set_global_policy('mixed_float16')

  config, model = model_utils.get_model(config)

  checkpoint = tf.train.latest_checkpoint(workdir / 'checkpoints')
  if checkpoint:
    print(f'Found checkpoint: loading model from {checkpoint}...')
    model.load_weights(checkpoint).expect_partial()

  train_ds, eval_ds = get_datasets(config)

  verbose = config.train.get('fit_verbose', 0) if 'train' in config else 0

  # We treat an epoch as the # of steps taken between logging and checkpoints.
  num_log_epochs = int(config.train.max_num_steps / config.train.log_step_freq)

  return model.fit(
      train_ds,
      validation_data=eval_ds,
      epochs=num_log_epochs,
      initial_epoch=config.train.get('initial_epoch', 0),
      steps_per_epoch=config.train.log_step_freq,
      callbacks=get_callbacks(workdir, config),
      verbose=verbose,
  )


def main(unused_argv):
  print('Running the training pipeline with the following config:')
  print(FLAGS.config)
  train(workdir=pathlib.Path(FLAGS.workdir), config=FLAGS.config)


if __name__ == '__main__':
  flags.mark_flags_as_required(['config', 'workdir'])
  app.run(main)
