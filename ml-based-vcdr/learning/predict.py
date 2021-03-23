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
r"""Script for generating predictions from a single model.

Model predictions are written to a CSV with a unique `predict_utils.ID_KEY` ID
column corresponding to each input image. The CSV contains a column for each
classification and regression target, i.e., a multiclass outcome of shape
`(None, N)` will have `N` columns. See `predict_utils.OUTCOME_COLUMN_MAP` for
column mapping.

This script should be run with the same `workdir` and `config` used in training.

Note: If generating predictions with a single member of an ensemble trained
using the `train_ensemble.py` script, the provided workdir should correspond to
the member's subdirectory, e.g., `base_workdir/member_{i}`.

Example usages:

# Generating predictions with a single model trained using `train.py`:
$ python3 predict.py \
    --config=configs/base.py \
    --workdir=./reproduce_vcdr \
    --output_filepath=./outcome_predictions.csv

# Generating predictions with an ensemble member trained using
# `train_ensemble.py`:
$ python3 predict.py \
    --config=configs/base.py \
    --workdir=./reproduce_vcdr_ensemble_base/member_9 \
    --output_filepath=./outcome_predictions.csv
"""
import pathlib

from absl import app
from absl import flags
import input_pipeline
import ml_collections
from ml_collections import config_flags
import model_utils
import predict_utils
import tensorflow as tf

FLAGS = flags.FLAGS

config_flags.DEFINE_config_file(
    'config',
    None,
    'A resource path to the ConfigDict used during training.',
    lock_config=True)
flags.DEFINE_string('workdir', None, 'The `workdir` used during training')
flags.DEFINE_string('output_filepath', 'predictions.csv',
                    'The filepath at which the output CSV is written.')


def predict(
    workdir: pathlib.Path,
    config: ml_collections.ConfigDict,
    output_filepath: str,
) -> None:
  """Generates model predictions using the best available checkpoint."""

  # Set seed for reproducibility.
  tf.random.set_seed(config.seed)

  if config.train.get('use_mixed_precision', False):
    tf.keras.mixed_precision.set_global_policy('mixed_float16')

  # Load the best checkpoint
  config, model = model_utils.get_model(config)
  best_checkpoint = tf.train.latest_checkpoint(workdir / 'checkpoints')
  print(f'Loading model from {best_checkpoint}...')
  model.load_weights(best_checkpoint).expect_partial()

  predict_ds = input_pipeline.build_predict_dataset(
      config.dataset,
      config.outcomes,
      cache=config.dataset.get('use_cache', False))

  batched_predictions = predict_utils.generate_predictions(model, predict_ds)

  predictions_df = predict_utils.merge_batched_predictions(batched_predictions)

  # Write the DataFrame to a CSV.
  with open(output_filepath, mode='wt') as f:
    predictions_df.to_csv(f, sep=',', index=False)


def main(unused_argv):
  print('Generating predictions with the following config:')
  print(FLAGS.config)
  predict(
      workdir=pathlib.Path(FLAGS.workdir),
      config=FLAGS.config,
      output_filepath=FLAGS.output_filepath)


if __name__ == '__main__':
  flags.mark_flags_as_required(['config', 'workdir'])
  app.run(main)
