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
"""Program for running DeepNull in series across folds."""

from typing import Sequence
from absl import app
from absl import flags
from absl import logging
from deepnull import config
from deepnull import data
from deepnull import metrics
from deepnull import train_eval
from ml_collections.config_flags import config_flags
import tensorflow as tf

_INPUT_TSV = flags.DEFINE_string('input_tsv', None,
                                 'Path to input PLINK/BOLT-formatted TSV')
_OUTPUT_TSV = flags.DEFINE_string(
    'output_tsv', None,
    'Path to output PLINK/BOLT-formatted TSV that contains predictions.')
_TARGET = flags.DEFINE_string('target', None, 'Target field to predict.')
_COVARIATES = flags.DEFINE_list('covariates', None,
                                'List of covariates to use to predict target.')
_MISSING_VALUE = flags.DEFINE_string(
    'missing_value', 'NA', 'Value used to encode missingness in input TSV.')
_PREDS_COL = flags.DEFINE_string(
    'preds_col', None,
    'Name to use for the DeepNull prediction column. If unspecified, will be '
    'the target column name with "_deepnull" suffix added.')
_NUM_FOLDS = flags.DEFINE_integer(
    'num_folds', 5, 'The number of cross-validation folds to use.')
_SEED = flags.DEFINE_integer('seed', None, 'Random seed to use.')
_LOGDIR = flags.DEFINE_string('logdir', '/tmp',
                              'Directory in which to write temporary outputs.')
_VERBOSE = flags.DEFINE_boolean(
    'verbose', False, 'If True, prints verbose model training output.')
_MODEL_CONFIG = config_flags.DEFINE_config_file(
    'model_config', None,
    'Specifies the model config file to use. If unspecified, defaults to the '
    'MLP-based TF model used for all main results of the DeepNull paper.')


def main(argv: Sequence[str]) -> None:
  if len(argv) > 1:
    raise app.UsageError('Too many command-line arguments.')

  tf.random.set_seed(_SEED.value)
  logging.info('Loading data from %s', _INPUT_TSV.value)
  input_df, binary_col_map = data.load_plink_or_bolt_file(
      path_or_buf=_INPUT_TSV.value, missing_value=_MISSING_VALUE.value)

  if _MODEL_CONFIG.value is None:
    full_config = config.get_config('deepnull')
  else:
    full_config = _MODEL_CONFIG.value

  logging.info('Training DeepNull model on %s with model %s', _TARGET.value,
               full_config)
  final_df, eval_metrics, _ = train_eval.create_deepnull_prediction(
      input_df=input_df,
      target=_TARGET.value,
      target_is_binary=_TARGET.value in binary_col_map,
      covariates=_COVARIATES.value,
      full_config=full_config,
      prediction_column=_PREDS_COL.value,
      num_folds=_NUM_FOLDS.value,
      seed=_SEED.value,
      logdir=_LOGDIR.value,
      # Level 2 is printing once per epoch during training.
      verbosity=2 if _VERBOSE.value else 0,
  )

  if not metrics.acceptable_model_performance(eval_metrics):
    logging.warning(
        'WARNING: data folds have substantially different performance. Consider'
        ' retraining model with a different seed.')

  logging.info('Writing trained results to %s', _OUTPUT_TSV.value)
  data.write_plink_or_bolt_file(
      input_df=final_df,
      path_or_buf=_OUTPUT_TSV.value,
      binary_column_mapping=binary_col_map,
      missing_value=_MISSING_VALUE.value,
      cast_ints=True)


if __name__ == '__main__':
  flags.mark_flags_as_required(
      ['input_tsv', 'output_tsv', 'target', 'covariates'])
  app.run(main)
