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
"""Tests for deepnull.main."""
import os
import tempfile
from unittest import mock
from absl import flags
from absl.testing import absltest
from absl.testing import flagsaver
import ml_collections
import numpy as np
import pandas as pd
from deepnull import config
from deepnull import main

FLAGS = flags.FLAGS

_DEFAULT_CONFIG = ml_collections.ConfigDict({
    'model_type':
        'deepnull',
    'model_config':
        ml_collections.ConfigDict({
            'mlp_units': (32, 16, 8),
            'mlp_activation': 'relu',
        }),
    'optimizer_config':
        ml_collections.ConfigDict({
            'learning_rate_batch_1024': 1e-4,
            'beta_1': 0.9,
            'beta_2': 0.99,
            'optimization_metric': '',
        }),
    'training_config':
        ml_collections.ConfigDict({
            'num_epochs': 2,
            'batch_size': 512,
        }),
})


def _create_df(size: int):
  ids = np.arange(size)
  feature_1 = np.random.random(size)
  feature_2 = np.random.random(size)
  err = 0.1 * np.random.random(size)
  val = feature_1 + feature_2 + feature_2**2 + err
  return pd.DataFrame(
      {
          'FID': ids,
          'IID': ids,
          'cov1': feature_1,
          'cov2': feature_2,
          'label': val
      },
      columns=['FID', 'IID', 'cov1', 'cov2', 'label'])


class MainTest(absltest.TestCase):

  def test_end_to_end_default_config(self):
    with tempfile.TemporaryDirectory() as tmpdir:
      input_filename = os.path.join(tmpdir, 'input.tsv')
      _create_df(size=1024).to_csv(input_filename, sep='\t', index=False)
      input_df = pd.read_csv(input_filename, sep='\t')
      output_filename = os.path.join(tmpdir, 'output.tsv')

      with flagsaver.flagsaver():
        FLAGS.input_tsv = input_filename
        FLAGS.output_tsv = output_filename
        FLAGS.target = 'label'
        FLAGS.covariates = ['cov1', 'cov2']
        FLAGS.num_folds = 3
        FLAGS.seed = 234
        FLAGS.logdir = tmpdir

        with mock.patch.object(config, 'get_config', autospec=True) as conf:
          conf.return_value = _DEFAULT_CONFIG
          # Run the e2e test.
          main.main(['main.py'])

          # Load the results.
          output_df = pd.read_csv(output_filename, sep='\t')

    # Compare the results of input and output df.
    input_columns = set(input_df.columns)
    output_columns = set(output_df.columns)
    self.assertEqual(input_columns & output_columns, input_columns)
    self.assertEqual(output_columns - input_columns, {'label_deepnull'})

    shared_columns = sorted(input_columns & output_columns)
    pd.testing.assert_frame_equal(input_df[shared_columns],
                                  output_df[shared_columns])

  # TODO: Figure out how to specify a --model_config flag as a string that will
  # be parsed appropriately within the test. That would enable testing an
  # explicit config as well as the default config above.


if __name__ == '__main__':
  absltest.main()
