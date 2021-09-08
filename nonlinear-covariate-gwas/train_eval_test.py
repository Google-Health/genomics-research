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
"""Tests for deepnull.train_eval."""
import tempfile
from absl.testing import absltest
from absl.testing import parameterized
import numpy as np
import pandas as pd
from deepnull import config
from deepnull import train_eval


def _create_df(target_is_binary: bool, size: int = 1000):
  feature_1 = np.random.random(size)
  feature_2 = np.random.random(size)
  err = 0.1 * np.random.random(size)
  val = feature_1 + feature_2 + feature_2**2 + err
  if target_is_binary:
    val = (val < 1.3).astype(int)
  return pd.DataFrame({'cov1': feature_1, 'cov2': feature_2, 'label': val})


class TrainEvalTest(parameterized.TestCase):

  @parameterized.parameters(
      dict(target_is_binary=False),
      dict(target_is_binary=True),
  )
  def test_create_deepnull_prediction(self, target_is_binary):
    size = 1000
    design_df = _create_df(target_is_binary=target_is_binary, size=size)
    design_df['FID'] = np.arange(size)
    design_df['IID'] = np.arange(size)
    design_df['unused_str_column'] = np.random.choice(list('abcdefg'), size)

    input_df = design_df.copy(deep=True)
    full_config = config.get_config(config.DEEPNULL)
    full_config.model_config.mlp_units = (32, 16)
    full_config.training_config.num_epochs = 1
    full_config.training_config.batch_size = 32

    with tempfile.TemporaryDirectory() as tmpdir:
      final_df, _, test_perf_df = train_eval.create_deepnull_prediction(
          input_df=input_df,
          target='label',
          target_is_binary=target_is_binary,
          covariates=['cov1', 'cov2'],
          full_config=full_config,
          prediction_column='label_deepnull',
          num_folds=3,
          seed=5,
          logdir=tmpdir,
          verbosity=0)

    pd.testing.assert_frame_equal(design_df, input_df)
    self.assertCountEqual(final_df.columns,
                          list(design_df.columns) + ['label_deepnull'])
    pd.testing.assert_frame_equal(design_df, final_df[design_df.columns])
    self.assertCountEqual(
        test_perf_df.columns,
        ['IID', 'label', 'label_deepnull', 'label_deepnull_eval_fold'])


if __name__ == '__main__':
  absltest.main()
