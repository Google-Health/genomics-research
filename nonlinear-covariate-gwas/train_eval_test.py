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
import itertools
import math
import tempfile
from absl.testing import absltest
from absl.testing import parameterized
import numpy as np
import pandas as pd
import tensorflow as tf
from deepnull import model as model_lib
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
      dict(num_epochs=1, batch_size=16),
      dict(num_epochs=None, batch_size=16),
      dict(num_epochs=2, batch_size=32),
      dict(num_epochs=13, batch_size=32),
  )
  def test_make_training_input_fn(self, num_epochs, batch_size):
    features_df = pd.DataFrame({
        'cov1': np.arange(100),
        'cov2': np.arange(0.5, 100.5)
    })
    labels = pd.Series([0] * 50 + [1] * 50)
    # For deterministic shuffling.
    tf.random.set_seed(123)

    ds = train_eval._make_input_fn(
        features_df,
        labels,
        num_epochs=num_epochs,
        is_training=True,
        batch_size=batch_size)

    if num_epochs is not None:
      expected_num_batches = 100 * num_epochs // batch_size
      values = list(ds.as_numpy_iterator())
    else:
      expected_num_batches = 50
      values = list(
          itertools.islice(ds.as_numpy_iterator(), expected_num_batches))

    # Check that shuffling occurred.
    self.assertNotEqual(list(values[0][1]), [0] * batch_size)
    self.assertLen(values, expected_num_batches)
    for features, labels in values:
      self.assertCountEqual(list(features.keys()), ['cov1', 'cov2'])
      self.assertLen(features['cov1'], batch_size)
      self.assertLen(features['cov2'], batch_size)
      self.assertIsInstance(labels, np.ndarray)
      self.assertLen(labels, batch_size)

  @parameterized.parameters(
      dict(batch_size=16),
      dict(batch_size=32),
      dict(batch_size=100),
      dict(batch_size=300),
  )
  def test_make_predict_input_fn(self, batch_size):
    features_df = pd.DataFrame({
        'cov1': np.arange(100),
        'cov2': np.arange(0.5, 100.5)
    })
    ds = train_eval._make_input_fn(
        features_df=features_df,
        labels=None,
        num_epochs=1,
        is_training=False,
        batch_size=batch_size)

    values = list(ds.as_numpy_iterator())
    self.assertLen(values, math.ceil(100 / batch_size))
    for i, features in enumerate(values):
      self.assertCountEqual(list(features.keys()), ['cov1', 'cov2'])
      np.testing.assert_array_equal(
          features['cov1'],
          np.arange(i * batch_size, min(100, (i + 1) * batch_size)))
      np.testing.assert_array_equal(
          features['cov2'],
          np.arange(i * batch_size + 0.5, min(100, (i + 1) * batch_size + 0.5)))

  @parameterized.parameters(
      dict(
          target_is_binary=False,
          expected_metrics=['loss', 'mse', 'tf_pearson']),
      dict(
          target_is_binary=True,
          expected_metrics=[
              'accuracy', 'auprc', 'auroc', 'crossentropy', 'loss'
          ]),
  )
  def test_train_binary_deepnull_model(self, target_is_binary,
                                       expected_metrics):
    size = 1000
    design_df = _create_df(target_is_binary=target_is_binary, size=size)
    train_df = design_df.iloc[:800]
    eval_df = design_df.iloc[800:]

    with tempfile.TemporaryDirectory() as tmpdir:
      model, history, eval_performance = train_eval.train_deepnull_model(
          train_df=train_df,
          eval_df=eval_df,
          target='label',
          target_is_binary=target_is_binary,
          covariates=['cov1', 'cov2'],
          model_params=model_lib.ModelParameters(
              batch_size=32, mlp_units=[32, 16], num_epochs=2),
          logdir=tmpdir,
          verbosity=0,
      )

    self.assertIsInstance(model, tf.keras.models.Model)
    self.assertIsInstance(history, tf.keras.callbacks.History)
    self.assertCountEqual(eval_performance.keys(), expected_metrics)

  @parameterized.parameters(
      dict(target_is_binary=False),
      dict(target_is_binary=True),
  )
  def test_predict(self, target_is_binary):
    design_df = _create_df(target_is_binary=target_is_binary, size=1000)
    train_df = design_df.iloc[:600]
    eval_df = design_df.iloc[600:800]
    test_df = design_df.iloc[800:].copy(deep=True)
    test_df['IID'] = np.arange(len(test_df))

    with tempfile.TemporaryDirectory() as tmpdir:
      model, _, _ = train_eval.train_deepnull_model(
          train_df=train_df,
          eval_df=eval_df,
          target='label',
          target_is_binary=target_is_binary,
          covariates=['cov1', 'cov2'],
          model_params=model_lib.ModelParameters(
              batch_size=32, mlp_units=[32, 16], num_epochs=1),
          logdir=tmpdir,
          verbosity=0,
      )

    actual = train_eval.predict(
        deepnull_model=model,
        df=test_df,
        covariates=['cov1', 'cov2'],
        prediction_column='deepnull_pred')
    self.assertCountEqual(actual.columns, ['IID', 'deepnull_pred'])
    pd.testing.assert_series_equal(actual.IID, test_df.IID)
    self.assertEqual(actual.deepnull_pred.isnull().sum(), 0)

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
    with tempfile.TemporaryDirectory() as tmpdir:
      final_df, _, _, test_perf_df = train_eval.create_deepnull_prediction(
          input_df=input_df,
          target='label',
          target_is_binary=target_is_binary,
          covariates=['cov1', 'cov2'],
          prediction_column='label_deepnull',
          num_folds=3,
          model_params=model_lib.ModelParameters(
              batch_size=32, mlp_units=[32, 16], num_epochs=1),
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
