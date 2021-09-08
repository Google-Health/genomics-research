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
"""Tests for deepnull.model."""
import itertools
import math
from typing import Tuple
from absl.testing import absltest
from absl.testing import parameterized
from deepnull import config
from deepnull import model as model_lib
import numpy as np
import pandas as pd
import tensorflow as tf


def _create_test_data() -> Tuple[pd.DataFrame, pd.DataFrame]:
  """Returns train and eval DataFrames for testing fit and predict."""
  num_samples = 1000
  rng = np.random.default_rng(1234)
  cov1 = rng.uniform(0, 1, num_samples)
  cov2 = 0.1 * rng.random(num_samples)
  err = 0.05 * rng.random(num_samples)
  cont_target = cov1 + cov2 + (cov1**2) * cov2 + err
  threshold = np.percentile(cont_target, 80)
  binary_target = (cont_target > threshold).astype(int)

  full_df = pd.DataFrame(
      data={
          'FID': [0] * num_samples,
          'IID': np.arange(num_samples),
          'cov1': cov1,
          'cov2': cov2,
          'cont_target': cont_target,
          'binary_target': binary_target
      })
  train_df = full_df.iloc[:800].copy()
  eval_df = full_df.iloc[800:].copy()
  return train_df, eval_df


def _accuracy(y_true: np.ndarray, y_prob: np.ndarray) -> float:
  """Simple accuracy computation for evaluating binary models."""
  # We define this to avoid importing sklearn solely for its metrics.
  y_pred = y_prob > 0.5
  return (y_true == y_pred).sum() / len(y_true)


class ModelTest(parameterized.TestCase):

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

    ds = model_lib._make_input_fn(
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
    ds = model_lib._make_input_fn(
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
      dict(cls=model_lib.QuantitativeDeepNull),
      dict(cls=model_lib.BinaryDeepNull),
  )
  def test_deepnull_model_compiles(self, cls):
    full_config = config.get_config(config.DEEPNULL)
    model = cls(
        target='target',
        covariates=['cov1', 'cov2'],
        full_config=full_config,
        fold_ix=0)
    self.assertIsInstance(model, model_lib._DeepNull)

  @parameterized.parameters(
      dict(
          cls=model_lib.QuantitativeDeepNull,
          metric=(lambda x, y: np.corrcoef(x, y)[0, 1]),
          target='cont_target',
          expected=0.98433785),
      dict(
          cls=model_lib.BinaryDeepNull,
          metric=_accuracy,
          target='binary_target',
          expected=0.195),
  )
  def test_deepnull_model_fit_and_predict(self, cls, metric, target, expected):
    train_df, eval_df = _create_test_data()
    full_config = config.get_config(config.DEEPNULL)
    full_config.model_config.mlp_units = (32, 16)
    full_config.training_config.num_epochs = 2
    full_config.training_config.batch_size = 200
    model = cls(
        target=target,
        covariates=['cov1', 'cov2'],
        full_config=full_config,
        fold_ix=0)

    tf.random.set_seed(42)
    model.fit(train_df=train_df, eval_df=eval_df, verbosity=0)
    actual_df = model.predict(
        df=eval_df, prediction_column='deepnull_prediction')
    actual_metric = metric(eval_df[target], actual_df['deepnull_prediction'])

    self.assertEqual(actual_df.shape, (200, 2))
    self.assertEqual(list(actual_df.columns), ['IID', 'deepnull_prediction'])
    self.assertAlmostEqual(actual_metric, expected)

  @parameterized.parameters(
      dict(
          cls=model_lib.QuantitativeXGBoost,
          metric=(lambda x, y: np.corrcoef(x, y)[0, 1]),
          target='cont_target',
          expected=0.997212255),
      dict(
          cls=model_lib.BinaryXGBoost,
          metric=_accuracy,
          target='binary_target',
          expected=0.985),
  )
  def test_xgboost_model_fit_and_predict(self, cls, metric, target, expected):
    train_df, eval_df = _create_test_data()
    full_config = config.get_config(config.XGBOOST)
    model = cls(
        target=target, covariates=['cov1', 'cov2'], full_config=full_config)
    model.fit(train_df=train_df, eval_df=eval_df, verbosity=0)
    actual_df = model.predict(
        df=eval_df, prediction_column='xgboost_prediction')
    actual_metric = metric(eval_df[target], actual_df['xgboost_prediction'])

    self.assertEqual(actual_df.shape, (200, 2))
    self.assertEqual(list(actual_df.columns), ['IID', 'xgboost_prediction'])
    self.assertAlmostEqual(actual_metric, expected)

  @parameterized.parameters(
      dict(
          config_name=config.DEEPNULL,
          binary=True,
          expected=model_lib.BinaryDeepNull),
      dict(
          config_name=config.DEEPNULL,
          binary=False,
          expected=model_lib.QuantitativeDeepNull),
      dict(
          config_name=config.XGBOOST,
          binary=True,
          expected=model_lib.BinaryXGBoost),
      dict(
          config_name=config.XGBOOST,
          binary=False,
          expected=model_lib.QuantitativeXGBoost),
  )
  def test_get_model(self, config_name, binary, expected):
    full_config = config.get_config(config_name)
    actual = model_lib.get_model(
        target='target',
        target_is_binary=binary,
        covariates=['cov1', 'cov2'],
        full_config=full_config,
        fold_ix=0,
        logdir='/tmp',
        seed=1)
    self.assertIsInstance(actual, expected)


if __name__ == '__main__':
  absltest.main()
