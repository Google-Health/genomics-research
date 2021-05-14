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
from absl.testing import absltest
from absl.testing import parameterized
import tensorflow as tf
from deepnull import metrics
from deepnull import model as model_lib


class ModelTest(parameterized.TestCase):

  @parameterized.parameters(
      dict(
          cls=model_lib.QuantitativeDeepNull,
          optimization_metric=metrics.get_optimization_metric('mse')),
      dict(
          cls=model_lib.BinaryDeepNull,
          optimization_metric=metrics.get_optimization_metric('crossentropy')),
  )
  def test_model_compiles(self, cls, optimization_metric):
    model = cls(
        feature_columns=[
            tf.feature_column.numeric_column('cov1', dtype=tf.float32),
            tf.feature_column.numeric_column('cov2', dtype=tf.float32),
        ],
        mlp_units=[64, 64, 32],
        mlp_activation='relu',
        optimization_metric=optimization_metric)

    model.compile(
        loss=model.loss_function(),
        optimizer=tf.keras.optimizers.Adam(
            learning_rate=1e-3,
            beta_1=0.99,
            beta_2=0.999,
        ),
        metrics=model.metrics_to_use())
    self.assertIsNotNone(model)


if __name__ == '__main__':
  absltest.main()
