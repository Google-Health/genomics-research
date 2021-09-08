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
"""Tests for deepnull.metrics."""
from absl.testing import absltest
from absl.testing import parameterized
from deepnull import metrics


class MetricsTest(parameterized.TestCase):

  @parameterized.parameters(
      dict(metric_name='tf_pearson', values=[-0.1], expected=False),
      dict(metric_name='tf_pearson', values=[-0.1, 1.0], expected=False),
      dict(metric_name='tf_pearson', values=[0.03, 0.02, 0.04], expected=False),
      dict(metric_name='tf_pearson', values=[0.03, 0.9], expected=False),
      dict(metric_name='tf_pearson', values=[0.8, 0.82, 0.85], expected=True),
      dict(metric_name='pearson', values=[-0.1], expected=False),
      dict(metric_name='pearson', values=[-0.1, 1.0], expected=False),
      dict(metric_name='pearson', values=[0.03, 0.02, 0.04], expected=False),
      dict(metric_name='pearson', values=[0.03, 0.9], expected=False),
      dict(metric_name='pearson', values=[0.8, 0.82, 0.85], expected=True),
      dict(metric_name='auroc', values=[-0.1], expected=False),
      dict(metric_name='auroc', values=[-0.1, 1.0], expected=False),
      dict(metric_name='auroc', values=[0.03, 0.02, 0.04], expected=False),
      dict(metric_name='auroc', values=[0.03, 0.9], expected=False),
      dict(metric_name='auroc', values=[0.8, 0.82, 0.85], expected=True),
      dict(metric_name='auc', values=[-0.1], expected=False),
      dict(metric_name='auc', values=[-0.1, 1.0], expected=False),
      dict(metric_name='auc', values=[0.03, 0.02, 0.04], expected=False),
      dict(metric_name='auc', values=[0.03, 0.9], expected=False),
      dict(metric_name='auc', values=[0.8, 0.82, 0.85], expected=True),
  )
  def test_acceptable_model_performance(self, metric_name, values, expected):
    """Test identification of acceptable model performance."""
    eval_metrics = [{metric_name: value} for value in values]
    actual = metrics.acceptable_model_performance(eval_metrics)
    self.assertEqual(actual, expected)


if __name__ == '__main__':
  absltest.main()
