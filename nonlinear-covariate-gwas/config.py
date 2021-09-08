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
"""Configuration for all model types.

This configuration file is used to specify all different supported types of
models for training DeepNull. The configuration is parsed by model.py for the
proper instantiation of the selected model.

See https://github.com/google/ml_collections for details on ConfigDict.
"""
import ml_collections

# Valid model types.
# The model used for the main figures in the paper.
DEEPNULL = 'deepnull'
# XGBoost-based models.
XGBOOST = 'xgboost'


def get_config(config_name: str) -> ml_collections.ConfigDict:
  """Returns the config specified by `config_name`."""
  supported_models = {
      DEEPNULL:
          ml_collections.ConfigDict({
              'model_type':
                  DEEPNULL,
              'model_config':
                  ml_collections.ConfigDict({
                      # The MLP units for the nonlinear path of DeepNull.
                      'mlp_units': (64, 64, 32, 16),
                      # The activation function to use. See
                      # https://keras.io/api/layers/activations.
                      'mlp_activation': 'relu',
                  }),
              'optimizer_config':
                  ml_collections.ConfigDict({
                      # Learning rate for a batch size of 1024. The actual
                      # learning rate used is scaled linearly as
                      # `learning_rate * batch_size / 1024`.
                      'learning_rate_batch_1024': 1e-4,
                      # Betas for the Adam optimizer.
                      'beta_1': 0.9,
                      'beta_2': 0.99,
                      # The optimization metric to use to select the best model
                      # checkpoint. This must be a metric generated during
                      # training (which depends on whether the target is a
                      # binary or continuous variable). If unspecified, the
                      # default metric for the associated target type is used.
                      'optimization_metric': '',
                  }),
              'training_config':
                  ml_collections.ConfigDict({
                      # Number of full passes through the training data.
                      'num_epochs': 1000,
                      # Number of training examples per batch.
                      'batch_size': 1024,
                  }),
          }),
      XGBOOST:
          ml_collections.ConfigDict({
              'model_type':
                  XGBOOST,
              'model_config':
                  ml_collections.ConfigDict({
                      # See
                      # https://xgboost.readthedocs.io/en/latest/parameter.html
                      # for full details on all parameters.
                      # The target objective. If unspecified, will be the
                      # default objective for the type of model prediction (i.e.
                      # regression vs classification).
                      'objective': '',
                      'max_depth': 3,
                      'eta': 0.32,
                      'alpha': 0.658,
                      'lambda': 2.0,
                      # If unspecified, will be the default metric for the type
                      # of model prediction.
                      'eval_metric': '',
                  }),
              'training_config':
                  ml_collections.ConfigDict({
                      'num_boost_round': 25,
                  }),
          }),
  }

  if config_name not in supported_models:
    raise ValueError(f'Config "{config_name}" is not a supported model: '
                     f'{sorted(supported_models)}')

  return supported_models[config_name]
