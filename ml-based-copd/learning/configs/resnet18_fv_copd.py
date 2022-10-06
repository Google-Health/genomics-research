# Copyright 2022 Google LLC
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#      http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
"""Demo configuration file for a baseline ResNet18 model."""
import ml_collections

from configs import config_utils


def get_config() -> ml_collections.ConfigDict:
  """Get the default hyperparameter configuration for a ResNet18 model.

  Trains a ResNet18 model to predict copd status from flow-volume spirograms.

  Returns:
    A ConfigDict parameterizing the experiment.
  """
  config = ml_collections.ConfigDict()

  # Note: Columns in `dataset_config.inputs` and `dataset_config.labels` are
  # included in the input pipeline's `inputs` and `labels` dictionaries but are
  # not necessarily passed to the model as input or used in the loss function
  # unless they are explicitly consumed by the model or included in as a head.
  config.dataset_config = ml_collections.ConfigDict({
      'inputs': {
          'eid',
          'flow_volume',
      },
      'labels': {'copd',},
      'batch_size': 256,
      'cache': True,
      'fold': 'ALL',
      'use_feature_scaling': True,
      'data_dir': ('/path/to/pkl/data/'),
      'version_suffix': 'v07',
      # `subsample_train_percent` is initialized with a placeholder so that it
      # can be set to integer values using `ml_collections.config_flag`
      # overrides.
      'subsample_train_percent': ml_collections.config_dict.placeholder(int),
  })

  config.head_configs = {
      'copd':
          config_utils.binary_head_config(
              'copd',
              1.0,
              additional_activation='swish',
              additional_kernel_l2=1e-7,
          ),
  }

  config.backbone_config = ml_collections.ConfigDict({
      'class_name': 'ResNet18',
      'kwargs': {
          'model_name': 'ResNet18',
          'input_names': 'flow_volume',
          'input_shape': (1000,),
          'conditional_input_names': '',
          'conditional_input_shape': (1,),
          'kernel_l2': 1e-07,
      },
  })

  config.optimizer_config = ml_collections.ConfigDict({
      'class_name': 'Adam',
      'kwargs': {
          'learning_rate': 0.000475177,
      }
  })

  config.train_config = ml_collections.ConfigDict({
      'use_mixed_precision': True,
      'num_epochs': 1000,
      'fit_verbose': 1,
      'seed': ml_collections.config_dict.placeholder(int),
  })
  # Seed is originally initialized with a placeholder so that it can be set to
  # `None` using `ml_collections.config_flag` overrides.
  config.train_config.seed = 42

  config.callbacks_config = ml_collections.ConfigDict({
      'checkpoint_best': True,
      'use_early_stopping': True,
      'early_stopping_patience': 50,
  })

  return config
