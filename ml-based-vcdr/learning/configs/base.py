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
"""Contains the default training configuration."""
import ml_collections


def get_config() -> ml_collections.ConfigDict:
  """Returns the default hyperparameter configuration."""
  config = ml_collections.ConfigDict()

  config.seed = None

  # misc. training
  config.train = ml_collections.ConfigDict({
      'use_mixed_precision': True,
      'use_distributed_training': False,
      'max_num_steps': 250000,
      'log_step_freq': 500,
      'fit_verbose': 1,
      'initial_epoch': 0,
  })

  # dataset and augmentation
  config.dataset = ml_collections.ConfigDict({
      'train': '/mnt/disks/data/train/train.tfrecord*',
      'eval': '/mnt/disks/data/train/eval.tfrecord*',
      'test': '/mnt/disks/data/train/test.tfrecord*',
      'predict': '/mnt/disks/data/predict/predict.tfrecord*',
      'num_train_examples': 79355,
      'batch_size': 16,
      'image_size': (587, 587),
      'random_horizontal_flip': True,
      'random_vertical_flip': True,
      'random_brightness_max_delta': 0.1147528,
      'random_saturation_lower': 0.5597273,
      'random_saturation_upper': 1.2748845,
      'random_hue_max_delta': 0.0251488,
      'random_contrast_lower': 0.9996807,
      'random_contrast_upper': 1.7704824,
      'use_cache': False,
  })

  # model architecture
  config.model = ml_collections.ConfigDict({
      'backbone': 'inceptionv3',
      'backbone_drop_rate': 0.2,
      'input_shape': (587, 587, 3),
      'weights': 'imagenet',
      'weight_decay': 0.00004,
  })

  # optimizer
  config.opt = ml_collections.ConfigDict({
      'optimizer': 'adam',
      'learning_rate': 0.001,
      'beta_1': 0.9,
      'beta_2': 0.999,
      'epsilon': 0.1,
      'amsgrad': False,
      'use_model_averaging': True,
      'update_model_averaging_weights': False,
  })

  config.schedule = ml_collections.ConfigDict({
      'schedule': 'exponential',
      'epochs_per_decay': 2,
      'decay_rate': 0.99,
      'staircase': True,
  })

  config.outcomes = [
      ml_collections.ConfigDict({
          'name': 'vertical_cup_to_disc',
          'type': 'regression',
          'num_classes': 1,
          'loss': 'mse',
          'loss_weight': 1.0,
      }),
      ml_collections.ConfigDict({
          'name': 'vertical_cd_visibility',
          'type': 'classification',
          'num_classes': 3,
          'loss': 'ce',
          'loss_weight': 1.0,
      }),
      ml_collections.ConfigDict({
          'name': 'glaucoma_suspect_risk',
          'type': 'classification',
          'num_classes': 4,
          'loss': 'ce',
          'loss_weight': 1.0,
      }),
      ml_collections.ConfigDict({
          'name': 'glaucoma_gradability',
          'type': 'classification',
          'num_classes': 3,
          'loss': 'ce',
          'loss_weight': 1.0,
      }),
  ]

  return config
