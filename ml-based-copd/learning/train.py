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
"""Loads and trains a `tf.keras.Model` given an experiment `ConfigDict`."""
import pathlib
from typing import List, Optional

import ml_collections
import tensorflow as tf

import callbacks
import dataset_util
import head_utils
import input_pipeline
import models
import optimizers
import predict_utils


def generate_and_write_predictions(work_dir: pathlib.Path,
                                   model: tf.keras.Model,
                                   data_type: dataset_util.DataType,
                                   dataset_config: ml_collections.ConfigDict,
                                   filename: str) -> None:
  """Writes a full predictions TSV for the given data type to the workdir."""
  # Load the latest checkpoint.
  model = predict_utils.load_latest_weights(
      model, work_dir / callbacks.CHECKPOINT_SUBDIR)

  # Generate and write predictions for the full fold for the given dataset type.
  fold_all_ds = input_pipeline.config_to_dataset(
      dataset_config,
      data_type,
      fold_override=dataset_util.Fold.ALL,
      cache_override=False)
  predict_utils.write_predictions_tsv(
      model, fold_all_ds, work_dir / filename, id_key=input_pipeline.EID)


def config_to_model(config: ml_collections.ConfigDict) -> tf.keras.Model:
  """Returns a compiled model parameterized by the backbone and head configs."""
  head_configs = config['head_configs']
  backbone_config = config['backbone_config']
  optimizer_config = config['optimizer_config']
  loss_dict, loss_weights = head_utils.get_losses(head_configs)
  model = models.get_model(backbone_config, head_configs)
  model.compile(
      optimizer=optimizers.get_optimizer(optimizer_config),
      loss=loss_dict,
      loss_weights=loss_weights,
      metrics=head_utils.get_metrics(head_configs))
  return model


def train(
    work_dir: str,
    config: ml_collections.ConfigDict,
    additional_callbacks: Optional[List[tf.keras.callbacks.Callback]] = None,
) -> None:
  """Load a model and datasets parameterized by the config and run training."""
  work_dir = pathlib.Path(work_dir)
  train_config = config['train_config']

  # Set global tensorflow state.
  tf.random.set_seed(train_config['seed'])
  if train_config['use_mixed_precision']:
    tf.keras.mixed_precision.set_global_policy('mixed_float16')

  model = config_to_model(config)

  callbacks_config = config['callbacks_config']
  train_callbacks = callbacks.get_callbacks(work_dir, callbacks_config)
  if additional_callbacks:
    train_callbacks.extend(additional_callbacks)

  dataset_config = config['dataset_config']
  train_ds = input_pipeline.config_to_dataset(dataset_config,
                                              dataset_util.DataType.TRAINING)
  validation_ds = input_pipeline.config_to_dataset(
      dataset_config, dataset_util.DataType.VALIDATION)

  _ = model.fit(
      train_ds,
      validation_data=validation_ds,
      epochs=train_config['num_epochs'],
      callbacks=train_callbacks,
      verbose=train_config['fit_verbose'])

  # Generate predictions for all folds on the training and validation datasets.
  generate_and_write_predictions(work_dir, model,
                                 dataset_util.DataType.TRAINING, dataset_config,
                                 'train_predictions.tsv')
  generate_and_write_predictions(work_dir, model,
                                 dataset_util.DataType.VALIDATION,
                                 dataset_config, 'validation_predictions.tsv')


def predict(work_dir: str, config: ml_collections.ConfigDict) -> None:
  """Load the config's model and datasets and generate predictions.

  Prediction TSVs are written to "{workdir}/train_predictions.tsv" and
  "{workdir}/validation_predictions.tsv" for the `Fold.ALL` training and
  validation datasets, respectively.

  Args:
    work_dir: The work directory from which to load the model checkpoint.
    config: The configuration file used to parameterize the model and dataset.
  """
  work_dir = pathlib.Path(work_dir)
  train_config = config['train_config']

  # Set global tensorflow state.
  tf.random.set_seed(train_config['seed'])
  if train_config['use_mixed_precision']:
    tf.keras.mixed_precision.set_global_policy('mixed_float16')

  model = config_to_model(config)
  dataset_config = config['dataset_config']

  # Generate predictions for all folds on the training and validation datasets.
  generate_and_write_predictions(work_dir, model,
                                 dataset_util.DataType.TRAINING, dataset_config,
                                 'train_predictions.tsv')
  generate_and_write_predictions(work_dir, model,
                                 dataset_util.DataType.VALIDATION,
                                 dataset_config, 'validation_predictions.tsv')
