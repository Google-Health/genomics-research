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
"""Utilities for generating prediction `pd.DataFrames`."""
import pathlib
from typing import Any, Dict, List, Type

import numpy as np
import pandas as pd
import tensorflow as tf


def load_latest_weights(model: tf.keras.Model,
                        checkpoint_dir: pathlib.Path) -> tf.keras.Model:
  """Loads model weights from the latest checkpoint in `checkpoint_dir`."""
  best_checkpoint = tf.train.latest_checkpoint(checkpoint_dir)
  # Calls `expect_partial` on the loaded status to avoid printing out a lot of
  # warnings about unused keys at exit time. Since the checkpoints saved from
  # tf.keras often generate extra keys in the checkpoint, we expect unused keys.
  _ = model.load_weights(best_checkpoint).expect_partial()
  return model


def batch_predict(model: tf.keras.Model,
                  dataset: tf.data.Dataset,
                  id_key: str,
                  id_type: Type[Any] = int) -> Dict[str, List[np.ndarray]]:
  """Returns a dictionary of batched predictions keyed on outcome.

  Args:
    model: The model used to generate predictions.
    dataset: The dataset used to generate predictions. Note: the dataset is
      expected to yield `(inputs, targets, weights)` elements of type
      `Tuple[Dict[str, tf.Tensor], Dict[str, tf.Tensor], Dict[str, tf.Tensor]]`.
    id_key: The input dictionary's key for the example's unique ID.
    id_type: The type of `id_key`; used to cast `id_key`'s value.

  Returns:
    A dictionary mapping outcome keys to batched model output.

  Raises:
    ValueError: The dataset doesn't yield `(inputs, targets, weights)` tuples.
    ValueError: The dataset's elements are not dictionaries.
    ValueError: The `id_key` is missing from the input dictionary's spec.
  """
  # Validate that the dataset yields elements of the expected type.
  element_spec = dataset.element_spec
  if len(element_spec) != 3:
    raise ValueError('Provided dataset does not yield '
                     f'"(inputs, targets, weights)" tuples: {element_spec}')
  for element in element_spec:
    if not isinstance(element, dict):
      raise ValueError('Provided dataset does not yield dictionary elements: '
                       f'{element}')
  if id_key not in element_spec[0]:
    raise ValueError(f'{id_key} not in the input dictionary: {element_spec[0]}')

  # Iterate over the dataset, accumulating model predictions for each outcome.
  # Note: "output_names" is a public attribute of `tf.keras.Model` and denotes a
  # list of string names for model outputs. The order of "output_names"
  # corresponds to the order of output tensors returned by `model()`.
  output_names = model.output_names
  is_single_headed = len(output_names) == 1
  predict_dict = {name: [] for name in [id_key] + output_names}
  for batch_input, _, _ in dataset:
    predict_dict[id_key].append(batch_input[id_key].numpy().astype(id_type))
    # Note: this check is required since a multi-headed TensorFlow model returns
    # a list of output tensors while a single-headed model returns a single
    # output tensor (rather than a list of size 1).
    model_output = model(batch_input)
    if is_single_headed:
      predict_dict[output_names[0]].append(model_output.numpy())
    else:
      for name, output_tensor in zip(output_names, model_output):
        predict_dict[name].append(output_tensor.numpy())

  return predict_dict


def merge_batch_predictions(batch_predictions: Dict[str, List[np.ndarray]],
                            id_key: str) -> pd.DataFrame:
  """Creates a pd.DataFrame containing outcome predictions for each `id_key`.

  Args:
    batch_predictions: A dictionary mapping outcome keys to batched model
      output.
    id_key: The key denoting the example's unique ID.

  Returns:
    A `pd.DataFrame` containing one column for each outcome.

  Raises:
    ValueError: If `id_key` is not present in `batch_predictions`.
  """
  if id_key not in batch_predictions:
    raise ValueError(f'"batch_predictions" must contain {id_key}: '
                     f'{batch_predictions.keys()}.')

  predictions = {}
  for outcome_name, ndarray_list in batch_predictions.items():
    predictions[outcome_name] = np.concatenate(ndarray_list, axis=None)

  predictions_df = pd.DataFrame(predictions)
  ordered_columns = [id_key]
  ordered_columns.extend(sorted([key for key in predictions if key != id_key]))
  predictions_df = predictions_df[ordered_columns]
  return predictions_df


def write_predictions_tsv(model: tf.keras.Model,
                          dataset: tf.data.Dataset,
                          output_filepath: pathlib.Path,
                          id_key: str,
                          id_type: Type[Any] = int) -> None:
  """Writes a TSV of model predictions for the dataset to `output_filepath`.

  Args:
    model: The model used to generate predictions.
    dataset: The dataset used to generate predictions.
    output_filepath: The output filepath at which to write predictions.
    id_key: The input dictionary's key for the example's unique ID.
    id_type: The type of `id_key`; used to cast `id_key`'s value.
  """
  batch_predictions = batch_predict(model, dataset, id_key, id_type)
  merged_predictions = merge_batch_predictions(batch_predictions, id_key)
  with open(str(output_filepath), 'wt') as f:
    merged_predictions.to_csv(f, sep='\t', index=False)
