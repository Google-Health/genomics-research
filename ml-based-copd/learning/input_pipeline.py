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
""""Input utilities for loading spirometry dataframes as TensorFlow datasets."""
import pathlib
from typing import Callable, Dict, Iterable, List, Mapping, Optional, Set, Tuple

import ml_collections
import numpy as np
import pandas as pd
import tensorflow as tf

import dataset_util

# Named input and label columns.
EID = 'eid'

# The list of supported array dataframe columns for use as inputs. These column
# values are assumed to be a rank-1 tensor of shape `(1000,)`. We assume that
# any column not specified below is a numerical scalar of shape `()`.
_1k_ARRAY_COLUMNS = [
    'time',
    'volume_pad_last',
    'flow_pad_zero',
    'flow_volume_pad_last',
]

# Maps known supported column names to `tf.TensorSpec` shapes and dtypes.
_COLUMN_OUTPUT_ARRAY_SIGNATURES: Mapping[str, tf.TensorSpec] = {
    column: tf.TensorSpec(shape=(1000,), dtype=tf.float32)
    for column in _1k_ARRAY_COLUMNS
}
_COLUMN_OUTPUT_SIGNATURES: Mapping[str, tf.TensorSpec] = {
    **_COLUMN_OUTPUT_ARRAY_SIGNATURES
}


def subsample_eids(eids: Iterable[int], percent: int) -> List[int]:
  r"""Subsamples EIDs w/out replacement to a percentage of the original sample.

  Given the same percentage, the subsample will be stable and won't change over
  time. Additionally, a set $S_1$ subsampled to percent $p_1$ will be a subset
  of a set $S_2$ sampled to percent $p_2$ if and only if $p_1 \lte p_2$.

  Args:
    eids: A list of EIDs.
    percent: The percentage of EID samples to keep.

  Returns:
    A subsampled list containing `percent`% of the original `eids` sample.

  Raises:
    ValueError: If `percent` is not in `[0, 100]`.
  """
  if not 0 <= percent <= 100:
    raise ValueError(
        'Subsample percentage must be greater than zero and no more than 100: '
        f'{percent}')

  eids = sorted(list(eids))
  target_size = int(len(eids) * percent / 100)
  rng = np.random.default_rng(42)
  rng.shuffle(eids)
  keep_eids = eids[:target_size]
  keep_eids.sort()
  return keep_eids


def _subsample_df(df: pd.DataFrame, percent: int) -> pd.DataFrame:
  """Returns a `df` copy subsampled to `percent` by EID without replacement."""
  if EID not in df.columns:
    raise ValueError(f'`df` missing `EID` column: {df.columns}')
  subsampled_eids = subsample_eids(set(df[EID]), percent)
  subsampled_df = df.loc[df[EID].isin(subsampled_eids)].copy()
  return subsampled_df


def _apply_feature_scaling(
    df: pd.DataFrame,
    feature_scaling_df: pd.DataFrame,
    inputs: Set[str],
) -> pd.DataFrame:
  """Centers and scales `inputs` by removing the mean and dividing by stddev."""
  columns_to_scale = inputs & set(feature_scaling_df.index)
  for column in columns_to_scale:
    mean = feature_scaling_df.loc[column]['mean']
    std = feature_scaling_df.loc[column]['std']
    df.loc[:, column] -= mean
    df.loc[:, column] /= std
  return df


def _dataframe_to_dataset(df: pd.DataFrame) -> tf.data.Dataset:
  """Converts a `pd.DataFrame` into a `tf.data.Dataset`.

  Each dataset element matches the output signatures defined in the
  `_COLUMN_OUTPUT_SIGNATURES` map for the corresponding columns in the
  DataFrame. If a dataset column is not present in `_COLUMN_OUTPUT_SIGNATURES`,
  we assume that the column denotes a scalar value of shape `()`.

  Args:
    df: The dataframe to convert.

  Returns:
    A dataset containing records from the dataframe.
  """
  output_signature = {}
  for column in df.columns:
    if column in _COLUMN_OUTPUT_SIGNATURES:
      output_signature[column] = _COLUMN_OUTPUT_SIGNATURES[column]
    else:
      output_signature[column] = tf.TensorSpec(shape=(), dtype=tf.float32)

  # Note: `from_generator` is used rather than `from_tensor_slices` since our
  # pd.DataFrame contains np.ndarray elements.
  def _generator_fn():
    for element in df.to_dict('records'):
      yield element

  ds = tf.data.Dataset.from_generator(
      _generator_fn, output_signature=output_signature)

  return ds


def _get_group_tensors_fn(
    inputs: Set[str], labels: Set[str]
) -> Callable[[Dict[str, tf.Tensor]], Tuple[Dict[str, tf.Tensor], Dict[
    str, tf.Tensor], Dict[str, tf.Tensor]]]:
  """Returns a function that generates input, label, and weight dictionaries.

  Note: Sample weights are 0 for NaN (i.e., missing) labels and 1 otherwise.

  Args:
    inputs: A set of input keys; used to group input tensors.
    labels: A set of label keys; used to group label tensors.

  Returns:
    A function that breaks a single `Dict[str, tf.Tensor]` into input, label,
    and sample weight `Dict[str, tf.Tensor]`s.

  Raises:
    ValueError: If inputs or labels are empty.
    ValueError: If inputs and labels are not disjoint.
  """
  if not inputs:
    raise ValueError(f'"inputs" must not be empty: {inputs}')
  if not labels:
    raise ValueError(f'"labels" must not be empty: {labels}')

  def _group_tensors_fn(
      tensor_dict: Dict[str, tf.Tensor]
  ) -> Tuple[Dict[str, tf.Tensor], Dict[str, tf.Tensor], Dict[str, tf.Tensor]]:
    inputs_dict = {}
    labels_dict = {}
    weights_dict = {}
    for tensor_name, tensor in tensor_dict.items():
      if tensor_name in inputs:
        inputs_dict[tensor_name] = tensor
      if tensor_name in labels:
        # Note: We generate non-NaN 0/1 weights for all labels where 0 denotes a
        # NaN value. A sample weight of 0 means that these records are not
        # considered when computing losses or metrics.
        not_nan_mask = tf.math.logical_not(tf.math.is_nan(tensor))
        # Note: If using an array column as a label, we must reduce the mask
        # from `(batch_size, 1000)` to `(batch_size, 1)` since the sample weight
        # corresponds to the entire tensor, not an individual array point.
        if tensor_name in _1k_ARRAY_COLUMNS:
          not_nan_mask = tf.reduce_any(not_nan_mask, -1)
        # Note: We convert masked NaN values to zero so that the weighted
        # `0*value` computation is non-NaN. Since the sample weight for these
        # records is always 0, this value could be any constant that does not
        # result in an error.
        masked_tensor = tf.where(not_nan_mask, tensor, tf.zeros_like(tensor))
        labels_dict[tensor_name] = masked_tensor
        weights_dict[tensor_name] = tf.cast(not_nan_mask, tf.float32)
    return inputs_dict, labels_dict, weights_dict

  return _group_tensors_fn


def _process_dataset(
    ds: tf.data.Dataset,
    inputs: Set[str],
    labels: Set[str],
    batch_size: int,
    cache: bool,
    shuffle: bool,
) -> tf.data.Dataset:
  """Converts a flat dataset into inputs, labels, and weights for model use.

  Args:
    ds: A flat TensorFlow dataset containing Dict[str, tf.Tensor] elements.
    inputs: A set of input keys; used to group input tensors.
    labels: A set of label keys; used to group label tensors.
    batch_size: The processed dataset's batch size.
    cache: Whether to cache the dataset in memory, i.e., `ds.cache()`.
    shuffle: Whether to shuffle the dataset.

  Returns:
    A processed dataset containing elements with spec `Tuple[
      inputs: Dict[str, tf.Tensor], labels: Dict[str, tf.Tensor],
      weights: Dict[str, tf.Tensor]]`.
  """
  group_tensor_fn = _get_group_tensors_fn(inputs, labels)
  ds = ds.map(group_tensor_fn, num_parallel_calls=tf.data.experimental.AUTOTUNE)
  if cache:
    ds = ds.cache()
  if shuffle:
    # Note: This dataset fills a buffer with `buffer_size` elements, then
    # randomly samples elements from this buffer, replacing the selected
    # elements with new elements. For perfect shuffling, a buffer size greater
    # than or equal to the full size of the dataset is required.
    ds = ds.shuffle(buffer_size=10000)
  ds = ds.batch(batch_size)
  ds = ds.prefetch(tf.data.experimental.AUTOTUNE)
  return ds


def build_dataframe(
    data_type: dataset_util.DataType,
    fold: dataset_util.Fold,
    inputs: Set[str],
    labels: Set[str],
    data_dir: pathlib.Path,
    version_suffix: str,
    subsample_percent: Optional[int] = None,
    use_feature_scaling: bool = False,
) -> pd.DataFrame:
  """Builds the dataframe for the given dataset `data_type` and `fold`.

  Args:
    data_type: The dataset `DataType` to load.
    fold: The dataset `Fold`.
    inputs: The set of columns corresponding to model inputs.
    labels: The set of columns corresponding to model prediction targets.
    data_dir: The base directory containing pickled dataframes.
    version_suffix: The suffix used to build the dataframe's basename.
    subsample_percent: If specified, subsamples the dataframe to the given
      percentage without replacement.
    use_feature_scaling: Whether to center and rescale supported `inputs`.

  Returns:
    The dataframe associated with the given `data_type` and `fold`.
  """
  dataset_fold_paths = dataset_util.get_dataset_paths(data_dir, version_suffix)
  filepath = dataset_fold_paths[fold][data_type]
  df = dataset_util.read_pickled_df(filepath)
  df = df[list(inputs.union(labels))]

  if subsample_percent is not None:
    df = _subsample_df(df, subsample_percent)

  if use_feature_scaling:
    feature_scaling_paths = dataset_util.get_feature_scaling_csv_paths(
        data_dir,
        version_suffix,
    )
    feature_scaling_df = dataset_util.read_df(
        feature_scaling_paths[fold],
        index_col='column',
    )
    df = _apply_feature_scaling(df, feature_scaling_df, inputs)

  return df


def build_dataset(
    data_type: dataset_util.DataType,
    fold: dataset_util.Fold,
    inputs: Set[str],
    labels: Set[str],
    batch_size: int,
    cache: bool,
    data_dir: pathlib.Path,
    version_suffix: str,
    subsample_percent: Optional[int] = None,
    use_feature_scaling: bool = False,
) -> tf.data.Dataset:
  """Builds the TensorFlow dataset for the given dataset `data_type` and `fold`.

  The dataset is shuffled when loading `DataType.TRAINING`.

  Args:
    data_type: The dataset `DataType` to load.
    fold: The dataset `Fold`.
    inputs: The set of columns corresponding to model inputs.
    labels: The set of columns corresponding to model prediction targets.
    batch_size: The batch size.
    cache: Whether to cache the dataset in memory, i.e., `ds.cache()`.
    data_dir: The base directory containing pickled dataframes.
    version_suffix: The suffix used to build the dataframe's basename.
    subsample_percent: If specified, subsamples the dataset to the given
      percentage without replacement.
    use_feature_scaling: Whether to center and rescale supported `inputs`.

  Returns:
    The dataset associated with the given data_type and fold.
  """
  df = build_dataframe(
      data_type,
      fold,
      inputs,
      labels,
      data_dir,
      version_suffix,
      subsample_percent,
      use_feature_scaling,
  )
  ds = _dataframe_to_dataset(df)
  return _process_dataset(
      ds,
      inputs,
      labels,
      batch_size,
      cache=cache,
      shuffle=data_type is dataset_util.DataType.TRAINING,
  )


def config_to_dataset(
    dataset_config: ml_collections.ConfigDict,
    data_type: dataset_util.DataType,
    fold_override: Optional[dataset_util.Fold] = None,
    cache_override: Optional[bool] = None,
) -> tf.data.Dataset:
  """Converts a `dataset_config` `ml_collections.ConfigDict` to a dataset.

  The `dataset_config` is expected to match the schema defined in the README.

  Args:
    dataset_config: A dataset `ConfigDict` parameratizing the dataset.
    data_type: The dataset `DataType` to load.
    fold_override: An optional dataset `Fold`; if specified, this value
      overrides the fold specified in `dataset_config`. This override is useful
      for training the model on the config's fold and evaluating on `Fold.ALL`.
    cache_override: An optional boolean denoting whether to cache the dataset in
      memory, i.e., `ds.cache()`; if specified, this value overrides the fold
      specified in `dataset_config`.

  Returns:
    The dataset for the given split.
  """
  config_fold_str = dataset_config['fold']
  config_fold = dataset_util.str_to_fold(config_fold_str)
  config_cache = dataset_config['cache']
  fold = fold_override if fold_override else config_fold
  cache = cache_override if cache_override else config_cache

  # Note: 'subsample_train_percent' only applies to the `DataType.TRAINING`
  # dataset, so this value is set to `None` if building a `DataType.VALIDATION`
  # or `DataType.TEST` dataset.
  config_percent = dataset_config.get('subsample_train_percent', None)
  if data_type == dataset_util.DataType.TRAINING:
    subsample_percent = config_percent
  else:
    subsample_percent = None

  return build_dataset(
      data_type,
      fold,
      dataset_config['inputs'],
      dataset_config['labels'],
      dataset_config['batch_size'],
      cache,
      pathlib.Path(dataset_config['data_dir']),
      dataset_config['version_suffix'],
      subsample_percent,
      dataset_config.get('use_feature_scaling', False),
  )
