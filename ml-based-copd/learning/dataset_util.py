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
""""Functions for loading and describing spirometry datasets."""
import enum
import os
import pathlib
from typing import Dict, Optional

import pandas as pd


class DataType(enum.Enum):
  """Represents a data type denoting the train, validation, or test datasets."""
  TRAINING = 'TRAINING'
  VALIDATION = 'VALIDATION'
  TEST = 'TEST'


class Fold(enum.Enum):
  """Represents a dataset fold. 'ALL' denotes all individuals, i.e., no fold."""
  ALL = 'ALL'
  ZERO = 'ZERO'
  ONE = 'ONE'


def _map_dataset_to_filepaths(
    data_dir: pathlib.Path,
    version_suffix: str,
    file_extension: str,
) -> Dict[Fold, Dict[DataType, pathlib.Path]]:
  """Returns a mapping of `Fold`s and `DataType`s to versioned filepaths.

  It's assumed that the `data_dir` contains `file_extension` files matching the
  following schema:
    - '{training,validation,test}.{version_suffix}.{file_extension}'
    - '{training,validation,test}.fold_{0,1}.{version_suffix}.{file_extension}'

  Args:
    data_dir: The base directory containing files of type `file_extension`.
    version_suffix: The suffix used to build the file's basename.
    file_extension: The file extension used to build the file's basename.

  Returns:
    A mapping of `Fold`s and `DataType`s to filepaths.

  Raises:
    ValueError: An expected filepath does not exist.
  """
  common_suffix = f'{version_suffix}.{file_extension}'
  dataset_fold_paths = {
      Fold.ALL: {
          DataType.TRAINING: data_dir / f'training.{common_suffix}',
          DataType.VALIDATION: data_dir / f'validation.{common_suffix}',
          DataType.TEST: data_dir / f'test.{common_suffix}',
      },
      Fold.ZERO: {
          DataType.TRAINING: data_dir / f'training.fold_0.{common_suffix}',
          DataType.VALIDATION: data_dir / f'validation.fold_0.{common_suffix}',
          DataType.TEST: data_dir / f'test.fold_0.{common_suffix}',
      },
      Fold.ONE: {
          DataType.TRAINING: data_dir / f'training.fold_1.{common_suffix}',
          DataType.VALIDATION: data_dir / f'validation.fold_1.{common_suffix}',
          DataType.TEST: data_dir / f'test.fold_1.{common_suffix}',
      },
  }

  return dataset_fold_paths


def get_dataset_paths(
    data_dir: pathlib.Path,
    version_suffix: str,
) -> Dict[Fold, Dict[DataType, pathlib.Path]]:
  """Returns a mapping of `Fold`s and `DataType`s to pickled dataframe paths.

  It's assumed that the `data_dir` contains 'pkl' files matching the following:
    - '{training,validation,test}.{version_suffix}.pkl'
    - '{training,validation,test}.fold_{0,1}.{version_suffix}.pkl'

  Args:
    data_dir: The base directory containing pickled dataframes.
    version_suffix: The suffix used to build the dataframe's basename.

  Returns:
    A mapping of `Fold`s and `DataType`s to pickled dataframe paths.
  """
  return _map_dataset_to_filepaths(data_dir, version_suffix, 'pkl')


def get_eid_csv_paths(
    data_dir: pathlib.Path,
    version_suffix: str,
) -> Dict[Fold, Dict[DataType, pathlib.Path]]:
  """Returns a mapping of `Fold`s and `DataType`s to EID CSV paths.

  It's assumed that the `data_dir` contains 'csv' files matching the following:
    - '{training,validation,test}.{version_suffix}.csv'
    - '{training,validation,test}.fold_{0,1}.{version_suffix}.csv'

  Args:
    data_dir: The base directory containing EID CSVs.
    version_suffix: The suffix used to build the EID CSV's basename.

  Returns:
    A mapping of `Fold`s and `DataType`s to EID CSV paths.
  """
  return _map_dataset_to_filepaths(data_dir, version_suffix, 'csv')


def get_feature_scaling_csv_paths(
    data_dir: pathlib.Path,
    version_suffix: str,
) -> Dict[Fold, pathlib.Path]:
  """Returns a mapping of `Fold`s and `DataType`s to feature scaling CSV paths.

  It's assumed that the `data_dir` contains 'csv' files matching the following:
    - '{feature_scaling}.{version_suffix}.csv'
    - '{feature_scaling}.fold_{0,1}.{version_suffix}.csv'

  The CSVs contain the mean and standard deviation for scalar columns.

  Args:
    data_dir: The base directory containing feature scaling CSVs.
    version_suffix: The suffix used to build the feature scaling CSV's basename.

  Returns:
    A mapping of `Fold`s and `DataType`s to feature scaling CSV paths.
  """
  feature_scaling_paths = {
      Fold.ALL: data_dir / f'feature_scaling.{version_suffix}.csv',
      Fold.ZERO: data_dir / f'feature_scaling.fold_0.{version_suffix}.csv',
      Fold.ONE: data_dir / f'feature_scaling.fold_1.{version_suffix}.csv',
  }

  return feature_scaling_paths


def str_to_data_type(data_type_name: str) -> DataType:
  """Returns the `DataType` enum instance corresponding to `data_type_name`.

  Args:
    data_type_name: The `DataType` attribute name.

  Returns:
    An instance of the enum corresponding to `DataType[data_type_name]`.

  Raises:
    ValueError: The given `data_type_name` does not correspond to an enum value.
  """
  try:
    data_type_attr = getattr(DataType, data_type_name)
  except AttributeError:
    raise ValueError(f'Invalid attr for "DataType": {data_type_name}') from None
  return data_type_attr


def str_to_fold(fold_name: str) -> Fold:
  """Returns the `Fold` enum instance corresponding to `fold_name`.

  Args:
    fold_name: The `Fold` attribute name.

  Returns:
    An instance of the enum corresponding to `Fold[fold_name]`.

  Raises:
    ValueError: The given `fold_name` does not correspond to an enum value.
  """
  try:
    data_type_attr = getattr(Fold, fold_name)
  except AttributeError:
    raise ValueError(f'Invalid attr for "Fold": {fold_name}') from None
  return data_type_attr


def read_pickled_df(path: pathlib.Path,) -> pd.DataFrame:
  """Reads and returns a pickled dataframe."""
  with open(str(path), 'rb') as f:
    df = pd.read_pickle(f)
  return df


def read_df(
    path: pathlib.Path,
    index_col: Optional[str] = None,
    sep: str = ',',
) -> pd.DataFrame:
  """Reads and returns a `sep`-separated dataframe."""
  with open(str(path), 'r') as f:
    df = pd.read_csv(f, sep=sep, index_col=index_col)
  return df
