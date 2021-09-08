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
"""Library for data loading, analysis, and manipulation."""
import io
import os
import random
from typing import Any, Dict, List, Optional, Tuple, Union
import numpy as np
import pandas as pd

# Individual ID column.
IID = 'IID'


def _missing_to_nan(data: pd.Series,
                    input_missing_value: Union[str, int, float]) -> pd.Series:
  """Replaces `input_missing_value` with np.nan."""
  retval = data.copy(deep=True)
  retval[pd.isnull(data)] = np.nan
  retval[data == input_missing_value] = np.nan
  return retval


def is_binary(ser: pd.Series) -> bool:
  """Returns True if and only if the series is a binary phenotype."""
  unique_values = sorted(ser.dropna().unique())

  return (len(unique_values) == 2 and
          ((unique_values[0] == 0 and unique_values[1] == 1) or
           (unique_values[0] == 1 and unique_values[1] == 2)))


def _get_binary_mapping(binary_series: pd.Series) -> Dict[Any, int]:
  """Returns a dictionary mapping from target values to {0, 1}.

  This function assumes that `is_binary(binary_series)` returns True; its
  behavior is incorrect otherwise.

  Args:
    binary_series: The series to provide the mapping for.

  Returns:
    A two-item dictionary mapping from original data values to data values
    in {0, 1}.
  """
  return {
      val: i for i, val in enumerate(sorted(binary_series.dropna().unique()))
  }


def load_plink_or_bolt_file(
    path_or_buf: Union[str, os.PathLike,
                       io.IOBase], missing_value: Union[str, int, float]
) -> Tuple[pd.DataFrame, Dict[str, Dict[int, Union[int, float]]]]:
  """Returns a DataFrame of the input data and binary field mapping.

  Beyond simply reading the DataFrame, this handles converting NA values to
  np.nan and transforming binary fields to {0, 1}. Note: BOLT and Regenie handle
  the string 'NA' as a missing value by default, which pandas automatically
  converts to np.nan. PLINK defaults to using -9 for missing; in this case any
  true values that equal -9 will be converted to np.nan.

  Binary fields are identified as having two non-NA values and must be in {0, 1}
  or {1, 2}.

  Args:
    path_or_buf: The path to the input file.
    missing_value: The "missing data" value used in the input file.

  Returns:
    A pd.DataFrame of the input data with the above conversions, and a mapping
    from column names to the binary encoding mapping of the raw input data.

  Raises:
    ValueError: The input is not a properly formatted PLINK or BOLT file.
  """
  df = pd.read_csv(path_or_buf, delimiter='\t')
  if list(df.columns[:2]) != ['FID', IID]:
    raise ValueError(
        f'"FID" and "IID" required to start PLINK/BOLT file: {df.columns}')
  if df[IID].isnull().any():
    raise ValueError(f'"{IID}" column of {path_or_buf} contains missing data.')
  if len(df[IID].unique()) != len(df):
    raise ValueError(
        f'"{IID}" column is not unique: '
        f'{len(df[IID].unique())} entries in df of size {len(df)}.')
  invalid_column_chars = {' ', '(', ')', '|'}
  for column in df.columns:
    if set(column) & invalid_column_chars:
      raise ValueError(
          f'Column name has invalid characters {invalid_column_chars}: {column}'
      )

  binary_column_map = {}
  for column in df.columns:
    if column in ['FID', IID]:
      continue
    df[column] = _missing_to_nan(df[column], missing_value)
    if is_binary(df[column]):
      input_to_dn_map = _get_binary_mapping(df[column])
      df[column] = df[column].replace(input_to_dn_map)
      binary_column_map[column] = {v: k for k, v in input_to_dn_map.items()}

  return df, binary_column_map


def write_plink_or_bolt_file(input_df: pd.DataFrame,
                             path_or_buf: Union[Optional[str], os.PathLike,
                                                io.IOBase],
                             binary_column_mapping: Dict[str,
                                                         Dict[int,
                                                              Union[int,
                                                                    float]]],
                             missing_value: Union[str, int, float],
                             cast_ints: bool = True) -> Optional[str]:
  """Writes a PLINK/BOLT formatted file of `input_df` to `path`.

  This is the complementary function to `load_plink_or_bolt_file`. In
  particular, the `binary_column_mapping` input is expected to be created by
  the loading function to ensure that non-DeepNull-predicted columns retain the
  same values as in the input data.

  Args:
    input_df: The DataFrame to write to TSV.
    path_or_buf: The path to write the TSV to.
    binary_column_mapping: The mapping from binary column name to the mapping of
      the binary represenation of that column in `input_df` to the original
      binary representation of the data.
    missing_value: The missing value to use when writing out. Typically 'NA' for
      BOLT or Regenie, and possibly -9 for PLINK.
    cast_ints: If True, any fields that contain only integer values are written
      as integers.

  Returns:
    The result as a string if `path_or_buf` is None, otherwise None.
  """
  # Sanity check.
  if list(input_df.columns[:2]) != ['FID', IID]:
    raise ValueError(f'"FID" and "{IID}" required to start PLINK/BOLT file: '
                     f'{input_df.columns}')

  # Make a copy since we mutate, then transform binary fields to their original
  # representation.
  df = input_df.copy()
  for column, mapping in binary_column_mapping.items():
    df[column] = df[column].replace(mapping)

  if cast_ints:
    for column in df.columns:
      values = df[column]
      mask = ~values.isnull()
      try:
        int_values = values[mask].astype(int)
      except ValueError:
        # This is a non-numeric field, leave it as-is.
        continue
      else:
        if (values[mask] == int_values).all():
          # All non-null values are integers. Convert to the 'Int64' type that
          # allows nullable integers. This requires nulls to use the pd.NA value
          # rather than np.nan.
          df[column] = values.fillna(pd.NA).astype('Int64')

  return df.to_csv(
      path_or_buf, sep='\t', index=False, na_rep=str(missing_value))


def split_data_in_folds(input_df: pd.DataFrame,
                        num_folds: int,
                        seed: Optional[int] = None):
  """Yields (train, eval, test) DataFrames for all `num_folds` data splits."""
  all_ids = sorted(input_df[IID])
  random.seed(seed)
  random.shuffle(all_ids)

  fold_ids = [[] for _ in range(num_folds)]
  for i, id_ in enumerate(all_ids):
    fold_ids[i % num_folds].append(id_)

  all_folds = set(range(num_folds))
  for eval_fold in range(num_folds):
    test_fold = (eval_fold + 1) % num_folds
    train_folds = all_folds - {eval_fold, test_fold}

    train_ids = []
    for train_fold in train_folds:
      train_ids += fold_ids[train_fold]

    eval_ids = fold_ids[eval_fold]
    test_ids = fold_ids[test_fold]
    yield (input_df.loc[input_df[IID].isin(train_ids)],
           input_df.loc[input_df[IID].isin(eval_ids)],
           input_df.loc[input_df[IID].isin(test_ids)])


def parse_covariates(covariate_str: str) -> List[str]:
  """Returns a list of unique covariates specified."""
  return sorted(
      {covar.strip() for covar in covariate_str.split(',') if covar.strip()})
