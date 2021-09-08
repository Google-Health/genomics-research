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
"""Library for performing model training and evaluation."""
import os
from typing import Dict, List, Optional, Tuple
from deepnull import data
from deepnull import model as model_lib
import ml_collections
import pandas as pd

# Suffix of column indicating test fold from which the DeepNull prediction came.
_DEEPNULL_FOLD_COL_SUFFIX = 'deepnull_eval_fold'


def create_deepnull_prediction(
    input_df: pd.DataFrame,
    target: str,
    target_is_binary: bool,
    covariates: List[str],
    full_config: ml_collections.ConfigDict,
    prediction_column: Optional[str] = None,
    num_folds: int = 5,
    seed: int = 318753108,  # From /dev/urandom.
    logdir: str = '/tmp',
    verbosity: int = 2,
) -> Tuple[pd.DataFrame, List[Dict[str, float]], pd.DataFrame]:
  """Runs the entire DeepNull algorithm to add the prediction to the input data.

  This is the main entrypoint for training DeepNull. It handles splitting the
  input DataFrame into folds, training a separate DeepNull model on each fold,
  and concatenating all results into a final DataFrame that contains the
  predictions.

  Args:
    input_df: A dataframe representing the input data, as loaded by
      `data.load_plink_or_bolt_file`.
    target:  The target value to predict using DeepNull.
    target_is_binary: True if and only if the target should be predicted as a
      binary value.
    covariates: The set of covariate values used to predict the target.
    full_config: The model, training, and optimization parameters to use.
    prediction_column: The name of the output column to add to the dataframe.
    num_folds: The number of folds to split the data into. `num_folds` - 2 folds
      of data are used to train each DeepNull model, with evaluation of the best
      model occurring in one fold, and final predictions occurring in the final
      fold.
    seed: The random seed used to split data into training folds.
    logdir: The directory in which to write intermediate data.
    verbosity: Level of verbosity when fitting each constituent DeepNull model.
      0=silent, 1=progress, 2=print once per epoch.

  Returns:
    A tuple of three items:
      1. A pd.DataFrame with all input data plus a single new column,
         `prediction_column`, that includes the DeepNull prediction of the
         phenotype in rows for which it could be computed.
      2. A list containing `num_folds` dictionaries, each holding validation
         data performance metrics for each of the validation folds of data.
      3. A pd.DataFrame containing the true target value, DeepNull prediction,
         and data fold from which the test prediction was made.

  Raises:
    ValueError: The input dataframe is not able to be used to run DeepNull.
  """
  if prediction_column is None:
    prediction_column = f'{target}_deepnull'
  deepnull_fold_col = f'{target}_{_DEEPNULL_FOLD_COL_SUFFIX}'

  if prediction_column in input_df.columns:
    raise ValueError(
        f'Prediction column "{prediction_column}" present in input.')
  if deepnull_fold_col in input_df.columns:
    raise ValueError(f'Reserved column "{deepnull_fold_col}" present in input.')

  if not os.path.exists(logdir):
    os.makedirs(logdir)

  # Sanity checks. Note that we assume the inputs have already been checked for
  # presence and uniqueness of IID, and missing data are represented as np.nan.
  if num_folds < 3:
    raise ValueError(f'Must specify at least 3 data folds: {num_folds}')
  if target not in input_df.columns:
    raise ValueError(f'Target {target} absent from df: {input_df.columns}')
  if any(cov not in set(input_df.columns) for cov in covariates):
    raise ValueError(f'One or more requested covariates ({covariates}) absent '
                     f'from df: {input_df.columns}')
  if target in covariates:
    raise ValueError(
        f'Target value {target} cannot be present in covariates: {covariates}')
  if len(covariates) != len(set(covariates)):
    raise ValueError(f'Duplicate covariates encountered: {covariates}')
  if prediction_column in input_df.columns:
    raise ValueError(f'Output column {prediction_column} already exists.')

  # Identify IIDs where we cannot predict because missingness exists in our
  # predictors or target.
  fields = ['IID', target] + covariates
  missing_mask = pd.isnull(input_df[fields]).any(axis=1)
  iids_with_missing = input_df.IID[missing_mask]

  trainable_df = input_df.loc[~input_df.IID.isin(iids_with_missing), fields]
  all_predictions = []
  validation_data_performance = []
  for fold, (train_df, eval_df, test_df) in enumerate(
      data.split_data_in_folds(trainable_df, num_folds=num_folds, seed=seed)):
    print(f'Beginning training for fold {fold} of {num_folds}...')
    fold_model = model_lib.get_model(
        target=target,
        target_is_binary=target_is_binary,
        covariates=covariates,
        full_config=full_config,
        fold_ix=fold,
        logdir=logdir,
        seed=seed)
    val_perf = fold_model.fit(
        train_df=train_df, eval_df=eval_df, verbosity=verbosity)
    fold_predictions = fold_model.predict(
        df=test_df, prediction_column=prediction_column)
    fold_predictions[deepnull_fold_col] = fold
    all_predictions.append(fold_predictions)
    validation_data_performance.append(val_perf)

  preds_df = pd.concat(all_predictions, ignore_index=True)
  assert len(preds_df) == len(set(preds_df.IID))
  assert set(preds_df.IID) == set(trainable_df.IID)
  # Extract a dataframe that contains the true target value, the DeepNull
  # prediction, and the data fold from which the prediction came, for downstream
  # analysis.
  test_performance_df = pd.merge(
      preds_df, input_df[['IID', target]], on='IID', how='inner')
  # Return as the output to write a dataframe that contains all the rows of the
  # input DataFrame with just the prediction column added for samples where it
  # could be computed.
  final_df = pd.merge(
      input_df, preds_df[['IID', prediction_column]], on='IID', how='left')
  return final_df, validation_data_performance, test_performance_df
