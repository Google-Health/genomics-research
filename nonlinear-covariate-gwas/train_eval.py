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
import math
import os
from typing import Dict, List, Optional, Tuple
import pandas as pd
import tensorflow as tf

from deepnull import data
from deepnull import metrics
from deepnull import model as model_lib

# Suffix of column indicating test fold from which the DeepNull prediction came.
_DEEPNULL_FOLD_COL_SUFFIX = 'deepnull_eval_fold'


def _make_input_fn(features_df: pd.DataFrame,
                   labels: Optional[pd.Series],
                   num_epochs: Optional[int],
                   is_training: bool,
                   batch_size: int,
                   shuffle_buffer_size: int = 10000) -> tf.data.Dataset:
  """Converts input data into a tf.data.Dataset.

  Args:
    features_df: A pd.DataFrame containing all of (and only) the features.
    labels: pd.Series containing target data. None if generating for `predict`
      mode.
    num_epochs: Number of iterations over the whole input dataset. When
      `num_epochs` is set to None, the input data feeds indefinitely.
    is_training: Indicates if the input data is training data. If so, the data
      is shuffled.
    batch_size: Number of data samples in a single batch.
    shuffle_buffer_size: The buffer size to perform shuffling. Has no effect if
      `shuffle` is False. More details are available in
      https://www.tensorflow.org/api_docs/python/tf/data/Dataset#shuffle.

  Returns:
    tf.data.Dataset that can be feed to tf.Keras.Model as input.

  Raises:
    ValueError: labels is None in training mode.
  """
  if labels is None:
    if is_training:
      raise ValueError('labels must not be None in training mode.')
    ds = tf.data.Dataset.from_tensor_slices(dict(features_df))
  else:
    ds = tf.data.Dataset.from_tensor_slices((dict(features_df), labels))
  if is_training:
    ds = ds.shuffle(shuffle_buffer_size)
  ds = ds.repeat(num_epochs)
  ds = ds.batch(batch_size, drop_remainder=is_training).prefetch(1)
  return ds


def train_deepnull_model(
    train_df: pd.DataFrame,
    eval_df: pd.DataFrame,
    target: str,
    target_is_binary: bool,
    covariates: List[str],
    model_params: Optional[model_lib.ModelParameters] = None,
    logdir: str = '/tmp',
    fold_ix: int = 0,
    verbosity: int = 1,
) -> Tuple[tf.keras.models.Model, List[tf.keras.callbacks.History], Dict[
    str, float]]:
  """Returns the best trained DeepNull model.

  This function handles running DeepNull training over the `train_df` and model
  checkpoint selection based on performance within the `eval_df`. Both
  dataframes are required to have all of the following columns, with no empty
  values:
    * `target` - The target value to predict.
    * `covariates` - The list of covariates used to predict `target`.

  Args:
    train_df: The input data used for model training.
    eval_df: The data used for model evaluation / optimal checkpoint selection.
    target: The target value to predict.
    target_is_binary: If True, models as a binary outcome. Otherwise models a
      quantitative outcome.
    covariates: The list of covariates used to predict the target.
    model_params: Parameters of the model to use.
    logdir: The log directory in which to write checkpoints.
    fold_ix: The fold index of the eval dataset. Only needed when training
      multiple folds of data so that logging parameters and checkpoints do not
      overwrite each other.
    verbosity: Level of verbosity when fitting the model. 0=silent, 1=progress,
      2=print once per epoch.

  Returns:
    A triplet of outputs: The first is a DataFrame with 'IID',
    `prediction_column` columns that represent the DeepNull predictions for all
    individuals in the `test_df`. The second is the history of fitting the model
    to the train and eval data. The third is the metrics on the validation set
    of the checkpoint used.
  """
  if model_params is None:
    model_params = model_lib.ModelParameters()

  train_input_fn = _make_input_fn(
      features_df=train_df[covariates],
      labels=train_df[target],
      num_epochs=None,  # Loop indefinitely.
      is_training=True,
      batch_size=model_params.batch_size)
  eval_input_fn = _make_input_fn(
      features_df=eval_df[covariates],
      labels=eval_df[target],
      num_epochs=1,
      is_training=False,
      batch_size=model_params.batch_size)
  feature_columns = [
      tf.feature_column.numeric_column(covariate_name, dtype=tf.float32)
      for covariate_name in covariates
  ]

  if target_is_binary:
    model_cls = model_lib.BinaryDeepNull
    optimization_metric = metrics.get_optimization_metric('crossentropy')
  else:
    model_cls = model_lib.QuantitativeDeepNull
    optimization_metric = metrics.get_optimization_metric('tf_pearson')
  deepnull_model = model_cls(
      feature_columns=feature_columns,
      mlp_units=model_params.mlp_units,
      mlp_activation=model_params.mlp_activation,
      optimization_metric=optimization_metric)

  deepnull_model.compile(
      loss=deepnull_model.loss_function(),
      optimizer=tf.keras.optimizers.Adam(
          learning_rate=model_params.learning_rate,
          beta_1=model_params.beta_1,
          beta_2=model_params.beta_2,
      ),
      metrics=deepnull_model.metrics_to_use())

  best_checkpoint_path = os.path.join(logdir, 'ckpts', f'best-{fold_ix}.ckpt')
  callbacks = [
      tf.keras.callbacks.ModelCheckpoint(
          filepath=best_checkpoint_path,
          save_weights_only=True,
          monitor=deepnull_model.best_checkpoint_metric(),
          mode=deepnull_model.best_checkpoint_mode(),
          save_freq='epoch',
          save_best_only=True),
      tf.keras.callbacks.TensorBoard(
          log_dir=os.path.join(logdir, f'tb{fold_ix}')),
  ]

  steps_per_epoch = math.ceil(len(train_df) / model_params.batch_size)
  history = deepnull_model.fit(
      train_input_fn,
      validation_data=eval_input_fn,
      epochs=model_params.num_epochs,
      steps_per_epoch=steps_per_epoch,
      verbose=verbosity,
      callbacks=callbacks)
  # Load the best model weights back into the model.
  deepnull_model.load_weights(best_checkpoint_path)
  best_ckpt_validation = deepnull_model.evaluate(
      eval_input_fn,
      batch_size=model_params.batch_size,
      # Evaluate only supports silent or progress bar.
      verbose=min(verbosity, 1),
      return_dict=True)
  return deepnull_model, history, best_ckpt_validation


def predict(deepnull_model: tf.keras.models.Model, df: pd.DataFrame,
            covariates: List[str], prediction_column: str) -> pd.DataFrame:
  """Returns a DataFrame of predictions for the given input data.

  Args:
    deepnull_model: The DeepNull model to use for prediction.
    df: The data for which to run prediction.
    covariates: The list of features to use for prediction.
    prediction_column: The output column name to use for the predictions.

  Returns:
    The DataFrame containing a prediction for all individuals in `data`.
  """
  features_df = df[['IID'] + covariates].set_index('IID')
  input_fn = _make_input_fn(
      features_df=features_df,
      labels=None,  # No label during prediction.
      num_epochs=1,
      is_training=False,
      batch_size=1000)
  y_preds = list(deepnull_model.predict(input_fn).ravel())

  return pd.DataFrame(data={'IID': df.IID, prediction_column: y_preds})


def create_deepnull_prediction(
    input_df: pd.DataFrame,
    target: str,
    target_is_binary: bool,
    covariates: List[str],
    prediction_column: Optional[str] = None,
    num_folds: int = 5,
    model_params: Optional[model_lib.ModelParameters] = None,
    seed: int = 318753108,  # From /dev/urandom.
    logdir: str = '/tmp',
    verbosity: int = 2,
) -> Tuple[pd.DataFrame, List[tf.keras.callbacks.History], List[Dict[
    str, float]], pd.DataFrame]:
  """Runs the entire DeepNull algorithm to add the prediction to the input data.

  This is the main entrypoint for training DeepNull. It handles splitting the
  input DataFrame into folds, training a separate DeepNull model on each fold,
  and concatenating all results into a final DataFrame that contains the
  predictions.

  Args:
    input_df: A dataframe representing the input data, as loaded by
      `load_plink_or_bolt_file` above.
    target:  The target value to predict using DeepNull.
    target_is_binary: True if and only if the target should be predicted as a
      binary value.
    covariates: The set of covariate values used to predict the target.
    prediction_column: The name of the output column to add to the dataframe.
    num_folds: The number of folds to split the data into. `num_folds` - 2 folds
      of data are used to train each DeepNull model, with evaluation of the best
      model occurring in one fold, and final predictions occurring in the final
      fold.
    model_params: Model parameters to use in training.
    seed: The random seed used to split data into training folds.
    logdir: The directory in which to write intermediate data.
    verbosity: Level of verbosity when fitting each constituent DeepNull model.
      0=silent, 1=progress, 2=print once per epoch.

  Returns:
    A tuple of four items:
      1. A pd.DataFrame with all input data plus a single new column,
         `prediction_column`, that includes the DeepNull prediction of the
         phenotype in rows for which it could be computed.
      2. A list containing `num_folds` histories, each holding the history of
         training the DeepNull model associated with that fold of data.
      3. A list containing `num_folds` dictionaries, each holding validation
         data performance metrics for each of the validation folds of data.
      4. A pd.DataFrame containing the true target value, DeepNull prediction,
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

  if model_params is None:
    # Use the default settings for all parameters.
    model_params = model_lib.ModelParameters()
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
  histories = []
  validation_data_performance = []
  for fold, (train_df, eval_df, test_df) in enumerate(
      data.split_data_in_folds(trainable_df, num_folds=num_folds, seed=seed)):
    print(f'Beginning training for fold {fold} of {num_folds}...')
    best_model, history, val_perf = train_deepnull_model(
        train_df=train_df,
        eval_df=eval_df,
        target=target,
        target_is_binary=target_is_binary,
        covariates=covariates,
        model_params=model_params,
        logdir=logdir,
        fold_ix=fold,
        verbosity=verbosity,
    )
    fold_predictions = predict(
        deepnull_model=best_model,
        df=test_df,
        covariates=covariates,
        prediction_column=prediction_column)
    fold_predictions[deepnull_fold_col] = fold
    all_predictions.append(fold_predictions)
    histories.append(history)
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
  return final_df, histories, validation_data_performance, test_performance_df
