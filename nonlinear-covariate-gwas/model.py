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
"""Library for defining DeepNull models and parameters.

To support additional model types, a wrapper class must be defined in this
module that adheres to an informal API with two methods, `fit` and `predict`,
with the following signatures:

  def fit(self, train_df: pd.DataFrame, eval_df: pd.DataFrame, verbosity: int)
    -> Dict[str, float]:
    Fits the model using `train_df` and evaluates performance with `eval_df`.

    Args:
      train_df: pd.DataFrame of the training data. All covariates and target
        phenotype specified in the model constructor must be present in the
        train_df.
      eval_df: pd.DataFrame of the evaluation data. All covariates and target
        phenotype specified in the model constructor must be present in the
        eval_df.
      verbosity: Level of verbosity to use when fitting the model.

    Returns:
      A dict of performance metrics of the trained model applied to `eval_df`.

  def predict(self, df: pd.DataFrame, prediction_column: str) -> pd.DataFrame:
    Returns a DataFrame of predictions applied to the input data.

    Args:
      df: pd.DataFrame of the data on which to predict. This dataframe must
        contain all of the covariates specified in the model constructor.
      prediction_column: The name of the output column in which to produce
        individual-level predictions.

    Returns:
      A pd.DataFrame with two columns: "IID" which is the unique individual
      identifier, and `prediction_column` which contains the individual-level
      prediction of the target attribute.

This API is not enforced as a strict abstract base class of each model solely to
simplify the calls to underlying libraries that may use the same method names
with different signatures.
"""
import abc
import math
import os
from typing import Dict, List, Optional
from deepnull import config
from deepnull import data
from deepnull import metrics
import ml_collections
import pandas as pd
import tensorflow as tf
import xgboost


########################### TensorFlow models ##################################
def _make_input_fn(features_df: pd.DataFrame,
                   labels: Optional[pd.Series],
                   num_epochs: Optional[int],
                   is_training: bool,
                   batch_size: int,
                   shuffle_buffer_size: int = 10000) -> tf.data.Dataset:
  """Converts input data into a tf.data.Dataset.

  This function is useful for all TensorFlow-based models, which operate on
  tf.data.Dataset objects for training, evaluation, and prediction.

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


class _DeepNull(tf.keras.models.Model, abc.ABC):
  """ABC for DeepNull model with MLP layers and direct linear connection.

  Attributes:
    self.mlp: The multi-layer perceptron path through the model.
    self.linear: The linear path through the model.
  """

  def __init__(self,
               target: str,
               covariates: List[str],
               full_config: ml_collections.ConfigDict,
               fold_ix: int,
               logdir: str = '/tmp',
               **kwargs):
    """Initialize.

    Args:
      target: Name of the target feature to predict.
      covariates: List of covariate features used to predict `target`.
      full_config: The model, training, and optimization parameters to use.
      fold_ix: A unique integer representing the fold of data the model is
        trained on.
      logdir: The directory under which logs and checkpoints are written.
      **kwargs: Other arguments for tf.keras.models.Model.
    """
    super().__init__(**kwargs)
    self.target = target
    self.covariates = covariates
    assert full_config.model_type == config.DEEPNULL
    self.model_config = full_config.model_config
    self.optimizer_config = full_config.optimizer_config
    self.training_config = full_config.training_config

    # Create the model.
    feature_columns = [
        tf.feature_column.numeric_column(covariate_name, dtype=tf.float32)
        for covariate_name in self.covariates
    ]
    dense_feature_layer = [tf.keras.layers.DenseFeatures(feature_columns)]
    # Non-linear path (long path) in DeepNull.
    mlp_layers = dense_feature_layer + [
        tf.keras.layers.Dense(
            unit, activation=self.model_config.mlp_activation, name=f'layer{i}')
        for i, unit in enumerate(self.model_config.mlp_units)
    ] + [tf.keras.layers.Dense(1, activation=None, name='linear_mlp')]

    # Linear path (short, ResNet-esque path) in DeepNull.
    linear_layers = dense_feature_layer + [
        tf.keras.layers.Dense(1, activation=None, name='linear')
    ]
    self.mlp = tf.keras.Sequential(mlp_layers)
    self.linear = tf.keras.Sequential(linear_layers)

    if self.optimizer_config.optimization_metric:
      opt_name = self.optimizer_config.optimization_metric
    else:
      opt_name = self.default_optimization_metric
    self.optimization_metric = metrics.get_optimization_metric(opt_name)

    learning_rate = (self.optimizer_config.learning_rate_batch_1024 *
                     self.training_config.batch_size) / 1024.

    self.compile(
        loss=self.loss_function(),
        optimizer=tf.keras.optimizers.Adam(
            learning_rate=learning_rate,
            beta_1=self.optimizer_config.beta_1,
            beta_2=self.optimizer_config.beta_2,
        ),
        metrics=self.metrics_to_use())
    self.best_checkpoint_path = os.path.join(logdir, 'ckpts',
                                             f'best-{fold_ix}.ckpt')
    self.callbacks = [
        tf.keras.callbacks.ModelCheckpoint(
            filepath=self.best_checkpoint_path,
            save_weights_only=True,
            monitor=self.optimization_metric.best_checkpoint_metric,
            mode=self.optimization_metric.best_checkpoint_mode,
            save_freq='epoch',
            save_best_only=True),
        tf.keras.callbacks.TensorBoard(
            log_dir=os.path.join(logdir, f'tb{fold_ix}')),
    ]

  def model(self, inputs) -> tf.keras.models.Model:
    """Defined to support use of model.summary()."""
    return tf.keras.models.Model(inputs=inputs, outputs=self.call(inputs))

  def call(self, inputs):
    """See https://keras.io/api/models/model/#model-class for details."""
    return self.final_activation(self.mlp(inputs) + self.linear(inputs))

  def fit(self,
          train_df: pd.DataFrame,
          eval_df: pd.DataFrame,
          verbosity: int = 1) -> Dict[str, float]:
    """Fit the model. See module docstring for details."""
    train_input_fn = _make_input_fn(
        features_df=train_df[self.covariates],
        labels=train_df[self.target],
        num_epochs=None,  # Loop indefinitely.
        is_training=True,
        batch_size=self.training_config.batch_size)
    eval_input_fn = _make_input_fn(
        features_df=eval_df[self.covariates],
        labels=eval_df[self.target],
        num_epochs=1,
        is_training=False,
        batch_size=self.training_config.batch_size)

    steps_per_epoch = math.ceil(len(train_df) / self.training_config.batch_size)
    super().fit(
        train_input_fn,
        validation_data=eval_input_fn,
        epochs=self.training_config.num_epochs,
        steps_per_epoch=steps_per_epoch,
        verbose=verbosity,
        callbacks=self.callbacks)
    # Load the best model weights back into the model.
    self.load_weights(self.best_checkpoint_path)
    best_ckpt_validation = self.evaluate(
        eval_input_fn,
        batch_size=self.training_config.batch_size,
        # Evaluate only supports silent or progress bar.
        verbose=min(verbosity, 1),
        return_dict=True)

    return best_ckpt_validation

  def predict(self, df: pd.DataFrame, prediction_column: str) -> pd.DataFrame:
    """Predict on `df`. See module docstring for details."""
    features_df = df[[data.IID] + self.covariates].set_index(data.IID)
    input_fn = _make_input_fn(
        features_df=features_df,
        labels=None,  # No label during prediction.
        num_epochs=1,
        is_training=False,
        batch_size=1000)
    y_preds = list(super().predict(input_fn).ravel())

    return pd.DataFrame(data={
        data.IID: df[data.IID],
        prediction_column: y_preds
    })

  @abc.abstractproperty
  def final_activation(self):
    """Returns the activation to apply to the final model output."""

  @abc.abstractproperty
  def default_optimization_metric(self):
    """Returns the default optimization metric to use for the model."""

  @abc.abstractmethod
  def metrics_to_use(self):
    """Returns the list of metrics to use when training the model."""

  @abc.abstractmethod
  def loss_function(self):
    """The loss function to use to train the model."""


class QuantitativeDeepNull(_DeepNull):
  """Concrete subclass to train quantitative phenotypes."""

  @property
  def final_activation(self):
    # Identity function: There should be no activation applied for a
    # quantitative phenotype.
    return tf.keras.activations.linear

  @property
  def default_optimization_metric(self):
    return 'tf_pearson'

  def metrics_to_use(self):
    return ['mse', metrics.tf_pearson]

  def loss_function(self):
    return tf.keras.losses.MeanSquaredError(name='mse')


class BinaryDeepNull(_DeepNull):
  """Concrete subclass to train binary phenotypes."""

  @property
  def final_activation(self):
    return tf.keras.activations.sigmoid

  @property
  def default_optimization_metric(self):
    return 'crossentropy'

  def metrics_to_use(self):
    return [
        'crossentropy', 'accuracy',
        tf.keras.metrics.AUC(curve='ROC', name='auroc'),
        tf.keras.metrics.AUC(curve='PR', name='auprc')
    ]

  def loss_function(self):
    return tf.keras.losses.BinaryCrossentropy(name='crossentropy')


############################# XGBoost models ###################################
class XGBoostModel(abc.ABC):
  """ABC for XGBoost models."""

  def __init__(self, target: str, covariates: List[str],
               full_config: ml_collections.ConfigDict):
    self.target = target
    self.covariates = covariates
    assert full_config.model_type == config.XGBOOST
    self.model_params = full_config.model_config.to_dict()
    if not self.model_params.get('objective'):
      self.model_params['objective'] = self.default_objective
    opt_metric = self.model_params.get('optimization_metric')
    if not opt_metric:
      opt_metric = self.default_optimization_metric
    self.model_params['eval_metric'] = self.get_eval_metrics(opt_metric)
    self.training_config = full_config.training_config
    self.model = None

  def fit(self,
          train_df: pd.DataFrame,
          eval_df: pd.DataFrame,
          verbosity: int = 1) -> Dict[str, float]:
    """Fit the model. See module docstring for details."""
    train_x = train_df[[data.IID] + self.covariates].set_index(data.IID)
    train_y = train_df[[data.IID, self.target]].set_index(data.IID)
    eval_x = eval_df[[data.IID] + self.covariates].set_index(data.IID)
    eval_y = eval_df[[data.IID, self.target]].set_index(data.IID)

    dtrain = xgboost.DMatrix(train_x, label=train_y)
    deval = xgboost.DMatrix(eval_x, label=eval_y)

    evallist = [(deval, 'eval')]
    num_boost_round = self.training_config.num_boost_round
    evals_result = {}
    self.model = xgboost.train(
        self.model_params,
        dtrain,
        num_boost_round,
        evallist,
        evals_result=evals_result,
        verbose_eval=bool(verbosity))

    retval = {}
    for metric_name, metric_values in evals_result['eval'].items():
      if metric_values:
        retval[metric_name] = float(metric_values[-1])

    # Add Pearson correlation if this is a quantitative phenotype... this is not
    # available as an eval metric so we compute it directly.
    if isinstance(self, QuantitativeXGBoost):
      eval_preds_df = self.predict(eval_df, prediction_column='eval_preds')
      retval['pearson'] = eval_preds_df['eval_preds'].corr(
          eval_df[self.target], method='pearson')

    return retval

  def predict(self, df: pd.DataFrame, prediction_column: str) -> pd.DataFrame:
    """Predict on `df`. See module docstring for details."""
    if self.model is None:
      raise ValueError('Calling predict on an XGBoost model that is not fit.')
    test_x = df[[data.IID] + self.covariates].set_index(data.IID)
    dtest = xgboost.DMatrix(test_x)
    preds = self.model.predict(dtest)
    return pd.DataFrame(data={data.IID: df[data.IID], prediction_column: preds})

  @abc.abstractproperty
  def default_optimization_metric(self):
    """Returns the default optimization metric to use for the model."""

  @abc.abstractproperty
  def default_objective(self):
    """Returns the default objective to use for the model."""

  @abc.abstractmethod
  def get_eval_metrics(self, optimization_metric: str) -> List[str]:
    """Returns the list of metrics to compute.

    Args:
      optimization_metric: The metric used to select the best checkpoint.

    Returns:
      The list of metrics to compute. Per XGBoost convention, the final item in
      the list is used as the optimization metric.
    """


class BinaryXGBoost(XGBoostModel):
  """Concrete subclass to train binary phenotypes."""

  @property
  def default_optimization_metric(self):
    return 'logloss'

  @property
  def default_objective(self):
    return 'binary:logistic'

  def get_eval_metrics(self, optimization_metric: str) -> List[str]:
    # 'aucpr' only supported for XGBoost versions >= 0.9.
    other_metrics = {'auc', 'aucpr', 'error'} - {optimization_metric}
    return sorted(other_metrics) + [optimization_metric]


class QuantitativeXGBoost(XGBoostModel):
  """Concrete subclass to train quantitative phenotypes."""

  @property
  def default_objective(self):
    # linear for XGBoost versions < 0.9, squarederror afterwards.
    return 'reg:squarederror'

  @property
  def default_optimization_metric(self):
    return 'rmse'

  def get_eval_metrics(self, optimization_metric: str) -> List[str]:
    other_metrics = {'rmse', 'mae'} - {optimization_metric}
    return sorted(other_metrics) + [optimization_metric]


def get_model(target: str, target_is_binary: bool, covariates: List[str],
              full_config: ml_collections.ConfigDict, fold_ix: int, logdir: str,
              seed: Optional[int]):
  """Returns the appropriate model for the given data and config.

  This function is the public entry point for accessing the models, used in
  training and evaluation.

  Args:
    target: Name of the target feature to predict.
    target_is_binary: True if and only if `target` is a binary feature.
    covariates: List of covariate features used to predict `target`.
    full_config: The model, training, and optimization parameters to use.
    fold_ix: A unique integer representing the fold of data the model is trained
      on.
    logdir: The directory under which logs and checkpoints are written.
    seed: A random seed used to ensure model determinism.
  """
  del seed  # Unused.
  if full_config.model_type == config.DEEPNULL:
    cls = BinaryDeepNull if target_is_binary else QuantitativeDeepNull
    return cls(
        target=target,
        covariates=covariates,
        full_config=full_config,
        fold_ix=fold_ix,
        logdir=logdir)
  elif full_config.model_type == config.XGBOOST:
    cls = BinaryXGBoost if target_is_binary else QuantitativeXGBoost
    return cls(target=target, covariates=covariates, full_config=full_config)
  else:
    raise ValueError(f'Unsupported model type: {full_config.model_type}')
