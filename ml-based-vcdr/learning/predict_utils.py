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
r"""Utilities for generating model predictions."""
import collections
from typing import Dict, List

import numpy as np
import pandas as pd
import tensorflow as tf

# The ID key for each input TensorDict.
ID_KEY = 'id'

# Maps each outcome's target indices to a column label. For example,
# `OUTCOME_COLUMN_MAP['glaucoma_gradability'][1]` corresponds to the second
# multi-class target of the model's 'glaucoma_gradability' head.
OUTCOME_COLUMN_MAP = {
    'id': {
        0: 'image_id'
    },
    'glaucoma_gradability': {
        0: 'glaucoma_gradability:UNGRADABLE',
        1: 'glaucoma_gradability:WITH_DIFFICULTY',
        2: 'glaucoma_gradability:GRADABLE',
    },
    'glaucoma_suspect_risk': {
        0: 'glaucoma_suspect_risk:NON_GLAUCOMATOUS',
        1: 'glaucoma_suspect_risk:LOW_RISK',
        2: 'glaucoma_suspect_risk:HIGH_RISK',
        3: 'glaucoma_suspect_risk:LIKELY',
    },
    'vertical_cd_visibility': {
        0: 'vertical_cd_visibility:UNABLE_TO_ASSESS',
        1: 'vertical_cd_visibility:COMPROMISED',
        2: 'vertical_cd_visibility:SUFFICIENT',
    },
    'vertical_cup_to_disc': {
        0: 'vertical_cup_to_disc:VERTICAL_CUP_TO_DISC'
    }
}


def generate_predictions(
    model: tf.keras.Model,
    predict_ds: tf.data.Dataset) -> Dict[str, List[np.ndarray]]:
  """Returns a dictionary of batched predictions for each outcome."""

  progbar = tf.keras.utils.Progbar(
      None,
      width=30,
      verbose=1,
      interval=0.05,
      stateful_metrics=None,
      unit_name='step')

  # Build the list of mode outputs.
  output_names = model.output_names.copy()
  output_names.append(ID_KEY)

  # Predict outcomes for all examples and build a dictionary of output arrays.
  batched_predictions = collections.defaultdict(list)
  for (inputs_batch, _, _) in predict_ds:
    predict_batch = model.predict_on_batch(inputs_batch)
    predict_batch.append(inputs_batch[ID_KEY].numpy())
    for output_name, ndarray in zip(output_names, predict_batch):
      batched_predictions[output_name].append(ndarray)
    progbar.add(1)
  print()

  return batched_predictions


def merge_batched_predictions(
    batched_predictions: Dict[str, List[np.ndarray]]) -> pd.DataFrame:
  """Creates a DataFrame containing all outcome predictions for each `ID_KEY`.

  Args:
    batched_predictions: A dictionary keyed on outcome name containing a list of
      outcome predictions for each batch.

  Returns:
    A DataFrame containing one column for each outcome's classification or
    regression targets, i.e., each outcome of shape `(None, N)` will have `N`
    associated columns in the DataFrame.

  Raises:
    ValueError: If `ID_KEY` is not present in `batched_predictions`.
  """
  if ID_KEY not in batched_predictions:
    raise ValueError(f'Batched predictions must contain the ID key: {ID_KEY}.')

  # Build a DataFrame for each prediction output. For each outcome `out` of
  # shape `(None, N)`, we create a column for each of the N classes using the
  # mapping in `OUTCOME_COLUMN_MAP`.
  outcome_dfs = {}
  for outcome_name, ndarray_list in batched_predictions.items():
    concatenated = np.concatenate(ndarray_list)
    outcome_columns = [
        OUTCOME_COLUMN_MAP[outcome_name][i]
        for i in range(concatenated.shape[-1])
    ]
    df = pd.DataFrame(concatenated, columns=outcome_columns)
    outcome_dfs[outcome_name] = df

  # Merge each of the outcome DataFrames into a single DataFrame.
  merged_df = outcome_dfs[ID_KEY]
  for outcome_name, outcome_df in sorted(outcome_dfs.items()):
    if outcome_name == ID_KEY:
      continue
    merged_df = pd.merge(
        merged_df, outcome_df, left_index=True, right_index=True)

  return merged_df
