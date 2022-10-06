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
"""Lightweight binary for mean-ensembling individual model predictions."""
import pathlib
from typing import List, Sequence

from absl import app
from absl import flags
import pandas as pd

import dataset_util

FLAGS = flags.FLAGS

flags.DEFINE_list(
    'member_predictions', None,
    'A list of paths to prediction TSVs from each ensemble member.')
flags.DEFINE_string(
    'output_filepath', None,
    'The filepath at which to write ensembled model predictions.')
flags.DEFINE_string(
    'id_column',
    'eid',
    'The name of the unique id column in prediction TSVs.',
)
flags.DEFINE_string(
    'pred_column',
    'copd',
    'The name of predicted phenotype column in prediction TSVs.',
)
flags.mark_flags_as_required([
    'member_predictions',
    'output_filepath',
])


def mean_ensemble_preds(
    preds: List[pd.DataFrame],
    id_column: str,
) -> pd.DataFrame:
  """Mean-ensembles all columns in `preds` on `id_column`."""
  pred_ids = set(preds[0][id_column])
  for pred_df in preds:
    if set(pred_df[id_column]) != pred_ids:
      raise ValueError('ID mismatch in prediction dataframes.')
  return pd.concat(preds).groupby(id_column, as_index=False).mean()


def main(argv: Sequence[str]) -> None:
  if len(argv) > 1:
    raise app.UsageError('Too many command-line arguments.')

  member_prediction_dfs = []
  for path in FLAGS.member_predictions:
    member_df = dataset_util.read_df(pathlib.Path(path), sep='\t')
    member_df = member_df[[FLAGS.id_column, FLAGS.pred_column]].copy()
    member_prediction_dfs.append(member_df)

  ensembled_df = mean_ensemble_preds(member_prediction_dfs, FLAGS.id_column)

  with open(FLAGS.output_filepath, mode='w') as f:
    ensembled_df.to_csv(f, sep='\t', index=None)


if __name__ == '__main__':
  app.run(main)
