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
"""Tests for deepnull.data."""
import collections
import io
import os
from absl.testing import absltest
from absl.testing import parameterized
import numpy as np
import pandas as pd
from deepnull import data


class DataTest(parameterized.TestCase):

  @parameterized.parameters(
      dict(
          input_data=pd.Series([1, 2, 3, 4]),
          value=-9,
          expected=pd.Series([1, 2, 3, 4])),
      dict(
          input_data=pd.Series([1, 2, 3, np.nan]),
          value=-9,
          expected=pd.Series([1, 2, 3, np.nan])),
      dict(
          input_data=pd.Series([1, -9, 3, np.nan]),
          value=-9,
          expected=pd.Series([1, np.nan, 3, np.nan])),
  )
  def test_missing_to_nan(self, input_data, value, expected):
    """Test that replacing missing data works."""
    actual = data._missing_to_nan(input_data, value)
    pd.testing.assert_series_equal(actual, expected)
    self.assertIsNot(actual, input_data)

  @parameterized.parameters(
      dict(series=pd.Series([0, 1, 2, 1, 3]), expected=False),
      dict(series=pd.Series([0, 0, 0]), expected=False),
      dict(series=pd.Series([
          0,
          0.5,
          0.5,
      ]), expected=False),
      dict(series=pd.Series([
          0,
          1,
          1,
          0,
      ]), expected=True),
      dict(series=pd.Series([2, 1, 1, 2, 2]), expected=True),
      dict(series=pd.Series([0, 2, 2, 0, 0, 2]), expected=False),
      dict(series=pd.Series([0., 1., np.nan, 1., 0., np.nan]), expected=True),
      dict(series=pd.Series([2, 1, np.nan, 1, 2, np.nan, 2]), expected=True),
      dict(series=pd.Series([0, 2, np.nan, 2, 0, np.nan]), expected=False),
  )
  def test_is_binary(self, series, expected):
    actual = data.is_binary(series)
    self.assertEqual(actual, expected)

  @parameterized.parameters(
      dict(series=pd.Series([0, 0, 1, 1, 0]), expected={
          0: 0,
          1: 1
      }),
      dict(series=pd.Series([1, 0, 0, 1, 0]), expected={
          0: 0,
          1: 1
      }),
      dict(series=pd.Series([1, 2, 1, 1, 2]), expected={
          1: 0,
          2: 1
      }),
      dict(series=pd.Series([2, 2, 1, 1, 2]), expected={
          1: 0,
          2: 1
      }),
      dict(series=pd.Series([0., np.nan, 1., 1., 0.]), expected={
          0.: 0,
          1.: 1
      }),
      dict(series=pd.Series([1, np.nan, 0, 1, 0]), expected={
          0.: 0,
          1.: 1
      }),
      dict(series=pd.Series([np.nan, 2., 1., 1., 2.]), expected={
          1.: 0,
          2.: 1
      }),
      dict(series=pd.Series([2, 2, 1, np.nan, 2]), expected={
          1.: 0,
          2.: 1
      }),
  )
  def test_get_binary_mapping(self, series, expected):
    actual = data._get_binary_mapping(series)
    self.assertEqual(actual, expected)

  def test_load_plink_or_bolt_file(self):
    contents = io.StringIO('FID\tIID\tage\tsex\tbinary_miss\tcont_miss\n'
                           '1\t1\t45\t1\t1\t0.5\n'
                           '2\t2\t50\t1\t2\t1.5\n'
                           '3\t3\t55\t2\t2\t2.5\n'
                           '4\t4\t60\t2\tNA\t3.5\n'
                           '5\t5\t65\t1\t1\tNA\n')

    expected_df = pd.DataFrame({
        'FID': [1, 2, 3, 4, 5],
        'IID': [1, 2, 3, 4, 5],
        'age': [45, 50, 55, 60, 65],
        'sex': [0, 0, 1, 1, 0],
        'binary_miss': [0., 1., 1., np.nan, 0.],
        'cont_miss': [0.5, 1.5, 2.5, 3.5, np.nan]
    })
    expected_binary_map = {'sex': {0: 1, 1: 2}, 'binary_miss': {0.: 1., 1.: 2.}}

    actual_df, actual_binary_map = data.load_plink_or_bolt_file(
        path_or_buf=contents, missing_value='NA')

    pd.testing.assert_frame_equal(actual_df, expected_df)
    self.assertEqual(actual_binary_map, expected_binary_map)

  @parameterized.parameters(
      dict(
          str_contents='eid\tage\n1\t50\n2\t55\n',
          msg='"FID" and "IID" required to start PLINK/BOLT'),
      dict(
          str_contents='FID\tIID\tage\n1\t1\t50\n2\tNA\t55\n',
          msg='"IID" column of .* contains missing data'),
      dict(
          str_contents='FID\tIID\tage\n1\t1\t50\n1\t1\t55\n',
          msg='"IID" column is not unique:'),
      dict(
          str_contents='FID\tIID\tage(SE)\n1\t1\t50\n2\t2\t55\n',
          msg='Column name has invalid characters'))
  def test_load_invalid_plink_or_bolt_file(self, str_contents, msg):
    contents = io.StringIO(str_contents)
    with self.assertRaisesRegex(ValueError, msg):
      data.load_plink_or_bolt_file(contents, missing_value='NA')

  def test_write_plink_or_bolt_file(self):
    df = pd.DataFrame(
        {
            'FID': ['fam1', 'fam1', 'fam1', 'fam2', 'fam3'],
            'IID': [1, 2, 3, 4, 5],
            'age': [45, 50, 55, 60, 65],
            'sex': [0, 0, 1, 1, 0],
            'binary_miss': [0., 1., 1., np.nan, 0.],
            'cont_miss': [0.5, 1.5, 2.5, 3.5, np.nan]
        },
        columns=['FID', 'IID', 'age', 'sex', 'binary_miss', 'cont_miss'])
    binary_column_map = {'sex': {0: 1, 1: 2}, 'binary_miss': {0.: 1., 1.: 2.}}

    expected = ('FID\tIID\tage\tsex\tbinary_miss\tcont_miss\n'
                'fam1\t1\t45\t1\t1\t0.5\n'
                'fam1\t2\t50\t1\t2\t1.5\n'
                'fam1\t3\t55\t2\t2\t2.5\n'
                'fam2\t4\t60\t2\tNA\t3.5\n'
                'fam3\t5\t65\t1\t1\tNA\n')

    actual = data.write_plink_or_bolt_file(
        df,
        path_or_buf=None,
        binary_column_mapping=binary_column_map,
        missing_value='NA',
        cast_ints=True)
    self.assertEqual(actual, expected)

  def test_roundtrip_plink_or_bolt_file(self):
    num_entries = 1000
    ids = np.arange(num_entries)
    continuous_int = np.random.choice(np.arange(40, 70), size=num_entries)
    continuous_float = np.random.choice(
        np.arange(30.5, 50.5).astype(float), size=num_entries)
    binary_zero_one = np.random.choice([0, 1], size=num_entries)
    binary_one_two = np.random.choice([1, 2], size=num_entries)
    continuous_two_values = np.random.choice([1, 3], size=num_entries)

    nulled_continuous = np.random.choice(np.arange(0.5, 20.5), size=num_entries)
    nulled_continuous[:10] = np.NaN
    # Note: This input nullable binary must be of type 'Int64' otherwise it will
    # get converted to float values prior to the initial writing to disk.
    nulled_binary = pd.Series(
        np.random.choice([1., 2.], size=num_entries), dtype='Int64')
    nulled_binary[-10:] = pd.NA

    init_df = pd.DataFrame(
        {
            'FID': ids,
            'IID': ids,
            'ci': continuous_int,
            'cf': continuous_float,
            'bzo': binary_zero_one,
            'bot': binary_one_two,
            'ctv': continuous_two_values,
            'nc': nulled_continuous,
            'nb': nulled_binary
        },
        columns=['FID', 'IID', 'ci', 'cf', 'bzo', 'bot', 'ctv', 'nc', 'nb'])

    initial_filename = os.path.join(absltest.get_default_test_tmpdir(),
                                    'init.tsv')
    final_filename = os.path.join(absltest.get_default_test_tmpdir(),
                                  'final.tsv')

    init_df.to_csv(initial_filename, sep='\t', na_rep='NA', index=False)

    deepnull_df, mapping = data.load_plink_or_bolt_file(initial_filename, 'NA')
    data.write_plink_or_bolt_file(deepnull_df, final_filename, mapping, 'NA')

    with open(initial_filename, 'rt') as f:
      initial_contents = f.read()
    with open(final_filename, 'rt') as g:
      final_contents = g.read()
    self.assertEqual(initial_contents, final_contents)

  @parameterized.parameters(
      dict(num_folds=3),
      dict(num_folds=4),
      dict(num_folds=5),
      dict(num_folds=10),
  )
  def test_split_data_in_folds(self, num_folds):
    num_entries = 1000
    ids = np.arange(num_entries)
    train_counts = collections.defaultdict(int)
    eval_counts = collections.defaultdict(int)
    test_counts = collections.defaultdict(int)
    df = pd.DataFrame({
        'FID': ids,
        'IID': ids,
        'values': np.random.random(size=num_entries)
    })

    for train_df, eval_df, test_df in data.split_data_in_folds(
        df, num_folds=num_folds):
      train_ids = set(train_df.IID)
      eval_ids = set(eval_df.IID)
      test_ids = set(test_df.IID)
      self.assertLen(train_ids, len(train_df))
      self.assertLen(eval_ids, len(eval_df))
      self.assertLen(test_ids, len(test_df))
      self.assertLen(df, len(train_ids) + len(eval_ids) + len(test_ids))

      self.assertEmpty(train_ids & eval_ids)
      self.assertEmpty(train_ids & test_ids)
      self.assertEmpty(eval_ids & test_ids)

      for train_id in train_ids:
        train_counts[train_id] += 1
      for eval_id in eval_ids:
        eval_counts[eval_id] += 1
      for test_id in test_ids:
        test_counts[test_id] += 1

    self.assertCountEqual(ids, list(train_counts))
    self.assertCountEqual(ids, list(eval_counts))
    self.assertCountEqual(ids, list(test_counts))
    self.assertEqual(set(train_counts.values()), {num_folds - 2})
    self.assertEqual(set(eval_counts.values()), {1})
    self.assertEqual(set(test_counts.values()), {1})

  @parameterized.parameters(
      dict(covars='', expected=[]),
      dict(covars='age', expected=['age']),
      dict(covars='sex, age', expected=['age', 'sex']),
      dict(
          covars='sex, age, pc1, pc2,, age',
          expected=['age', 'pc1', 'pc2', 'sex']),
  )
  def test_parse_covariates(self, covars, expected):
    actual = data.parse_covariates(covars)
    self.assertEqual(actual, expected)


if __name__ == '__main__':
  absltest.main()
