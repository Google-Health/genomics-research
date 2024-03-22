# Copyright 2024 Google LLC.
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
r"""Generates 12-lead ECG and ECG lead I+PPG datasets from UK Biobank demo data.

Example:
$ python3 generate_dataset_from_ukb_demo.py \
    --out_dir=data \
    --dataset=ecgppg
"""

from typing import Mapping, Sequence
import xml.etree.ElementTree as ET

from absl import app
from absl import flags
import numpy as np
import pandas as pd
from scipy import signal

_INPUT_FILE = flags.DEFINE_string(
    'input_file',
    None,
    'The path of the input raw ecg xml file.',
)

_PPG_INPUT_FILE = flags.DEFINE_string(
    'ppg_input_file',
    None,
    'The path of the input raw ppg file.',
)

_OUT_DIR = flags.DEFINE_string(
    'out_dir',
    None,
    'The path of the output directory in which to write csv files.',
    short_name='o',
)

_DATASET = flags.DEFINE_enum(
    'dataset',
    'ecg12',
    ['ecg12', 'ecgppg'],
    'Which dataset to generate.',
)

_DUPLICATES = flags.DEFINE_integer(
    'duplicates',
    1,
    'The number of duplicates of the example record to generate.',
)

flags.mark_flags_as_required(['out_dir'])


EID = 'eid'
INSTANCE = 'instance'


AT_REST_XML_ATTRIBUTE_STR_TO_LEAD = [
    ("WaveformData_{'lead': 'I'}", "I"),
    ("WaveformData_{'lead': 'II'}", "II"),
    ("WaveformData_{'lead': 'III'}", "III"),
    ("WaveformData_{'lead': 'aVR'}", "aVR"),
    ("WaveformData_{'lead': 'aVL'}", "aVL"),
    ("WaveformData_{'lead': 'aVF'}", "aVF"),
    ("WaveformData_{'lead': 'V1'}", "V1"),
    ("WaveformData_{'lead': 'V2'}", "V2"),
    ("WaveformData_{'lead': 'V3'}", "V3"),
    ("WaveformData_{'lead': 'V4'}", "V4"),
    ("WaveformData_{'lead': 'V5'}", "V5"),
    ("WaveformData_{'lead': 'V6'}", "V6"),
]


ECG_SCALE_FACTOR_BY_LEAD = {
    'I': 0.004620402770859238,
    'II': 0.004445349003816847,
    'III': 0.005797972153441361,
    'V1': 0.004091206778272114,
    'V2': 0.0032893396652595597,
    'V3': 0.003212186903484846,
    'V4': 0.0027996401756864918,
    'V5': 0.002665685715648794,
    'V6': 0.0030775297385237793,
    'aVF': 0.00580883064010386,
    'aVL': 0.005936622083050387,
    'aVR': 0.005098348779426751,
}


PPG_SCALE_FACTOR = 0.00010005002501250626


FIR_FILTER = signal.firwin(
    numtaps=101,
    cutoff=[0.05, 49],
    window=('kaiser', 8),
    pass_zero=False,
    fs=600,
)


def read_xml_file(filepath: str) -> ET.ElementTree:
  """Reads an XML file and returns an ElementTree."""
  tree = ET.parse(filepath)
  return tree


def get_record(xml_tree: ET.ElementTree) -> dict[str, str]:
  """Gets 12-lead ECG waveform records from the XML tree."""
  eid, instance_index = 12345, 2

  record = {}
  record[EID] = eid
  record[INSTANCE] = instance_index
  for node in xml_tree.findall(
      './RestingECGMeasurements/MedianSamples/WaveformData'
  ):
    key = node.tag
    if node.attrib:
      key += '_' + str(node.attrib)
    if key in record:
      raise ValueError(
          f'Duplate key for record: {record} when adding node {node}.'
      )
    record[key] = node.text.strip()
  return record


def apply_filter_on_waveform_record(
    record: Mapping[str, str],
    filter_coefficients: list[float],
) -> dict[str, np.ndarray]:
  """Applies a 1D filter on an ECG record.

  Args:
    record: The ECG waveform record.
    filter_coefficients: The 1D filter.

  Returns:
    The ECG record where the waveform in each channel is filtered.
  """
  filtered_record = {}
  for key, value in record.items():
    if key == 'eid' or key == 'instance':
      filtered_record[key] = value
    else:
      # signal = [float(v) for v in value.split(',')]
      filtered_signal = np.convolve(value, filter_coefficients, mode='same')
      filtered_record[key] = filtered_signal

  return filtered_record


def get_scaled_record(
    ecg_record: dict[str, np.ndarray], scales: dict[str, float]
) -> dict[str, np.ndarray]:
  """Get scaled ecg waveform records.

  Args:
    ecg_record: 12-lead ECG waveform records.
    scales: A dictionary with scale values for the 12 leads.

  Returns:
    A dicitionary of scaled ecg waveform records which are split into train and
     validation.
  """
  scaled_record = {}
  for key in ecg_record.keys():
    if key == EID or key == INSTANCE:
      continue
    scaled_record[key] = list(ecg_record[key] * scales[key])

  return scaled_record


def parse_ppg_raw_data(raw_waveform: str) -> list[float]:
  """Parses a raw ppg data.

  Args:
    raw_waveform: The raw ppg waveform from UKB.

  Returns:
    ppg waveform as a list of float.
  """
  pieces = raw_waveform.split('|')
  n = int(pieces[0])
  assert n == 100
  assert pieces[101] == ''

  x = [int(piece.split(',')[0]) for piece in pieces[1:101]]
  y = [int(piece.split(',')[1]) for piece in pieces[1:101]]
  assert x == list(range(100))
  return y


def get_12_lead_ecg_ml_dataframe_from_ukb_raw_data(
    input_file: str
) -> pd.DataFrame:
  """Generates 12-lead ECG ML dataset from UK Biobank field 20205 demo data.

  Parses, runs preprocessing, scales the raw UK Biobank ECG xml file, turning it
  into a Pandas DataFrame which is ready for model training and inferencing.

  The input xml file can be downloaded by:
     wget  -nd  biobank.ndph.ox.ac.uk/ukb/ukb/examples/20205_2_0.xml

  Args:
    input_file: A xml file contains raw 12-lead ECG waveforms.

  Returns:
    12-lead ECG ML DataFrame.
  """
  ecg_demo_xml_tree = read_xml_file(input_file)
  ecg_xml_record = get_record(ecg_demo_xml_tree)
  ecg_record = {}
  for xml_att, lead in AT_REST_XML_ATTRIBUTE_STR_TO_LEAD:
    ecg_record[lead] = [float(v) for v in ecg_xml_record[xml_att].split(',')]

  filtered_record = apply_filter_on_waveform_record(ecg_record, FIR_FILTER)
  scaled_record = get_scaled_record(filtered_record, ECG_SCALE_FACTOR_BY_LEAD)
  ecg_ml_data_df = pd.DataFrame.from_records([scaled_record])
  return ecg_ml_data_df


def get_ecg_ppg_ml_dataframe_from_raw_data(
    ppg_input_file: str, ecg_df: pd.DataFrame,
) -> np.ndarray:
  """Generates ECG+PPG ML dataset from UKB field 20205 and 4205 demo data.

  Parses, runs preprocessing, scales the raw UK Biobank PPG data. Combines it
  with ECG lead I wave and turns it into a Pandas DataFrame which is ready for
  model training and inferencing.

  The ppg file can be downloaded by:
    wget  -nd  biobank.ndph.ox.ac.uk/ukb/ukb/examples/eg_stiff_4205.dat

  Args:
    ppg_input_file: The file contains raw ppg.
    ecg_df: A preprocessed ECG DataFrame.

  Returns:
    An ECG lead I + PPG Pandas DataFrame which is ready for model training and
    inferencing.
  """
  assert 'I' in ecg_df.columns

  f = open(ppg_input_file, 'r')
  raw_ppg = f.readline()
  f.close()
  ppg = parse_ppg_raw_data(raw_ppg)
  scaled_ppg = list(np.array(ppg) * PPG_SCALE_FACTOR)
  # ecg_ppg_record = {'ecg': ecg_df.iloc[0]['I'], 'ppg': scaled_ppg}
  ecg_ppg_arr = np.array(ecg_df.iloc[0]['I'] + scaled_ppg)
  return ecg_ppg_arr


def main(unused_argv: Sequence[str]) -> None:
  input_file = 'data/ukb_ecg_demo_20205_2_0.xml'
  if _INPUT_FILE.value:
    input_file = _INPUT_FILE.value

  ecg_ml_data_df = get_12_lead_ecg_ml_dataframe_from_ukb_raw_data(
      input_file
  )

  if _DATASET.value == 'ecg12':
    output_file = f'{_OUT_DIR.value}/ecg_ml_data.npy'
    leads = sorted(ecg_ml_data_df.columns)
    ecg_ml_arr = np.stack(ecg_ml_data_df[leads].iloc[0]).transpose()
    ecg_ml_arr = np.expand_dims(ecg_ml_arr, axis=0)
    if _DUPLICATES.value > 0:
      ecg_ml_arr = np.repeat(ecg_ml_arr, _DUPLICATES.value, axis=0)
    np.save(output_file, ecg_ml_arr)

  elif _DATASET.value == 'ecgppg':
    ppg_raw_data = 'data/ukb_ppg_demo_4205.dat'
    if _PPG_INPUT_FILE.value:
      ppg_raw_data = _PPG_INPUT_FILE.value
    ecg_ppg_arr = get_ecg_ppg_ml_dataframe_from_raw_data(
        ppg_raw_data, ecg_ml_data_df
    )
    ecg_ppg_arr = np.expand_dims(ecg_ppg_arr, axis=0)
    if _DUPLICATES.value > 0:
      ecg_ppg_arr = np.repeat(ecg_ppg_arr, _DUPLICATES.value, axis=0)
    output_file = f'{_OUT_DIR.value}/ecgppg_ml_data.npy'
    np.save(output_file, ecg_ppg_arr)

if __name__ == '__main__':
  app.run(main)
