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
"""Lightweight binary for running spirometry pipeline training."""
from typing import Sequence

from absl import app
from absl import flags
from ml_collections.config_flags import config_flags

import train

FLAGS = flags.FLAGS

config_flags.DEFINE_config_file(
    'config',
    None,
    'A resource path to a ConfigDict training configuration. Also allows for '
    'command-line config overrides of the form `--config.seed=43.`',
    lock_config=True)
flags.DEFINE_string(
    'work_dir', None,
    'The working directory in which experiment artifacts, including logs, '
    'checkpoints, and profiling results, are written.')
flags.mark_flags_as_required(['config', 'work_dir'])


def main(argv: Sequence[str]) -> None:
  if len(argv) > 1:
    raise app.UsageError('Too many command-line arguments.')

  # Generate and save predictions from the model checkpoint.
  train.predict(work_dir=FLAGS.work_dir, config=FLAGS.config)


if __name__ == '__main__':
  app.run(main)
