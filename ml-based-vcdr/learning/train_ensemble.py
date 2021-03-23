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
r"""Sequentually trains `num_members` ensemble members.

Each ensemble is trained individually and uses the same configuration. The
provided `workdir` is treated as a base work directory, and each ensemble member
`i` will have a corresponding work subdirectory `{workdir}/member_{i}`.

Example usage:

$ python3 train_ensemble.py \
    --config=configs/base.py \
    --workdir=./reproduce_vcdr_ensemble_base \
    --num_members=10 \
    --main_seed=42

Note: This script assumes that the `num_members` are trained sequentially.
Assuming `N` GPUs are available, `N` models can be trained in parallel. Simply
run this script `N` times with separate `CUDA_VISIBLE_DEVICES`. For example,
if `N=3` and `num_members=10`:

$ CUDA_VISIBLE_DEVICES=0 python3 train_ensemble.py \
    --config=configs/base.py \
    --workdir=./reproduce_vcdr_ensemble_base/gpu_0 \
    --num_members=4 \
    --main_seed=42

$ CUDA_VISIBLE_DEVICES=1 python3 train_ensemble.py \
    --config=configs/base.py \
    --workdir=./reproduce_vcdr_ensemble_base/gpu_1 \
    --num_members=3 \
    --main_seed=43

$ CUDA_VISIBLE_DEVICES=2 python3 train_ensemble.py \
    --config=configs/base.py \
    --workdir=./reproduce_vcdr_ensemble_base/gpu_2 \
    --num_members=3 \
    --main_seed=44

Once training completes, all runs in `./reproduce_vcdr_ensemble_base/gpu_*` can
be moved to `./reproduce_vcdr_ensemble_base/` and models can be evaluated as an
ensemble. When moving files, it's important to avoid naming conflicts between
each GPU's members:

$ ensemble_basename="./reproduce_vcdr_ensemble_base";
  for member_path in "${ensemble_basename}"/gpu_*/*; do
    gpu_path=$(dirname "${member_path}")
    gpu_basename=$(basename "${gpu_path}")
    member_basename=$(basename "${member_path}")
    new_member_path="${ensemble_basename}/${member_basename}_${gpu_basename}"
    mv "${member_path}" "${new_member_path}"
  done
  rm -rf "${ensemble_basename}"/gpu_*

Important: In order to ensure ensemble diversity, the `main_seed` should be
different in each `CUDA_VISIBLE_DEVICES` run.
"""
import copy
import pathlib

from absl import app
from absl import flags
import ml_collections
import numpy as np
import train

FLAGS = flags.FLAGS

flags.DEFINE_integer('num_members', 10, 'The number of ensemble members.')
flags.DEFINE_integer(
    'main_seed', None,
    'The main random seed. Set this to make the random seeds used by each '
    'member deterministic.')


def train_ensemble(
    base_workdir: pathlib.Path,
    config: ml_collections.ConfigDict,
    num_members: int,
    main_seed: int = None,
) -> None:
  """Trains `num_members` models and returns training/evaluation histories."""

  # Generate seeds for each ensemble member.
  prng = np.random.RandomState(main_seed)
  ensemble_seeds = prng.randint(
      0, high=np.iinfo(np.int16).max, size=num_members)

  for i, seed in enumerate(ensemble_seeds):
    member_workdir = base_workdir / f'member_{i}'
    member_config = copy.deepcopy(config)
    member_config.seed = seed
    train.train(member_workdir, member_config)


def main(unused_argv):
  print(f'Training an ensemble with {FLAGS.num_members} members using the '
        'following config:')
  print(FLAGS.config)
  train_ensemble(
      base_workdir=pathlib.Path(FLAGS.workdir),
      config=FLAGS.config,
      num_members=FLAGS.num_members,
      main_seed=FLAGS.main_seed)


if __name__ == '__main__':
  flags.mark_flags_as_required(['config', 'workdir'])
  app.run(main)
