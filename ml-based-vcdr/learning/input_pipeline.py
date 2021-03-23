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
"""Utility functions for loading `TFRecord`s as `tf.data.Dataset`s."""
import enum
from typing import Callable
from typing import Dict
from typing import List
from typing import Tuple

import ml_collections
import tensorflow as tf

TensorDict = Dict[str, tf.Tensor]
TensorDictTriple = Tuple[TensorDict, TensorDict, TensorDict]

IMAGE_KEY = 'image'


class Split(enum.Enum):
  """Denotes the train, evaluation, test, and prediction splits.

  These values correspond to the dataset path keys in `config.dataset`.
  """
  TRAIN = 'train'
  EVAL = 'eval'
  TEST = 'test'
  PREDICT = 'predict'


def _get_augment_element_fn(
    dataset_config: ml_collections.ConfigDict
) -> Callable[[TensorDict, TensorDict, TensorDict], TensorDictTriple]:
  """"Returns a function that augments the `IMAGE_KEY` input tensor.

  The following transformations are applied if specified in `dataset_config`:

    - tf.image.random_flip_left_right
    - tf.image.random_flip_up_down
    - tf.image.random_brightness
    - tf.image.random_hue
    - tf.image.random_saturation
    - tf.image.random_contrast

  Important: The returned function requires that image tensors have dtype
  tf.float32 and have values in range [0, 1]. Augmented images are then clipped
  back to the [0, 1] range.

  Args:
    dataset_config: A ConfigDict used to build the set of applied augmentations.

  Returns:
    A function that applies the set of augmentations to an input TensorDict's
    `IMAGE_KEY` image.
  """

  horizontal_flip = dataset_config.get('random_horizontal_flip', False)
  vertical_flip = dataset_config.get('random_vertical_flip', False)
  brightness_max_delta = dataset_config.get('random_brightness_max_delta', None)
  hue_max_delta = dataset_config.get('random_hue_max_delta', None)

  saturation_lower = dataset_config.get('random_saturation_lower', None)
  saturation_upper = dataset_config.get('random_saturation_upper', None)
  apply_saturation = saturation_lower and saturation_upper
  if apply_saturation and (saturation_upper <= saturation_lower):
    raise ValueError(
        f'Invalid saturation range: ({saturation_lower}, {saturation_upper})')

  contrast_lower = dataset_config.get('random_contrast_lower', None)
  contrast_upper = dataset_config.get('random_contrast_upper', None)
  apply_contrast = contrast_lower and contrast_upper
  if apply_contrast and (contrast_upper <= contrast_lower):
    raise ValueError(
        f'Invalid contrast range: ({contrast_lower}, {contrast_upper})')

  def _augment_element_fn(inputs: TensorDict, labels: TensorDict,
                          weights: TensorDict) -> TensorDictTriple:

    image = inputs[IMAGE_KEY]

    # Ensure images are in the expected format. Image augmentations assume that
    # the image tensor is a tf.float32 and contains pixels in range [0, 1].
    tf.debugging.assert_type(image, tf.float32)
    tf.debugging.assert_less_equal(tf.math.reduce_max(image), 1.0)
    tf.debugging.assert_greater_equal(tf.math.reduce_min(image), 0.0)

    if horizontal_flip:
      image = tf.image.random_flip_left_right(image)
    if vertical_flip:
      image = tf.image.random_flip_up_down(image)
    if brightness_max_delta:
      image = tf.image.random_brightness(image, max_delta=brightness_max_delta)
    if hue_max_delta:
      image = tf.image.random_hue(image, max_delta=hue_max_delta)
    if apply_saturation:
      image = tf.image.random_saturation(
          image, lower=saturation_lower, upper=saturation_upper)
    if apply_contrast:
      image = tf.image.random_contrast(
          image, lower=contrast_lower, upper=contrast_upper)

    # Clip image back to [0.0, 1.0] prior to architecture-specific centering.
    image = tf.clip_by_value(image, 0.0, 1.0)

    inputs[IMAGE_KEY] = image

    return inputs, labels, weights

  return _augment_element_fn


def _build_tf_record_features(
    split: Split, outcomes: List[ml_collections.ConfigDict]
) -> Dict[str, tf.io.FixedLenFeature]:
  """Returns a feature dictionary used to parse TFRecord examples.

  We assume that the TFRecords are defined using the following schema:

    1. An encoded image with key `image/encoded` that can be decoded using
       `tf.image.decode_png`.
    2. A unique identifier for each image with key `image/id`.
    3. Two keys for each outcome, `image/{outcome.name}/value` and
       `image/{outcome.name}/weight`, corresponding to the outcome value and
       a weight applied to the examples's loss for the outcome head.

  The `tf.io.parse_single_example` function uses resulting feature dictionary
  to parse each TFRecord.

  Args:
    split: The dataset split (train, validation, test, or predict).
    outcomes: A list of outcomes ConfigDicts corresponding to model heads.

  Returns:
    A feature dictionary for parsing TFRecords.
  """

  def _get_value_key(outcome: ml_collections.ConfigDict) -> str:
    return f'image/{outcome.name}/value'

  def _get_weight_key(outcome: ml_collections.ConfigDict) -> str:
    return f'image/{outcome.name}/weight'

  features = {}
  features['image/encoded'] = tf.io.FixedLenFeature([], tf.string)
  features['image/id'] = tf.io.FixedLenFeature([1], tf.string)

  if split != Split.PREDICT:
    for outcome in outcomes:
      value_key = _get_value_key(outcome)
      weight_key = _get_weight_key(outcome)
      num_classes = outcome.num_classes
      features[value_key] = tf.io.FixedLenFeature([num_classes], tf.float32)
      features[weight_key] = tf.io.FixedLenFeature([1], tf.float32)

  return features


def _get_parse_example_fn(
    features: Dict[str, tf.io.FixedLenFeature]
) -> Callable[[tf.train.Example], TensorDict]:
  """Returns a function that parses a TFRecord example using `features`."""

  def _parse_example(example: tf.train.Example) -> TensorDict:
    return tf.io.parse_single_example(example, features)

  return _parse_example


def _get_rename_keys(
    features: Dict[str, tf.io.FixedLenFeature]
) -> Callable[[TensorDict], TensorDictTriple]:
  """Returns a function that converts TensorDicts to TensorDictTriples.

  The triple contains three TensorDicts: `(inputs, labels, weights)`. Labels and
  weights correspond to keys with `/value` and `/weight` suffixes, respectively.
  All other keys are considered input keys. The `image/encoded` key in
  `features` is renamed to `IMAGE_KEY`.

  Args:
    features: The features dictionary used by `tf.io.parse_single_example`.

  Returns:
    A function that converts a single TensorDict into separate inputs, labels,
    and weights TensorDicts.
  """
  weight_keys = {key for key in features if key.endswith('/weight')}
  label_keys = {key for key in features if key.endswith('/value')}
  input_keys = set(features.keys()) - weight_keys - label_keys

  def _rename_keys(example: TensorDict) -> TensorDictTriple:
    inputs = {}
    labels = {}
    weights = {}

    for key, value in example.items():
      key_split = key.split('/')
      new_key = key_split[1]

      if key in weight_keys:
        weights[new_key] = value
      elif key in label_keys:
        labels[new_key] = value
      elif key in input_keys:
        new_key = IMAGE_KEY if new_key == 'encoded' else new_key
        inputs[new_key] = value
      else:
        raise ValueError(f'Unexpected key: {key}.')
    return inputs, labels, weights

  return _rename_keys


def _decode_img(inputs: TensorDict, labels: TensorDict,
                weights: TensorDict) -> TensorDictTriple:
  """Decodes the input's `IMAGE_KEY` tensor and casts to tf.float32."""
  image = inputs[IMAGE_KEY]
  image = tf.image.decode_png(image, channels=3)
  image = tf.image.convert_image_dtype(image, tf.float32)
  inputs[IMAGE_KEY] = image
  return inputs, labels, weights


def _resize_and_center(
    inputs: TensorDict,
    labels: TensorDict,
    weights: TensorDict,
    image_size: Tuple[int, int] = (587, 587)
) -> TensorDictTriple:
  """Resizes the input image to `image_size` and shifts pixels to [-1, 1].

  Note: Images must be of dtype tf.float32 with pixel values in range [0, 1].

  Args:
    inputs: An input TensorDict containing an image with key `IMAGE_KEY`.
    labels: A label TensorDict; not modified in this map.
    weights: A sample weight TensorDict; not modified in this map.
    image_size: A width-height tuple used to resize the image.

  Returns:
    A TensorDictTriple containg inputs and labels. The inputs TensorDict
    contains a resized and centered image tensor with key `IMAGE_KEY`.
  """
  image = inputs[IMAGE_KEY]
  image = tf.image.resize(image, image_size)
  # Note: We do not use `tf.keras.applications.inception_v3.preprocess_input`
  # since our image augmentations assume pixels in [0, 1] rather than [0, 255].
  # We reproduce the method's logic here.
  image = tf.math.subtract(image, 0.5)
  image = tf.math.multiply(image, 2.0)
  inputs[IMAGE_KEY] = image
  return inputs, labels, weights


def _load_data(
    split: Split,
    dataset_config: ml_collections.ConfigDict,
    outcomes: List[ml_collections.ConfigDict],
) -> tf.data.Dataset:
  """Loads and parses TFRecords for the given dataset split.

  Elements are `TensorDictTriple`s and contain inputs, labels, and weights. The
  `inputs: TensorDict` must contain an `IMAGE_KEY` key with rgb tensor values of
  shape `dataset_config.image_size`. The `labels: TensorDict` should contain one
  key per outcome, while the `weights: TensorDict` should contain
  `subsample_weights` for each outcome.

  Args:
    split: The dataset split (train, validation, test, or predict).
    dataset_config: A dataset ConfigDict containing hparams and augmentations.
    outcomes: A list of outcome ConfigDicts used to define labels.

  Returns:
    A dataset containing decoded image tensors, labels, and weights.
  """
  # Build features for parsing TFRecords.
  features = _build_tf_record_features(split, outcomes)

  # Fetch the list of globbed TFRecords filepaths.
  filenames_ds = tf.data.Dataset.list_files(dataset_config[split.value])

  # Convert each filepath to a TFRecord.
  tf_record_ds = tf.data.TFRecordDataset(filenames=filenames_ds)

  # Convert each TFRecord to a TensorDict.
  ds = tf_record_ds.map(
      _get_parse_example_fn(features), num_parallel_calls=tf.data.AUTOTUNE)

  # Rename keys and break features into inputs, labels, and weights.
  ds = ds.map(_get_rename_keys(features), num_parallel_calls=tf.data.AUTOTUNE)

  # Decode the images.
  ds = ds.map(_decode_img, num_parallel_calls=tf.data.AUTOTUNE)

  return ds


def build_dataset(split: Split,
                  dataset_config: ml_collections.ConfigDict,
                  outcomes: List[ml_collections.ConfigDict],
                  shuffle: bool = False,
                  shuffle_buffer: int = 64,
                  repeat: bool = False,
                  cache: bool = False) -> tf.data.Dataset:
  """Builds a dataset for the given data split."""
  ds = _load_data(split, dataset_config, outcomes)
  if cache:
    ds = ds.cache()
  # Use image augmentation when training.
  if split == Split.TRAIN:
    ds = ds.map(
        _get_augment_element_fn(dataset_config),
        num_parallel_calls=tf.data.AUTOTUNE)
  ds = ds.map(_resize_and_center, num_parallel_calls=tf.data.AUTOTUNE)
  if shuffle:
    ds = ds.shuffle(shuffle_buffer)
  ds = ds.batch(dataset_config.batch_size, drop_remainder=True)
  if repeat:
    ds = ds.repeat()
  ds = ds.prefetch(tf.data.experimental.AUTOTUNE)
  return ds


def build_train_dataset(dataset_config: ml_collections.ConfigDict,
                        outcomes: List[ml_collections.ConfigDict],
                        shuffle: bool = True,
                        shuffle_buffer: int = 64,
                        repeat: bool = True,
                        cache: bool = False) -> tf.data.Dataset:
  """Returns the train dataset."""
  return build_dataset(
      Split.TRAIN,
      dataset_config,
      outcomes,
      shuffle=shuffle,
      shuffle_buffer=shuffle_buffer,
      repeat=repeat,
      cache=cache)


def build_eval_dataset(dataset_config: ml_collections.ConfigDict,
                       outcomes: List[ml_collections.ConfigDict],
                       cache: bool = False) -> tf.data.Dataset:
  """Returns the evaluation dataset."""
  return build_dataset(Split.EVAL, dataset_config, outcomes, cache=cache)


def build_predict_dataset(dataset_config: ml_collections.ConfigDict,
                          outcomes: List[ml_collections.ConfigDict],
                          cache: bool = False) -> tf.data.Dataset:
  """Returns the evaluation dataset."""
  return build_dataset(Split.PREDICT, dataset_config, outcomes, cache=cache)


def build_datasets(
    dataset_config: ml_collections.ConfigDict,
    outcomes: List[ml_collections.ConfigDict],
    cache: bool = False) -> Tuple[tf.data.Dataset, tf.data.Dataset]:
  """Returns train and evaluation datasets."""
  train_ds = build_train_dataset(
      dataset_config, outcomes, shuffle=True, repeat=True, cache=cache)
  eval_ds = build_eval_dataset(dataset_config, outcomes, cache=cache)
  return train_ds, eval_ds
