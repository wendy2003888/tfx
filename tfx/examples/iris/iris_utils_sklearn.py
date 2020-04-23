# Lint as: python3
# Copyright 2020 Google LLC. All Rights Reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     https://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
"""Python source file include Iris pipeline functions and necessary utils.

The utilities in this file are used to build a model with scikit-learn.
This module file will be used in Transform and generic Trainer.

Note: This example uses a scikit-learn model that supports partial_fit(). To
train a model that only has fit(), set _TRAIN_BATCH_SIZE to the
size of the training dataset and fit the model on the first (and only) batch.
"""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import os
from typing import Text

import absl
import joblib
import numpy as np
from sklearn.neural_network import MLPClassifier
import tensorflow as tf
import tensorflow_transform as tft

from tfx.components.trainer.executor import TrainerFnArgs

_FEATURE_KEYS = ['sepal_length', 'sepal_width', 'petal_length', 'petal_width']
_LABEL_KEY = 'variety'
_NUM_CLASSES = 3

# Iris dataset has 150 records, and is divided to train and eval splits in 2:1
# ratio.
_TRAIN_DATA_SIZE = 100
_EVAL_DATA_SIZE = 50
_TRAIN_BATCH_SIZE = 20
_EVAL_BATCH_SIZE = 10


def _transformed_name(key):
  return key + '_xf'


def _gzip_reader_fn(filenames):
  """Small utility returning a record reader that can read gzip'ed files."""
  return tf.data.TFRecordDataset(filenames, compression_type='GZIP')


def _tf_dataset_to_numpy(dataset):
  """Converts a tf.data.dataset into a generator for numpy arrays.

  Args:
    dataset: A tf.data.dataset that contains (features, indices) tuple where
      features is a dictionary of Tensors, and indices is a single Tensor of
      label indices.

  Yields:
    A (features, indices) tuple where features is a matrix of features, and
      indices is a single vector of label indices.
  """
  for features, labels in dataset:
    x = [features[_transformed_name(key)] for key in _FEATURE_KEYS]
    x = np.concatenate(x, axis=1)
    y = np.ravel(labels)
    yield x, y


def _input_fn(file_pattern: Text,
              tf_transform_output: tft.TFTransformOutput,
              batch_size: int = 200) -> tf.data.Dataset:
  """Generates features and label for tuning/training.

  Args:
    file_pattern: input tfrecord file pattern.
    tf_transform_output: A TFTransformOutput.
    batch_size: representing the number of consecutive elements of returned
      dataset to combine in a single batch. Models without partial_fit()
      should have a batch size equal to the number of samples.

  Returns:
    A dataset that contains (features, indices) tuple where features is a
      dictionary of Tensors, and indices is a single Tensor of label indices.
  """
  transformed_feature_spec = (
      tf_transform_output.transformed_feature_spec().copy())

  dataset = tf.data.experimental.make_batched_features_dataset(
      file_pattern=file_pattern,
      batch_size=batch_size,
      features=transformed_feature_spec,
      reader=_gzip_reader_fn,
      label_key=_transformed_name(_LABEL_KEY))

  return _tf_dataset_to_numpy(dataset)


def _build_sklearn_model() -> MLPClassifier:
  """Creates a DNN scikit-learn model for classifying iris data.

  Returns:
    An untrained scikit-learn model.
  """
  model = MLPClassifier(
      hidden_layer_sizes=[8, 8, 8],
      activation='relu',
      solver='adam',
      learning_rate_init=0.0005)
  return model


def _fit_sklearn_model(model, dataset, classes, epochs=1, steps_per_epoch=None,
                       validation_data=None,
                       validation_steps=None) -> MLPClassifier:
  """Fits the scikit-learn model with the given data.

  Args:
    model: scikit-learn model to train.
    dataset: generator that produces training data in the form of
      (features, labels).
    classes: vector of ids for all classes in training data.
    epochs: number of times to iterate over the training data.
    steps_per_epoch: number of batches in an epoch.
    validation_data: tf.data.dataset containing validation data.
    validation_steps: number of batches to use for validation.

  Returns:
    A trained scikit-learn model.
  """
  if not epochs:
    return model

  step = 0
  for x, y in dataset:
    model.partial_fit(x, y, classes)
    step += 1
    if step % steps_per_epoch == 0:
      epoch = step // steps_per_epoch
      absl.logging.info('Epoch: %d, loss = %f', epoch, model.loss_)
      if epoch >= epochs:
        break

  if validation_data:
    score = _score_sklearn_model(model, validation_data, validation_steps)
    absl.logging.info('Accuracy: %f', score)

  absl.logging.info(model)
  return model


def _score_sklearn_model(model, dataset, steps):
  """Scores the scikit-learn model with the given validation data.

  Args:
    model: scikit-learn model to score.
    dataset: generator that produces validation data in the form of
      (features, labels).
    steps: number of batches to use for validation.

  Returns:
    Model accuracy.
  """
  step = score = 0
  for x, y in dataset:
    if step >= steps:
      break
    score += model.score(x, y)
    step += 1
  return score / steps


# TFX Transform will call this function.
def preprocessing_fn(inputs):
  """tf.transform's callback function for preprocessing inputs.

  Args:
    inputs: map from feature keys to raw not-yet-transformed features.

  Returns:
    Map from string feature key to transformed feature operations.
  """
  outputs = {}

  for key in _FEATURE_KEYS:
    outputs[_transformed_name(key)] = tft.scale_to_z_score(inputs[key])
  outputs[_transformed_name(_LABEL_KEY)] = inputs[_LABEL_KEY]

  return outputs


# TFX Trainer will call this function.
def run_fn(fn_args: TrainerFnArgs):
  """Train the model based on given args.

  Args:
    fn_args: Holds args used to train the model as name/value pairs.
  """
  tf_transform_output = tft.TFTransformOutput(fn_args.transform_output)

  train_dataset = _input_fn(fn_args.train_files, tf_transform_output,
                            batch_size=_TRAIN_BATCH_SIZE)
  eval_dataset = _input_fn(fn_args.eval_files, tf_transform_output,
                           batch_size=_EVAL_BATCH_SIZE)

  steps_per_epoch = _TRAIN_DATA_SIZE / _TRAIN_BATCH_SIZE

  model = _build_sklearn_model()
  model = _fit_sklearn_model(
      model=model,
      dataset=train_dataset,
      classes=np.arange(_NUM_CLASSES),
      epochs=int(fn_args.train_steps / steps_per_epoch),
      steps_per_epoch=steps_per_epoch,
      validation_data=eval_dataset,
      validation_steps=fn_args.eval_steps)

  # TODO(humichael): handle serving
  os.makedirs(fn_args.serving_model_dir)
  model_path = os.path.join(fn_args.serving_model_dir, 'saved_model.joblib')
  joblib.dump(model, model_path)
