# Copyright 2017 Paul Larsen. All rights reserved, modified from TensorFlow tutorial here:
# https://www.tensorflow.org/programmers_guide/reading_data#reading-from-files
# with the same licensing as the original, copied in below from other TensorFlow stuff:
# Copyright 2015 The TensorFlow Authors. All Rights Reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# ==============================================================================

"""IOs from csv for risklearning
Implements the inference/loss/training pattern for model building.
1. inference() - Builds the model as far as is required for running the network
forward to make predictions.
2. loss() - Adds to the inference model the layers required to generate loss.
3. training() - Adds to the loss model the Ops required to generate and
apply gradients.
This file is used by the various "fully_connected_*.py" files and not meant to
be run.
"""
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import tensorflow as tf



def read_counts(filename_queue, n_features, n_labels):
    """ Reads in count data from csv for risklearning frequency modeling
    :param filename_queue:
    :param n_features:
    :param n_labels:
    :return: example, labelas decoded tf objects
    """

    record_defaults = [[1.0] for _ in range(n_features + n_labels)]
    reader = tf.TextLineReader()
    key, record_string = reader.read(filename_queue)
    records = tf.decode_csv(record_string, record_defaults=record_defaults)
    example = records[0:n_features]
    label = records[n_features:n_features + n_labels]
    #processed_example = rlf.preprocess(example)
    return example, label


def input_pipeline(filenames, batch_size, n_features, n_labels, num_epochs=None):
    """ Creates input pipeline from list of csv files
    :param filenames:
    :param batch_size:
    :param n_features:
    :param n_labels:
    :param num_epochs:
    :return:
    """
    filename_queue = tf.train.string_input_producer(filenames, num_epochs=num_epochs, shuffle=True)
    example, label = read_counts(filename_queue, n_features, n_labels)
    # min_after_dequeue defines how big a buffer we will randomly sample
    #   from -- bigger means better shuffling but slower start up and more
    #   memory used.
    # capacity must be larger than min_after_dequeue and the amount larger
    #   determines the maximum we will prefetch.  Recommendation:
    #   min_after_dequeue + (num_threads + a small safety margin) * batch_size

    min_after_dequeue = 10000
    capacity = min_after_dequeue + 3 * batch_size
    example_batch, label_batch = tf.train.shuffle_batch(
        [example, label], batch_size=batch_size, capacity=capacity,
        min_after_dequeue=min_after_dequeue)
    return example_batch, label_batch
