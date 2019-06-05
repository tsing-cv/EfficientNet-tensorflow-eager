# Copyright 2019 The TensorFlow Authors. All Rights Reserved.
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
"""Contains definitions for EfficientNet model.

[1] Mingxing Tan, Quoc V. Le
  EfficientNet: Rethinking Model Scaling for Convolutional Neural Networks.
  ICML'19, https://arxiv.org/abs/1905.11946
"""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import numpy as np
import tensorflow as tf
import tensorflow.keras.layers as KL
from tensorflow.keras.utils import get_file

from params import get_model_params, IMAGENET_WEIGHTS
from initializers import conv_kernel_initializer, dense_kernel_initializer


def round_filters(filters, global_params):
    """Round number of filters based on depth multiplier."""
    orig_f = filters
    multiplier = global_params.width_coefficient
    divisor = global_params.depth_divisor
    min_depth = global_params.min_depth
    if not multiplier:
        return filters

    filters *= multiplier
    min_depth = min_depth or divisor
    new_filters = max(min_depth, int(filters + divisor / 2) // divisor * divisor)
    # Make sure that round down does not go down by more than 10%.
    if new_filters < 0.9 * filters:
        new_filters += divisor
    # print('round_filter input={} output={}'.format(orig_f, new_filters))
    return int(new_filters)


def round_repeats(repeats, global_params):
    """Round number of filters based on depth multiplier."""
    multiplier = global_params.depth_coefficient
    if not multiplier:
        return repeats
    return int(np.ceil(multiplier * repeats))

class Swish(KL.Layer):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
    def call(self, inputs):
        return tf.nn.swish(inputs)


class DropConnect(KL.Layer):

    def __init__(self, drop_connect_rate=0., **kwargs):
        super().__init__(**kwargs)
        self.drop_connect_rate = drop_connect_rate

    def call(self, inputs, training=None):
        keep_prob = 1.0 - self.drop_connect_rate

        # Compute drop_connect tensor
        batch_size = tf.shape(inputs)[0]
        random_tensor = keep_prob
        random_tensor += tf.random_uniform([batch_size, 1, 1, 1], dtype=inputs.dtype)
        binary_tensor = tf.floor(random_tensor)
        output = tf.div(inputs, keep_prob) * binary_tensor
        return output


class SEBlock(KL.Layer):
    def __init__(self, block_args, global_params, name='seblock', **kwargs):
        super().__init__(name=name, **kwargs)
        num_reduced_filters = max(1, int(block_args.input_filters * block_args.se_ratio))
        filters = block_args.input_filters * block_args.expand_ratio
        self.gap = KL.Lambda(lambda a: tf.reduce_mean(a, axis=[1, 2], keepdims=True))
        self.conv1 = KL.Conv2D(num_reduced_filters,
                                kernel_size=[1, 1],
                                strides=[1, 1],
                                kernel_initializer=conv_kernel_initializer,
                                padding='same',
                                use_bias=True)
        self.act1 = Swish()#KL.ReLU()
        # Excite
        self.conv2 = KL.Conv2D(filters,
                                kernel_size=[1, 1],
                                strides=[1, 1],
                                kernel_initializer=conv_kernel_initializer,
                                padding='same',
                                use_bias=True)
        self.act2 = KL.Activation('sigmoid')
    def call(self, inputs, training=False):
        x = self.gap(inputs)
        x = self.conv1(x)
        x = self.act1(x)
        # Excite
        x = self.conv2(x)
        x = self.act2(x)
        out = tf.math.multiply(x, inputs)
        return out


class MBConvBlock(KL.Layer):
    def __init__(self, block_args, global_params, drop_connect_rate=None, name='mbconvblock', **kwargs):
        super().__init__(name=name, **kwargs)
        batch_norm_momentum = global_params.batch_norm_momentum
        batch_norm_epsilon = global_params.batch_norm_epsilon
        self.has_se = (block_args.se_ratio is not None) and (
            block_args.se_ratio > 0) and (block_args.se_ratio <= 1)

        filters = block_args.input_filters * block_args.expand_ratio
        kernel_size = block_args.kernel_size
        self.block_args = block_args
        self.drop_connect_rate = drop_connect_rate
        self.conv = KL.Conv2D(filters,
                            kernel_size=[1, 1],
                            strides=[1, 1],
                            kernel_initializer=conv_kernel_initializer,
                            padding='same',
                            use_bias=False)
        self.norm = KL.BatchNormalization(axis=-1,
                                        momentum=batch_norm_momentum,
                                        epsilon=batch_norm_epsilon)
        self.act = Swish()#KL.ReLU()

        self.conv1 = KL.DepthwiseConv2D([kernel_size, kernel_size],
                                        strides=block_args.strides,
                                        depthwise_initializer=conv_kernel_initializer,
                                        padding='same',
                                        use_bias=False)
        self.norm1 = KL.BatchNormalization(axis=-1,
                                        momentum=batch_norm_momentum,
                                        epsilon=batch_norm_epsilon)
        self.act1 = Swish()#KL.ReLU()


        self.seblock = SEBlock(block_args, global_params)

        self.conv2 = KL.Conv2D(block_args.output_filters,
                            kernel_size=[1, 1],
                            strides=[1, 1],
                            kernel_initializer=conv_kernel_initializer,
                            padding='same',
                            use_bias=False)
        self.norm2 = KL.BatchNormalization(axis=-1,
                                        momentum=batch_norm_momentum,
                                        epsilon=batch_norm_epsilon)
        self.dropconnect = DropConnect(drop_connect_rate)
    def call(self, inputs, training=False):
        if self.block_args.expand_ratio != 1:
            x = self.conv(inputs)
            x = self.norm(x, training=training)
            x = self.act(x)
        else:
            x = inputs

        x = self.conv1(x)
        x = self.norm1(x, training=training)
        x = self.act1(x)

        if self.has_se:
            x = self.seblock(x, training=training)

        # output phase
        x = self.conv2(x)
        x = self.norm2(x, training=training)

        if self.block_args.id_skip:
            if all(s == 1 for s in self.block_args.strides) and self.block_args.input_filters == self.block_args.output_filters:
                # only apply drop_connect if skip presents.
                if self.drop_connect_rate:
                    x = self.dropconnect(x)
                x = tf.math.add(x, inputs)
        return x


class EfficientNet(tf.keras.Model):
    def __init__(self, block_args_list, global_params, include_top=True, name='efficientnet', **kwargs):
        super().__init__(name=name, **kwargs)
        batch_norm_momentum = global_params.batch_norm_momentum
        batch_norm_epsilon = global_params.batch_norm_epsilon
        self.block_args_list = block_args_list
        self.global_params = global_params
        self.include_top = include_top
        self.conv1 = KL.Conv2D(filters=round_filters(32, global_params),
                            kernel_size=[3, 3],
                            strides=[2, 2],
                            kernel_initializer=conv_kernel_initializer,
                            padding='same',
                            use_bias=False)
        self.norm1 = KL.BatchNormalization(axis=-1,
                                        momentum=batch_norm_momentum,
                                        epsilon=batch_norm_epsilon)
        self.act1 = Swish()#KL.ReLU()

        # Blocks part
        block_idx = 1
        n_blocks = sum([block_args.num_repeat for block_args in block_args_list])
        drop_rate = global_params.drop_connect_rate or 0
        drop_rate_dx = drop_rate / n_blocks

        for block_args in block_args_list:
            assert block_args.num_repeat > 0
            # Update block input and output filters based on depth multiplier.
            block_args = block_args._replace(
                input_filters=round_filters(block_args.input_filters, global_params),
                output_filters=round_filters(block_args.output_filters, global_params),
                num_repeat=round_repeats(block_args.num_repeat, global_params)
            )

            # The first block needs to take care of stride and filter size increase.
            setattr(self, f"mbconvblock{block_idx}", MBConvBlock(block_args, global_params,
                            drop_connect_rate=drop_rate_dx * block_idx))
            block_idx += 1

            if block_args.num_repeat > 1:
                block_args = block_args._replace(input_filters=block_args.output_filters, strides=[1, 1])

            for _ in range(block_args.num_repeat - 1):
                setattr(self, f"mbconvblock{block_idx}", MBConvBlock(block_args, global_params,
                                drop_connect_rate=drop_rate_dx * block_idx))
                block_idx += 1

        # Head part
        self.conv2 = KL.Conv2D(filters=round_filters(1280, global_params),
                            kernel_size=[1, 1],
                            strides=[1, 1],
                            kernel_initializer=conv_kernel_initializer,
                            padding='same',
                            use_bias=False)
        self.norm2 = KL.BatchNormalization(axis=-1,
                                        momentum=batch_norm_momentum,
                                        epsilon=batch_norm_epsilon)
        self.act2 = Swish()#KL.ReLU()


        self.gap = KL.GlobalAveragePooling2D(data_format=global_params.data_format)
        self.dropout = KL.Dropout(global_params.dropout_rate)
        self.fc = KL.Dense(global_params.num_classes, kernel_initializer=dense_kernel_initializer)
        self.softmax = KL.Activation('softmax')
    def call(self, inputs, training=True):
        x = self.conv1(inputs)
        x = self.norm1(x, training=training)
        x = self.act1(x)

        block_idx = 1
        for block_args in self.block_args_list:
            x = getattr(self, f"mbconvblock{block_idx}")(x, training=training)
            block_idx += 1
            if block_args.num_repeat > 1:
                block_args = block_args._replace(input_filters=block_args.output_filters, strides=[1, 1])

            for _ in range(block_args.num_repeat - 1):
                x = getattr(self, f"mbconvblock{block_idx}")(x, training=training)
                block_idx += 1

        # Head part
        x = self.conv2(x)
        x = self.norm2(x, training=training)
        x = self.act2(x)

        if self.include_top:
            x = self.gap(x)
            if self.global_params.dropout_rate > 0:
                x = self.dropout(x)
            x = self.fc(x)
            x = self.softmax(x)
        return x


def _get_model_by_name(model_name, input_shape=None, include_top=True, weights=None, classes=1000):
    """Reference: https://arxiv.org/abs/1807.11626
    Args:
        input_shape: optional, if ``None`` default_input_shape is used
            EfficientNetB0 - (224, 224, 3)
            EfficientNetB1 - (240, 240, 3)
            EfficientNetB2 - (260, 260, 3)
            EfficientNetB3 - (300, 300, 3)
            EfficientNetB4 - (380, 380, 3)
            EfficientNetB5 - (456, 456, 3)
            EfficientNetB6 - (528, 528, 3)
            EfficientNetB7 - (600, 600, 3)
        include_top: whether to include the fully-connected
            layer at the top of the network.
        weights: one of `None` (random initialization),
              'imagenet' (pre-training on ImageNet).
        classes: optional number of classes to classify images
            into, only to be specified if `include_top` is True, and
            if no `weights` argument is specified.

    Returns:
        A Keras model instance.

    """
    if weights not in {None, 'imagenet'}:
        raise ValueError('Parameter `weights` should be one of [None, "imagenet"]')

    if weights == 'imagenet' and model_name not in IMAGENET_WEIGHTS:
        raise ValueError('There are not pretrained weights for {} model.'.format(model_name))

    if weights == 'imagenet' and include_top and classes != 1000:
        raise ValueError('If using `weights` and `include_top`'
                         ' `classes` should be 1000')

    block_agrs_list, global_params, default_input_shape = get_model_params(
        model_name, override_params={'num_classes': classes}
    )

    model = EfficientNet(block_agrs_list, global_params, include_top=include_top)

    if weights:
        if not include_top:
            weights_name = model_name + '-notop'
        else:
            weights_name = model_name
        weights = IMAGENET_WEIGHTS[weights_name]
        weights_path = get_file(weights['name'],weights['url'],cache_subdir='models',md5_hash=weights['md5'])
        model.load_weights(weights_path)

    return model


def EfficientNetB0(include_top=True, weights=None, classes=1000):
    return _get_model_by_name('efficientnet-b0', include_top=include_top, weights=weights, classes=classes)

def EfficientNetB1(include_top=True, weights=None, classes=1000):
    return _get_model_by_name('efficientnet-b1', include_top=include_top, weights=weights, classes=classes)

def EfficientNetB2(include_top=True, weights=None, classes=1000):
    return _get_model_by_name('efficientnet-b2', include_top=include_top, weights=weights, classes=classes)

def EfficientNetB3(include_top=True, weights=None, classes=1000):
    return _get_model_by_name('efficientnet-b3', include_top=include_top, weights=weights, classes=classes)

def EfficientNetB4(include_top=True, weights=None, classes=1000):
    return _get_model_by_name('efficientnet-b4', include_top=include_top, weights=weights, classes=classes)

def EfficientNetB5(include_top=True, weights=None, classes=1000):
    return _get_model_by_name('efficientnet-b5', include_top=include_top, weights=weights, classes=classes)

def EfficientNetB6(include_top=True, weights=None, classes=1000):
    return _get_model_by_name('efficientnet-b6', include_top=include_top, weights=weights, classes=classes)

def EfficientNetB7(include_top=True, weights=None, classes=1000):
    return _get_model_by_name('efficientnet-b7', include_top=include_top, weights=weights, classes=classes)


