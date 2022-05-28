"""
Maxout(活性化関数)
コード引用(一部改変) : https://qiita.com/ozora/items/c7d1f2e0b113f11d419d
"""

import tensorflow as tf
from keras import layers

class Maxout(layers.Layer):
    #num_unitで出力後の次元数を指定
    #axisでMaxをとりたい軸を指定（通常はデフォルト値。Channel firstの場合は1を指定してください）
    def __init__(self, num_units: int, axis: int = -1, **kwargs):
        super().__init__(**kwargs)
        self.num_units = num_units
        self.axis = axis

    def call(self, inputs):
        inputs = tf.convert_to_tensor(inputs)
        shape = inputs.get_shape().as_list()
        # Dealing with batches with arbitrary sizes
        for i in range(len(shape)):
            if shape[i] is None:
                shape[i] = tf.shape(inputs)[i]

        num_channels = shape[self.axis]
        if not isinstance(num_channels, tf.Tensor) and num_channels % self.num_units:
            raise ValueError(
                "number of features({}) is not "
                "a multiple of num_units({})".format(num_channels, self.num_units)
            )

        if self.axis < 0:
            axis = self.axis + len(shape)
        else:
            axis = self.axis
        assert axis >= 0, "Find invalid axis: {}".format(self.axis)

        expand_shape = shape[:]
        expand_shape[axis] = self.num_units
        k = num_channels // self.num_units
        expand_shape.insert(axis, k)

        outputs = tf.math.reduce_max(
            tf.reshape(inputs, expand_shape), axis, keepdims=False
        )
        return outputs

    def compute_output_shape(self, input_shape):
        input_shape = tf.TensorShape(input_shape).as_list()
        input_shape[self.axis] = self.num_units
        return tf.TensorShape(input_shape)

    def get_config(self):
        config = {"num_units": self.num_units, "axis": self.axis}
        base_config = super().get_config()
        return {**base_config, **config}