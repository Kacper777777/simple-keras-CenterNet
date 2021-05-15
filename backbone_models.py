import tensorflow as tf
from tensorflow.keras.layers import Conv2D, MaxPooling2D, AveragePooling2D, Concatenate, \
    Flatten, Dense, BatchNormalization
from tensorflow.keras.models import Model


def inception_module(x, filters):
    # 1x1
    path1 = Conv2D(filters=filters[0], kernel_size=(1, 1), strides=1, padding='same', activation='relu')(x)

    # 1x1->3x3
    path2 = Conv2D(filters=filters[1][0], kernel_size=(1, 1), strides=1, padding='same', activation='relu')(x)
    path2 = Conv2D(filters=filters[1][1], kernel_size=(3, 3), strides=1, padding='same', activation='relu')(path2)

    # 1x1->5x5
    path3 = Conv2D(filters=filters[2][0], kernel_size=(1, 1), strides=1, padding='same', activation='relu')(x)
    path3 = Conv2D(filters=filters[2][1], kernel_size=(5, 5), strides=1, padding='same', activation='relu')(path3)

    # 3x3->1x1
    path4 = MaxPooling2D(pool_size=(3, 3), strides=1, padding='same')(x)
    path4 = Conv2D(filters=filters[3], kernel_size=(1, 1), strides=1, padding='same', activation='relu')(path4)

    return Concatenate(axis=-1)([path1, path2, path3, path4])


def small_conv(input_shape):
    # input image
    inputs = tf.keras.Input(shape=input_shape)

    x = Conv2D(32, kernel_size=(3, 3), strides=1,
               padding='valid', activation='relu')(inputs)
    x = MaxPooling2D(pool_size=(3, 3), strides=1, padding='valid')(x)

    x = Conv2D(64, kernel_size=(3, 3), strides=1,
               padding='valid', activation='relu')(x)
    x = MaxPooling2D(pool_size=(3, 3), strides=1, padding='valid')(x)

    x = Conv2D(128, kernel_size=(3, 3), strides=1,
               padding='valid', activation='relu')(x)
    x = MaxPooling2D(pool_size=(3, 3), strides=1, padding='valid')(x)

    x = Conv2D(256, kernel_size=(3, 3), strides=1,
               padding='valid', activation='relu')(x)
    x = MaxPooling2D(pool_size=(3, 3), strides=1, padding='valid')(x)

    x = Conv2D(512, kernel_size=(3, 3), strides=1,
               padding='valid', activation='relu')(x)
    x = MaxPooling2D(pool_size=(3, 3), strides=1, padding='valid')(x)

    x = Conv2D(1024, kernel_size=(3, 3), strides=1,
               padding='valid', activation='relu')(x)
    x = MaxPooling2D(pool_size=(3, 3), strides=1, padding='valid')(x)

    x = AveragePooling2D(pool_size=(5, 5), strides=2, padding='valid')(x)

    x = Conv2D(128, kernel_size=(1, 1), strides=1,
               padding='valid', activation='relu')(x)

    outputs = Flatten()(x)
    model = Model(inputs=inputs, outputs=outputs)
    return model


def inception_net(input_shape):
    # input image
    inputs = tf.keras.Input(shape=input_shape)

    x = Conv2D(32, kernel_size=(3, 3), strides=1, padding='valid', activation='relu')(inputs)
    x = Conv2D(64, kernel_size=(3, 3), strides=1, padding='valid', activation='relu')(x)
    x = BatchNormalization()(x)

    x = inception_module(x, [64, (64, 128), (16, 32), 16])
    x = inception_module(x, [128, (128, 192), (32, 64), 32])
    x = inception_module(x, [192, (192, 256), (64, 128), 48])
    x = BatchNormalization()(x)

    x = AveragePooling2D(pool_size=(5, 5), strides=2, padding='valid')(x)

    x = Conv2D(128, kernel_size=(1, 1), strides=2,
               padding='valid', activation='relu')(x)

    outputs = Flatten()(x)
    model = Model(inputs=inputs, outputs=outputs)
    return model


def googlenet(input_shape):
    # def auxiliary(x, class_num, name=None):
    #     layer = AveragePooling2D(pool_size=(5, 5), strides=3, padding='valid')(x)
    #     layer = Conv2D(filters=128, kernel_size=(1, 1), strides=1, padding='same', activation='relu')(layer)
    #     layer = Flatten()(layer)
    #     layer = Dense(units=256, activation='relu')(layer)
    #     layer = Dropout(0.4)(layer)
    #     layer = Dense(units=class_num, activation='softmax', name=name)(layer)
    #     return layer

    layer_in = tf.keras.Input(shape=input_shape)

    # stage-1
    layer = Conv2D(filters=64, kernel_size=(7, 7), strides=2, padding='same', activation='relu')(layer_in)
    layer = MaxPooling2D(pool_size=(3, 3), strides=2, padding='same')(layer)
    layer = BatchNormalization()(layer)

    # stage-2
    layer = Conv2D(filters=64, kernel_size=(1, 1), strides=1, padding='same', activation='relu')(layer)
    layer = Conv2D(filters=192, kernel_size=(3, 3), strides=1, padding='same', activation='relu')(layer)
    layer = BatchNormalization()(layer)
    layer = MaxPooling2D(pool_size=(3, 3), strides=2, padding='same')(layer)

    # stage-3x
    layer = inception_module(layer, [64, (96, 128), (16, 32), 32])  # 3a
    layer = inception_module(layer, [128, (128, 192), (32, 96), 64])  # 3b
    layer = MaxPooling2D(pool_size=(3, 3), strides=2, padding='same')(layer)

    # stage-4
    layer = inception_module(layer, [192, (96, 208), (16, 48), 64])  # 4a
    # aux1 = auxiliary(layer, output_num, name='aux1')
    layer = inception_module(layer, [160, (112, 224), (24, 64), 64])  # 4b
    layer = inception_module(layer, [128, (128, 256), (24, 64), 64])  # 4c
    layer = inception_module(layer, [112, (144, 288), (32, 64), 64])  # 4d
    # aux2 = auxiliary(layer, output_num, name='aux2')
    layer = inception_module(layer, [256, (160, 320), (32, 128), 128])  # 4e
    layer = MaxPooling2D(pool_size=(3, 3), strides=2, padding='same')(layer)

    # stage-5
    layer = inception_module(layer, [256, (160, 320), (32, 128), 128])  # 5a
    layer = inception_module(layer, [384, (192, 384), (48, 128), 128])  # 5b
    layer = AveragePooling2D(pool_size=(7, 7), strides=1, padding='valid')(layer)

    layer = Flatten()(layer)
    model = Model(inputs=layer_in, outputs=layer)
    return model


if __name__ == '__main__':
    # build the model
    backbone = small_conv((25, 50, 1))
    backbone.summary()
