import tensorflow as tf
from tensorflow.keras.layers import Conv2D, Conv2DTranspose, MaxPooling2D, AveragePooling2D, \
    Flatten, Dense, Reshape, Concatenate, Add, Input, Lambda, BatchNormalization
from tensorflow.keras.models import Model
from tensorflow_core.python.keras.layers import ReLU
from tensorflow_core.python.keras.regularizers import l2
from tensorflow.keras.applications import ResNet50
from losses import main_loss


def topk(hm, max_objects=20):
    b, h, w, c = tf.shape(hm)[0], tf.shape(hm)[1], tf.shape(hm)[2], tf.shape(hm)[3]
    hm = tf.reshape(hm, (b, -1))  # (b, h * w * c)
    scores, indices = tf.nn.top_k(hm, k=max_objects)  # (b, k), (b, k)
    class_ids = indices % c
    xs = indices // c % w
    ys = indices // c // w
    indices = ys * w + xs
    return scores, indices, class_ids, xs, ys


def decode(hm, wh, reg, max_objects=20):
    scores, indices, class_ids, xs, ys = topk(hm, max_objects=max_objects)
    b = tf.shape(hm)[0]
    # (b, h * w, 2)
    reg = tf.reshape(reg, (b, -1, tf.shape(reg)[-1]))
    # (b, h * w, 2)
    wh = tf.reshape(wh, (b, -1, tf.shape(wh)[-1]))
    # (b, k, 2)
    topk_reg = tf.gather(reg, indices, batch_dims=1)
    # (b, k, 2)
    topk_wh = tf.cast(tf.gather(wh, indices, batch_dims=1), tf.float32)
    topk_cx = tf.cast(tf.expand_dims(xs, axis=-1), tf.float32) + topk_reg[..., 0:1]
    topk_cy = tf.cast(tf.expand_dims(ys, axis=-1), tf.float32) + topk_reg[..., 1:2]
    scores = tf.expand_dims(scores, axis=-1)
    class_ids = tf.cast(tf.expand_dims(class_ids, axis=-1), tf.float32)
    topk_x1 = topk_cx - topk_wh[..., 0:1] / 2
    topk_x2 = topk_cx + topk_wh[..., 0:1] / 2
    topk_y1 = topk_cy - topk_wh[..., 1:2] / 2
    topk_y2 = topk_cy + topk_wh[..., 1:2] / 2
    # (b, k, 6)
    detections = tf.concat([topk_x1, topk_y1, topk_x2, topk_y2, scores, class_ids], axis=-1)
    return detections


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


def googlenet(input_shape=(512, 512, 3), output_shape=(128, 128), num_classes=1, max_objects=20):

    image_input = tf.keras.Input(shape=input_shape)

    # stage-1
    layer = Conv2D(filters=64, kernel_size=(7, 7), strides=2, padding='same', activation='relu')(image_input)
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
    layer = AveragePooling2D(pool_size=(9, 9), strides=1, padding='valid')(layer)

    # layer = Conv2DTranspose(filters=256, kernel_size=3, strides=2, padding='same')(layer)
    layer = Conv2DTranspose(filters=128, kernel_size=3, strides=2, padding='same')(layer)
    layer = Conv2DTranspose(filters=64, kernel_size=3, strides=2, padding='same')(layer)
    layer = Conv2DTranspose(filters=32, kernel_size=3, strides=2, padding='same')(layer)
    layer = Conv2DTranspose(filters=1, kernel_size=3, strides=2, padding='same')(layer)

    backbone_output = layer

    hm_input = Input(shape=(output_shape[0], output_shape[1], num_classes))
    wh_input = Input(shape=(max_objects, 2))
    reg_input = Input(shape=(max_objects, 2))
    reg_mask_input = Input(shape=(max_objects,))
    index_input = Input(shape=(max_objects,))

    x = BatchNormalization()(backbone_output)

    # hm header
    y1 = Conv2D(64, 1, kernel_initializer='he_normal', padding='same')(x)
    y1 = BatchNormalization()(y1)
    y1 = ReLU()(y1)
    y1 = Conv2D(num_classes, 1, kernel_initializer='he_normal', kernel_regularizer=l2(5e-4),
                activation='sigmoid')(y1)

    # wh header
    y2 = Conv2D(64, 1, kernel_initializer='he_normal', padding='same')(x)
    y2 = BatchNormalization()(y2)
    y2 = ReLU()(y2)
    y2 = Conv2D(2, 1, kernel_initializer='he_normal', kernel_regularizer=l2(5e-4))(y2)

    # reg header
    y3 = Conv2D(64, 1, kernel_initializer='he_normal', padding='same')(x)
    y3 = BatchNormalization()(y3)
    y3 = ReLU()(y3)
    y3 = Conv2D(2, 1, kernel_initializer='he_normal', kernel_regularizer=l2(5e-4))(y3)

    loss_ = Lambda(main_loss, name='centernet_loss')(
        [y1, y2, y3, hm_input, wh_input, reg_input, reg_mask_input, index_input])
    model = Model(inputs=[image_input, hm_input, wh_input, reg_input,
                          reg_mask_input, index_input], outputs=[loss_])

    detections = Lambda(lambda z: decode(*z))([y1, y2, y3])
    prediction_model = Model(inputs=image_input, outputs=detections)
    debug_model = Model(inputs=image_input, outputs=[y1, y2, y3])
    return model, prediction_model, debug_model


def small_conv(input_shape=(64, 64, 1), output_shape=(16, 16), num_classes=10, max_objects=20):
    # input image
    image_input = tf.keras.Input(shape=input_shape)

    x = Conv2D(32, kernel_size=(3, 3), strides=1,
               padding='valid', activation='relu')(image_input)
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

    x = Conv2D(512, kernel_size=(3, 3), strides=2,
               padding='valid', activation='relu')(x)
    x = MaxPooling2D(pool_size=(3, 3), strides=2, padding='valid')(x)

    x = AveragePooling2D(pool_size=(4, 4), strides=1, padding='valid')(x)

    x = Conv2DTranspose(filters=128, kernel_size=3, strides=1, padding='valid')(x)
    x = Conv2DTranspose(filters=64, kernel_size=3, strides=1, padding='valid')(x)
    x = Conv2DTranspose(filters=32, kernel_size=3, strides=1, padding='valid')(x)
    x = Conv2DTranspose(filters=1, kernel_size=3, strides=1, padding='valid')(x)

    backbone_output = x
    x = BatchNormalization()(backbone_output)
    # model = Model(image_input=image_input, outputs=outputs)
    # return model
    ##########

    hm_input = Input(shape=(output_shape[0], output_shape[1], num_classes))
    wh_input = Input(shape=(max_objects, 2))
    reg_input = Input(shape=(max_objects, 2))
    reg_mask_input = Input(shape=(max_objects,))
    index_input = Input(shape=(max_objects,))

    # hm header
    y1 = Conv2D(64, 1, kernel_initializer='he_normal', padding='same')(x)
    y1 = BatchNormalization()(y1)
    y1 = ReLU()(y1)
    y1 = Conv2D(num_classes, 1, kernel_initializer='he_normal', kernel_regularizer=l2(5e-4),
                activation='sigmoid')(y1)

    # wh header
    y2 = Conv2D(64, 1, kernel_initializer='he_normal', padding='same')(x)
    y2 = BatchNormalization()(y2)
    y2 = ReLU()(y2)
    y2 = Conv2D(2, 1, kernel_initializer='he_normal', kernel_regularizer=l2(5e-4))(y2)

    # reg header
    y3 = Conv2D(64, 1, kernel_initializer='he_normal', padding='same')(x)
    y3 = BatchNormalization()(y3)
    y3 = ReLU()(y3)
    y3 = Conv2D(2, 1, kernel_initializer='he_normal', kernel_regularizer=l2(5e-4))(y3)

    loss_ = Lambda(main_loss, name='centernet_loss')(
        [y1, y2, y3, hm_input, wh_input, reg_input, reg_mask_input, index_input])
    model = Model(inputs=[image_input, hm_input, wh_input, reg_input,
                          reg_mask_input, index_input], outputs=[loss_])

    detections = Lambda(lambda z: decode(*z))([y1, y2, y3])
    prediction_model = Model(inputs=image_input, outputs=detections)
    debug_model = Model(inputs=image_input, outputs=[y1, y2, y3])
    return model, prediction_model, debug_model
