import tensorflow as tf
from tensorflow.keras.layers import Conv2D, MaxPooling2D, AveragePooling2D, \
    Flatten, Dense, Reshape, Concatenate, Add, Input, Lambda, BatchNormalization
from tensorflow.keras.models import Model
from tensorflow_core.python.keras.layers import ReLU
from tensorflow_core.python.keras.regularizers import l2
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


def small_convnet(input_shape=(64, 64, 1), num_classes=1, max_objects=50):
    # input image
    image_input = tf.keras.Input(shape=input_shape)

    x = Conv2D(32, kernel_size=(3, 3), strides=1,
               padding='same', activation='relu')(image_input)

    x = Conv2D(64, kernel_size=(3, 3), strides=1,
               padding='same', activation='relu')(x)

    x = Conv2D(96, kernel_size=(3, 3), strides=1,
               padding='same', activation='relu')(x)

    x = MaxPooling2D(pool_size=(3, 3), strides=2, padding='same')(x)

    x = Conv2D(128, kernel_size=(3, 3), strides=1,
               padding='same', activation='relu')(x)

    x = Conv2D(192, kernel_size=(3, 3), strides=1,
               padding='same', activation='relu')(x)

    x = Conv2D(256, kernel_size=(3, 3), strides=1,
               padding='same', activation='relu')(x)

    x = MaxPooling2D(pool_size=(3, 3), strides=2, padding='same')(x)

    backbone_output = BatchNormalization()(x)
    backbone_output = ReLU()(backbone_output)

    hm_input = Input(shape=(input_shape[0] / 4, input_shape[1] / 4, num_classes))
    wh_input = Input(shape=(max_objects, 2))
    reg_input = Input(shape=(max_objects, 2))
    reg_mask_input = Input(shape=(max_objects,))
    index_input = Input(shape=(max_objects,))

    # hm header
    y1 = Conv2D(64, 1, kernel_initializer='he_normal', padding='same')(backbone_output)
    y1 = BatchNormalization()(y1)
    y1 = ReLU()(y1)
    y1 = Conv2D(num_classes, 1, kernel_initializer='he_normal', kernel_regularizer=l2(5e-4),
                activation='sigmoid')(y1)

    # wh header
    y2 = Conv2D(64, 1, kernel_initializer='he_normal', padding='same')(backbone_output)
    y2 = BatchNormalization()(y2)
    y2 = ReLU()(y2)
    y2 = Conv2D(2, 1, kernel_initializer='he_normal', kernel_regularizer=l2(5e-4))(y2)

    # reg header
    y3 = Conv2D(64, 1, kernel_initializer='he_normal', padding='same')(backbone_output)
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


def average_convnet(input_shape=(256, 256, 3), num_classes=1, max_objects=50):
    # input image
    image_input = tf.keras.Input(shape=input_shape)

    x = Conv2D(32, kernel_size=(3, 3), strides=1,
               padding='same', activation='relu')(image_input)

    x = Conv2D(48, kernel_size=(3, 3), strides=1,
               padding='same', activation='relu')(image_input)

    x = Conv2D(64, kernel_size=(3, 3), strides=1,
               padding='same', activation='relu')(x)

    x = Conv2D(80, kernel_size=(3, 3), strides=1,
               padding='same', activation='relu')(x)

    x = Conv2D(96, kernel_size=(3, 3), strides=1,
               padding='same', activation='relu')(x)

    x = Conv2D(112, kernel_size=(3, 3), strides=1,
               padding='same', activation='relu')(x)

    x = MaxPooling2D(pool_size=(3, 3), strides=2, padding='same')(x)

    x = Conv2D(128, kernel_size=(3, 3), strides=1,
               padding='same', activation='relu')(x)

    x = Conv2D(144, kernel_size=(3, 3), strides=1,
               padding='same', activation='relu')(x)

    x = Conv2D(160, kernel_size=(3, 3), strides=1,
               padding='same', activation='relu')(x)

    x = Conv2D(176, kernel_size=(3, 3), strides=1,
               padding='same', activation='relu')(x)

    x = Conv2D(192, kernel_size=(3, 3), strides=1,
               padding='same', activation='relu')(x)

    x = Conv2D(256, kernel_size=(3, 3), strides=1,
               padding='same', activation='relu')(x)

    x = MaxPooling2D(pool_size=(3, 3), strides=2, padding='same')(x)

    backbone_output = BatchNormalization()(x)
    backbone_output = ReLU()(backbone_output)

    hm_input = Input(shape=(input_shape[0] / 4, input_shape[1] / 4, num_classes))
    wh_input = Input(shape=(max_objects, 2))
    reg_input = Input(shape=(max_objects, 2))
    reg_mask_input = Input(shape=(max_objects,))
    index_input = Input(shape=(max_objects,))

    # hm header
    y1 = Conv2D(64, 1, kernel_initializer='he_normal', padding='same')(backbone_output)
    y1 = BatchNormalization()(y1)
    y1 = ReLU()(y1)
    y1 = Conv2D(num_classes, 1, kernel_initializer='he_normal', kernel_regularizer=l2(5e-4),
                activation='sigmoid')(y1)

    # wh header
    y2 = Conv2D(64, 1, kernel_initializer='he_normal', padding='same')(backbone_output)
    y2 = BatchNormalization()(y2)
    y2 = ReLU()(y2)
    y2 = Conv2D(2, 1, kernel_initializer='he_normal', kernel_regularizer=l2(5e-4))(y2)

    # reg header
    y3 = Conv2D(64, 1, kernel_initializer='he_normal', padding='same')(backbone_output)
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


if __name__ == '__main__':
    model, prediction_model, debug_model = small_convnet()
    model.summary()
