import tensorflow as tf
from tensorflow.keras.layers import Conv2D, MaxPooling2D, AveragePooling2D, \
    Flatten, Dense, Reshape, Concatenate, Input, Lambda, BatchNormalization
from tensorflow.keras.models import Model
from losses import main_loss


def topk(hm, max_objects=10):
    b, h, w, c = tf.shape(hm)[0], tf.shape(hm)[1], tf.shape(hm)[2], tf.shape(hm)[3]
    hm = tf.reshape(hm, (b, -1))  # (b, h * w * c)
    scores, indices = tf.nn.top_k(hm, k=max_objects)  # (b, k), (b, k)
    class_ids = indices % c
    xs = indices // c % w
    ys = indices // c // w
    indices = ys * w + xs
    return scores, indices, class_ids, xs, ys


def decode(hm, wh, reg, max_objects=10):
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


def centernet(num_classes, backbone, output_shape, max_objects):

    hm_input = Input(shape=(output_shape[0], output_shape[1], num_classes))
    wh_input = Input(shape=(max_objects, 2))
    reg_input = Input(shape=(max_objects, 2))
    reg_mask_input = Input(shape=(max_objects,))
    index_input = Input(shape=(max_objects,))

    backbone_output = backbone.outputs[-1]
    backbone_output = Flatten()(backbone_output)
    x = BatchNormalization()(backbone_output)

    x = Dense(output_shape[0] * output_shape[1] * 1)(x)
    x = Reshape((output_shape[0], output_shape[1], 1))(x)

    # hm header
    y1 = Conv2D(num_classes, 1, kernel_initializer=tf.keras.initializers.Constant(0.01),
                activation='sigmoid')(x)
    # y1 = MaxPooling2D(pool_size=(3, 3), strides=1, padding='same')(y1)

    # wh header
    y2 = Conv2D(2, 1, kernel_initializer=tf.keras.initializers.Constant(0.0))(x)

    # reg header
    y3 = Conv2D(2, 1, kernel_initializer=tf.keras.initializers.Constant(0.0))(x)

    loss_ = Lambda(main_loss, name='centernet_loss')(
        [y1, y2, y3, hm_input, wh_input, reg_input, reg_mask_input, index_input])
    model = Model(inputs=[backbone.inputs, hm_input, wh_input, reg_input,
                          reg_mask_input, index_input], outputs=[loss_])

    detections = Lambda(lambda z: decode(*z))([y1, y2, y3])
    prediction_model = Model(inputs=backbone.inputs, outputs=detections)
    debug_model = Model(inputs=backbone.inputs, outputs=[y1, y2, y3])
    return model, prediction_model, debug_model
