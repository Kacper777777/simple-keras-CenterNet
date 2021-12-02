import tensorflow as tf
from tensorflow.keras.layers import Conv2D, BatchNormalization, ReLU
from tensorflow.keras.regularizers import l2
from losses import main_loss


def topk(hm, max_objects):
    b, h, w, c = tf.shape(hm)[0], tf.shape(hm)[1], tf.shape(hm)[2], tf.shape(hm)[3]
    hm = tf.reshape(hm, (b, -1))  # (b, h * w * c)
    scores, indices = tf.nn.top_k(hm, k=max_objects)  # (b, k), (b, k)
    class_ids = indices % c
    xs = indices // c % w
    ys = indices // c // w
    indices = ys * w + xs
    return scores, indices, class_ids, xs, ys


def decode(hm, wh, reg, max_objects):
    max_pool_2d = tf.keras.layers.MaxPooling2D(pool_size=(3, 3), strides=1, padding='same')
    maxpooled_hm = max_pool_2d(hm)
    difference = maxpooled_hm - hm
    mask = tf.cast(tf.equal(difference, 0), tf.float32)
    hm = hm * mask

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


def heatmap_header(backbone_output, num_filters, num_classes):
    y = Conv2D(num_filters, 1, kernel_initializer='he_normal', padding='same')(backbone_output)
    y = BatchNormalization()(y)
    y = ReLU()(y)
    y = Conv2D(num_classes, 1, kernel_initializer='he_normal', kernel_regularizer=l2(5e-4),
               activation='sigmoid')(y)
    return y


def regression_header(backbone_output, num_filters):
    y = Conv2D(num_filters, 1, kernel_initializer='he_normal', padding='same')(backbone_output)
    y = BatchNormalization()(y)
    y = ReLU()(y)
    y = Conv2D(2, 1, kernel_initializer='he_normal', kernel_regularizer=l2(5e-4))(y)
    return y
