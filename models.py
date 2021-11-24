import tensorflow as tf
from tensorflow.keras.layers import Conv2D, MaxPooling2D, AveragePooling2D, \
    Flatten, Dense, Reshape, Concatenate, Add, Input, Lambda, BatchNormalization, \
    ReLU, Conv2DTranspose, UpSampling2D, Dropout
from tensorflow.keras.models import Model
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


def miniconvnet(image_shape=(64, 64, 1), num_classes=1, max_objects=1):
    # input image
    image_input = tf.keras.Input(shape=image_shape)

    x = Conv2D(16, kernel_size=(3, 3), strides=1,
               padding='same', activation='relu')(image_input)
    x = MaxPooling2D(pool_size=(3, 3), strides=2, padding='same')(x)
    x = BatchNormalization()(x)

    x = Conv2D(32, kernel_size=(3, 3), strides=1,
               padding='same', activation='relu')(x)
    x = MaxPooling2D(pool_size=(3, 3), strides=2, padding='same')(x)
    x = BatchNormalization()(x)

    backbone_output = ReLU()(BatchNormalization()(x))

    x = Conv2D(48, kernel_size=(3, 3), strides=1,
               padding='same', activation='relu')(x)
    x = MaxPooling2D(pool_size=(3, 3), strides=2, padding='same')(x)
    x = BatchNormalization()(x)

    # code
    x = Conv2D(64, kernel_size=(3, 3), strides=1,
               padding='same', activation='relu')(x)
    x = MaxPooling2D(pool_size=(3, 3), strides=2, padding='same')(x)
    encoded = x

    x = Conv2DTranspose(48, kernel_size=(3, 3), strides=1, padding='same', activation='relu')(encoded)
    x = UpSampling2D((2, 2))(x)
    x = BatchNormalization()(x)

    x = Conv2DTranspose(32, kernel_size=(3, 3), strides=1, padding='same', activation='relu')(x)
    x = UpSampling2D((2, 2))(x)
    x = BatchNormalization()(x)

    x = Conv2DTranspose(16, kernel_size=(3, 3), strides=1, padding='same', activation='relu')(x)
    x = UpSampling2D((2, 2))(x)
    x = BatchNormalization()(x)

    # decoded image
    x = Conv2DTranspose(1, kernel_size=(3, 3), strides=1, padding='same', activation='relu')(x)
    x = UpSampling2D((2, 2))(x)
    x = BatchNormalization()(x)
    decoded = x

    #backbone_output = ReLU()(BatchNormalization()(encoded))

    downsample_factor = 4
    hm_input = Input(shape=(image_shape[0] // downsample_factor, image_shape[1] // downsample_factor, num_classes))
    wh_input = Input(shape=(max_objects, 2))
    reg_input = Input(shape=(max_objects, 2))
    reg_mask_input = Input(shape=(max_objects,))
    index_input = Input(shape=(max_objects,))

    # hm header
    y1 = heatmap_header(backbone_output=backbone_output, num_filters=32, num_classes=num_classes)

    # wh header
    y2 = regression_header(backbone_output=backbone_output, num_filters=32)

    # reg header
    y3 = regression_header(backbone_output=backbone_output, num_filters=32)

    autoencoder_model = Model(inputs=image_input, outputs=decoded)
    loss_ = Lambda(main_loss, name='centernet_loss')(
        [y1, y2, y3, hm_input, wh_input, reg_input, reg_mask_input, index_input])
    model = Model(inputs=[image_input, hm_input, wh_input, reg_input,
                          reg_mask_input, index_input], outputs=[loss_])
    detections = Lambda(lambda z: decode(*z, max_objects=max_objects))([y1, y2, y3])
    prediction_model = Model(inputs=image_input, outputs=detections)
    debug_model = Model(inputs=image_input, outputs=[y1, y2, y3])

    return autoencoder_model, model, prediction_model, debug_model


def googlenet(image_shape=(224, 224, 3), num_classes=1, max_objects=1):
    def inception(x, filters):
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

    def auxiliary(x, name=None):
        layer = AveragePooling2D(pool_size=(5, 5), strides=3, padding='valid')(x)
        layer = Conv2D(filters=128, kernel_size=(1, 1), strides=1, padding='same', activation='relu')(layer)
        layer = Flatten()(layer)
        layer = Dense(units=256, activation='relu')(layer)
        layer = Dropout(0.4)(layer)
        layer = Dense(units=1000, activation='softmax', name=name)(layer)
        return layer

    # input image
    image_input = tf.keras.Input(shape=image_shape)

    # stage-1
    layer = Conv2D(filters=64, kernel_size=(7, 7), strides=2, padding='same', activation='relu')(image_input)
    layer = MaxPooling2D(pool_size=(3, 3), strides=2, padding='same')(layer)
    layer = BatchNormalization()(layer)

    # stage-2
    layer = Conv2D(filters=64, kernel_size=(1, 1), strides=1, padding='same', activation='relu')(layer)
    layer = Conv2D(filters=192, kernel_size=(3, 3), strides=1, padding='same', activation='relu')(layer)
    layer = BatchNormalization()(layer)
    layer = MaxPooling2D(pool_size=(3, 3), strides=2, padding='same')(layer)

    # stage-3
    layer = inception(layer, [64, (96, 128), (16, 32), 32])  # 3a
    layer = inception(layer, [128, (128, 192), (32, 96), 64])  # 3b
    layer = MaxPooling2D(pool_size=(3, 3), strides=2, padding='same')(layer)

    # stage-4
    layer = inception(layer, [192, (96, 208), (16, 48), 64])  # 4a
    aux1 = auxiliary(layer, name='aux1')
    layer = inception(layer, [160, (112, 224), (24, 64), 64])  # 4b
    layer = inception(layer, [128, (128, 256), (24, 64), 64])  # 4c
    layer = inception(layer, [112, (144, 288), (32, 64), 64])  # 4d
    aux2 = auxiliary(layer, name='aux2')
    layer = inception(layer, [256, (160, 320), (32, 128), 128])  # 4e
    layer = MaxPooling2D(pool_size=(3, 3), strides=2, padding='same')(layer)

    # stage-5
    layer = inception(layer, [256, (160, 320), (32, 128), 128])  # 5a
    layer = inception(layer, [384, (192, 384), (48, 128), 128])  # 5b

    before_average_pooling = layer

    layer = AveragePooling2D(pool_size=(7, 7), strides=1, padding='valid')(layer)

    # stage-6
    # layer = Flatten()(layer)
    # layer = Dropout(0.4)(layer)
    # layer = Dense(units=256, activation='linear')(layer)
    # main = Dense(units=1000, activation='softmax', name='main')(layer)

    #original_model = Model(inputs=image_input, outputs=[main, aux1, aux2])
    #model_without_auxiliary = Model(inputs=image_input, outputs=main)

    x = Conv2DTranspose(256, kernel_size=(3, 3), strides=1, padding='same', activation='relu')(before_average_pooling)
    x = UpSampling2D((2, 2))(x)
    x = BatchNormalization()(x)

    x = Conv2DTranspose(192, kernel_size=(3, 3), strides=1, padding='same', activation='relu')(x)
    x = UpSampling2D((2, 2))(x)
    x = BatchNormalization()(x)

    x = Conv2DTranspose(128, kernel_size=(3, 3), strides=1, padding='same', activation='relu')(x)
    x = UpSampling2D((2, 2))(x)
    x = BatchNormalization()(x)

    backbone_output = ReLU()(x)

    x = Conv2DTranspose(64, kernel_size=(3, 3), strides=1, padding='same', activation='relu')(x)
    x = UpSampling2D((2, 2))(x)
    x = BatchNormalization()(x)

    # decoded image
    x = Conv2DTranspose(1, kernel_size=(3, 3), strides=1, padding='same', activation='relu')(x)
    x = UpSampling2D((2, 2))(x)
    x = BatchNormalization()(x)
    decoded = x

    # backbone_output = ReLU()(BatchNormalization()(encoded))

    downsample_factor = 4
    hm_input = Input(shape=(image_shape[0] // downsample_factor, image_shape[1] // downsample_factor, num_classes))
    wh_input = Input(shape=(max_objects, 2))
    reg_input = Input(shape=(max_objects, 2))
    reg_mask_input = Input(shape=(max_objects,))
    index_input = Input(shape=(max_objects,))

    # hm header
    y1 = heatmap_header(backbone_output=backbone_output, num_filters=32, num_classes=num_classes)

    # wh header
    y2 = regression_header(backbone_output=backbone_output, num_filters=32)

    # reg header
    y3 = regression_header(backbone_output=backbone_output, num_filters=32)

    autoencoder_model = Model(inputs=image_input, outputs=decoded)
    loss_ = Lambda(main_loss, name='centernet_loss')(
        [y1, y2, y3, hm_input, wh_input, reg_input, reg_mask_input, index_input])
    model = Model(inputs=[image_input, hm_input, wh_input, reg_input,
                          reg_mask_input, index_input], outputs=[loss_])
    detections = Lambda(lambda z: decode(*z, max_objects=max_objects))([y1, y2, y3])
    prediction_model = Model(inputs=image_input, outputs=detections)
    debug_model = Model(inputs=image_input, outputs=[y1, y2, y3])

    return autoencoder_model, model, prediction_model, debug_model


if __name__ == '__main__':
    #autoencoder_model, model, prediction_model, debug_model = miniconvnet_64(image_shape=(64, 64, 1))
    autoencoder_model, model, prediction_model, debug_model = googlenet()
    model.summary()
