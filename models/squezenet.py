import tensorflow as tf
from tensorflow.keras import backend as K
from tensorflow.keras.layers import Input, Convolution2D, Conv2D, MaxPooling2D, AveragePooling2D, \
    GlobalAveragePooling2D, Activation, ReLU, concatenate, Concatenate, UpSampling2D, \
    Dropout, BatchNormalization, Lambda
from tensorflow.keras.models import Model
from models.centernet_utils import topk, decode, heatmap_header, regression_header, main_loss

sq1x1 = "squeeze1x1"
exp1x1 = "expand1x1"
exp3x3 = "expand3x3"
relu = "relu_"


# Modular function for Fire Node.
def fire_module(x, fire_id, squeeze=16, expand=64):
    s_id = 'fire' + str(fire_id) + '/'

    if K.image_data_format() == 'channels_first':
        channel_axis = 1
    else:
        channel_axis = 3

    x = Convolution2D(squeeze, (1, 1), padding='valid', name=s_id + sq1x1)(x)
    x = Activation('relu', name=s_id + relu + sq1x1)(x)

    left = Convolution2D(expand, (1, 1), padding='valid', name=s_id + exp1x1)(x)
    left = Activation('relu', name=s_id + relu + exp1x1)(left)

    right = Convolution2D(expand, (3, 3), padding='same', name=s_id + exp3x3)(x)
    right = Activation('relu', name=s_id + relu + exp3x3)(right)

    x = concatenate([left, right], axis=channel_axis, name=s_id + 'concat')
    return x


def squeezenet(image_shape=(256, 256, 3),
               use_bn_on_input=False,
               first_stride=2):
    raw_image_input = tf.keras.Input(shape=image_shape)
    if use_bn_on_input:
        image_input = BatchNormalization()(raw_image_input)
    else:
        image_input = raw_image_input

    x = Convolution2D(64, (3, 3), strides=(first_stride, first_stride), padding='same', name='conv1')(image_input)
    x = Activation('relu', name='relu_conv1')(x)
    x = MaxPooling2D(pool_size=(3, 3), strides=(2, 2), padding='same', name='pool1')(x)

    x = fire_module(x, fire_id=2, squeeze=16, expand=64)
    x = fire_module(x, fire_id=3, squeeze=16, expand=64)
    x = MaxPooling2D(pool_size=(3, 3), strides=(2, 2), padding='same', name='pool3')(x)

    x = fire_module(x, fire_id=4, squeeze=32, expand=128)
    x = fire_module(x, fire_id=5, squeeze=32, expand=128)
    x = MaxPooling2D(pool_size=(3, 3), strides=(2, 2), padding='same', name='pool5')(x)

    x = fire_module(x, fire_id=6, squeeze=48, expand=192)
    x = fire_module(x, fire_id=7, squeeze=48, expand=192)
    x = fire_module(x, fire_id=8, squeeze=64, expand=256)
    x = fire_module(x, fire_id=9, squeeze=64, expand=256)
    x = Dropout(0.5, name='drop9')(x)

    x = Convolution2D(1000, (1, 1), padding='valid', name='conv10')(x)
    x = Activation('relu', name='relu_conv10')(x)
    x = GlobalAveragePooling2D()(x)
    out = Activation('softmax', name='loss')(x)

    model = Model(image_input, out, name='squeezenet')
    return model


def squeezenet_decoder(encoded_image_shape):
    # encoded image
    encoded_image_input = tf.keras.Input(shape=encoded_image_shape[1:])

    x = UpSampling2D((2, 2))(encoded_image_input)
    x = Conv2D(128, kernel_size=(3, 3), strides=1, padding='same', activation='relu')(x)
    x = BatchNormalization()(x)

    x = UpSampling2D((2, 2))(x)
    x = Conv2D(64, kernel_size=(3, 3), strides=1, padding='same', activation='relu')(x)
    x = BatchNormalization()(x)

    x = UpSampling2D((2, 2))(x)
    x = Conv2D(32, kernel_size=(3, 3), strides=1, padding='same', activation='relu')(x)
    x = BatchNormalization()(x)

    x = UpSampling2D((2, 2))(x)
    x = Conv2D(3, kernel_size=(3, 3), strides=1, padding='same', activation='relu')(x)
    x = BatchNormalization()(x)

    return Model(inputs=encoded_image_input, outputs=x)


def squeezenet_autoencoder(image_shape=(256, 256, 3)):
    # input image
    image_input = tf.keras.Input(shape=image_shape)

    # create encoder
    squeezenet_original = squeezenet(image_shape=image_shape)
    encoder = Model(inputs=squeezenet_original.inputs,
                    outputs=squeezenet_original.get_layer('drop9').output)

    # calculate encoder output
    encoder_out = encoder(image_input)

    # create decoder
    decoder = squeezenet_decoder(encoded_image_shape=encoder.output_shape)

    # calculate decoder output
    decoder_out = decoder(encoder_out)

    model = Model(inputs=image_input, outputs=decoder_out)
    return model


def squeezenet_centernet(image_shape=(256, 256, 3), num_classes=1, max_objects=1):
    # input image
    image_input = tf.keras.Input(shape=image_shape)

    # create encoder
    squeezenet_original = squeezenet(image_shape=image_shape)
    encoder = Model(inputs=squeezenet_original.inputs,
                    outputs=squeezenet_original.get_layer('drop9').output)

    # calculate encoder output
    encoder_out = encoder(image_input)

    x = Conv2D(256, kernel_size=(1, 1), strides=1, padding='same', activation='relu')(encoder_out)

    x = UpSampling2D((2, 2))(x)
    x = Conv2D(256, kernel_size=(3, 3), strides=1, padding='same', activation='relu')(x)
    x = BatchNormalization()(x)

    x = UpSampling2D((2, 2))(x)
    x = Conv2D(128, kernel_size=(3, 3), strides=1, padding='same', activation='relu')(x)
    x = BatchNormalization()(x)

    backbone_output = x

    downsample_factor = 4
    hm_input = Input(shape=(image_shape[0] // downsample_factor, image_shape[1] // downsample_factor, num_classes))
    wh_input = Input(shape=(max_objects, 2))
    reg_input = Input(shape=(max_objects, 2))
    reg_mask_input = Input(shape=(max_objects,))
    index_input = Input(shape=(max_objects,))

    # hm header
    y1 = heatmap_header(backbone_output=backbone_output, num_filters=64, num_classes=num_classes)

    # wh header
    y2 = regression_header(backbone_output=backbone_output, num_filters=64)

    # reg header
    y3 = regression_header(backbone_output=backbone_output, num_filters=64)

    loss_ = Lambda(main_loss, name='centernet_loss')(
        [y1, y2, y3, hm_input, wh_input, reg_input, reg_mask_input, index_input])
    model = Model(inputs=[image_input, hm_input, wh_input, reg_input,
                          reg_mask_input, index_input], outputs=[loss_])
    detections = Lambda(lambda z: decode(*z, max_objects=max_objects))([y1, y2, y3])
    prediction_model = Model(inputs=image_input, outputs=detections)
    debug_model = Model(inputs=image_input, outputs=[y1, y2, y3])

    return model, prediction_model, debug_model


if __name__ == '__main__':
    snet = squeezenet(image_shape=(256, 256, 3))
    snet_autoencoder = squeezenet_autoencoder(image_shape=(256, 256, 3))
    snet_centernet = squeezenet_centernet(image_shape=(256, 256, 3), num_classes=1, max_objects=1)
    snet.summary()
    snet_autoencoder.summary()
    snet_centernet[0].summary()
