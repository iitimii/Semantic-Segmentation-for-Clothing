import tensorflow as tf
import tensorflow.keras.layers as tfl

res = 224

def enc_conv(input_layer=None, filters=32, MaxPool=True, dropout=0):
    conv = tfl.Conv2D(filters, (3,3), activation='relu', padding='same', kernel_initializer='he_normal')(input_layer)
    conv = tfl.Conv2D(filters, (3,3), activation='relu', padding='same', kernel_initializer='he_normal')(conv)

    if dropout>0:
        conv = tfl.Dropout(dropout)(conv)

    if MaxPool:
        next_layer = tfl.MaxPool2D((2,2), strides=(2,2))(conv)
    else:
        next_layer = conv

    skip = conv
    return next_layer, skip


def dec_conv(dec_input, skip, filters=32):
    up = tfl.Conv2DTranspose(filters, (3,3), strides=(2,2), padding='same')(dec_input)

    merge = tfl.concatenate([up, skip], axis=3)

    conv = tfl.Conv2D(filters, (3,3), activation='relu', padding='same', kernel_initializer='he_normal')(merge)
    conv = tfl.Conv2D(filters, (3,3), activation='relu', padding='same', kernel_initializer='he_normal')(conv)
    return conv

def U_Net(input_size=(res, res, 3), filters=32, n_classes=32):
    Input = tf.keras.Input(input_size)
    cblock1 = enc_conv(Input, filters)
    cblock2 = enc_conv(cblock1[0], filters*2)
    cblock3 = enc_conv(cblock2[0], filters*4)
    cblock4 = enc_conv(cblock3[0], filters*8, dropout=0.2)
    cblock5 = enc_conv(cblock4[0], filters*16, dropout=0.2, MaxPool=False) 
    ublock6 = dec_conv(cblock5[0], cblock4[1], filters*8)
    ublock7 = dec_conv(ublock6, cblock3[1], filters*4)
    ublock8 = dec_conv(ublock7, cblock2[1], filters*2)
    ublock9 = dec_conv(ublock8, cblock1[1], filters)

    conv9 = tfl.Conv2D(filters, (3,3), activation='relu', padding='same', kernel_initializer='he_normal')(ublock9)
    Output = tfl.Conv2D(n_classes, (1,1), padding='same')(conv9)

    model = tf.keras.Model(Input, Output)

    return model
