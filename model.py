import tensorflow as tf
import numpy as np
import tensorflow.keras.backend as K


def weighted_categorical_crossentropy(weights):
    weights = (K.variable(weights))

    def loss(y_true, y_pred):
        # scale predictions so that the class probas of each sample sum to 1
        y_pred /= (K.sum(y_pred, axis=-1, keepdims=True))

        # clip to prevent NaN's and Inf's
        y_pred = K.clip(y_pred, K.epsilon(), 1 - K.epsilon())

        # calc
        loss = y_true * K.log(y_pred) * weights
        loss = -K.sum(loss, -1)
        return loss

    return loss


def resnet_se_block(inputs, num_filters, kernel_size, strides, ratio):
    # 1D conv
    x = tf.keras.layers.Conv2D(filters=num_filters, kernel_size=kernel_size, strides=strides, padding='same',
                               kernel_initializer='he_normal')(inputs)
    x = tf.keras.layers.BatchNormalization()(x)
    x = tf.keras.layers.Activation('relu')(x)
    x = tf.keras.layers.Conv2D(filters=num_filters, kernel_size=kernel_size, strides=strides, padding='same',
                               kernel_initializer='he_normal')(x)
    x = tf.keras.layers.BatchNormalization()(x)

    # se block
    se = tf.keras.layers.Lambda(lambda x: K.mean(x, axis=-2, keepdims=True))(
        x)  # equal to tf.keras.layers.GlobalAveragePooling1D
    se = tf.keras.layers.Dense(units=num_filters // ratio)(se)
    se = tf.keras.layers.Activation('relu')(se)
    se = tf.keras.layers.Dense(units=num_filters)(se)
    se = tf.keras.layers.Activation('sigmoid')(se)
    x = tf.keras.layers.multiply([x, se])

    # skip connection
    x_skip = tf.keras.layers.Conv2D(filters=num_filters, kernel_size=1, strides=1, padding='same',
                                    kernel_initializer='he_normal')(inputs)
    x_skip = tf.keras.layers.BatchNormalization()(x_skip)

    x = tf.keras.layers.Add()([x, x_skip])
    x = tf.keras.layers.Activation('relu')(x)
    # x = tf.keras.layers.MaxPool2D(pool_size=(2,1), strides=(2,1), padding='same', data_format = 'channels_first')(x)

    return x


def create_model(Fs=100, n_classes=5, seq_length=15, summary=True):
    x_input = tf.keras.Input(shape=(seq_length, 30 * Fs, 1))  # (Batch_Size, seq_length, 3000, 1)
    # print(x_input.shape)
    x = resnet_se_block(x_input, 32, 3, 1, 4)  # (Batch_Size, seq_length, 3000, 32)
    x = tf.keras.layers.MaxPool2D(pool_size=(4, 1), strides=(4, 1), padding='same', data_format='channels_first')(
        x)  # (Batch_Size, seq_length, 750, 32)

    x = resnet_se_block(x, 64, 5, 1, 4)  # (Batch_Size, seq_length, 750, 64)
    x = tf.keras.layers.MaxPool2D(pool_size=(4, 1), strides=(4, 1), padding='same', data_format='channels_first')(
        x)  # (Batch_Size, seq_length, 188, 64)

    x = resnet_se_block(x, 128, 7, 1, 4)  # (Batch_Size, seq_length, 188, 128)
    x = tf.keras.layers.Lambda(lambda x: K.mean(x, axis=-2, keepdims=False))(
        x)  # (Batch_Size, seq_length, 128), equal to tf.keras.layers.GlobalAveragePool

    x = tf.keras.layers.Dropout(rate=0.5)(x)  # (Batch_Size, seq_length, 128)

    # LSTM
    x = tf.keras.layers.LSTM(units=64, dropout=0.5, activation='relu', return_sequences=True)(
        x)  # (Batch_Size, seq_length, 64)

    # Classify
    x_out = tf.keras.layers.Dense(units=n_classes, activation='softmax')(x)  # (Batch_Size, seq_length, 5)

    model = tf.keras.models.Model(x_input, x_out)
    model_loss = weighted_categorical_crossentropy(np.array([1, 1.5, 1, 1, 1]))  # np.array([1, 3, 1, 5, 3]), np.array([1.5, 2.5, 1, 1.5, 2.5])

    model.compile(optimizer='adam', loss=model_loss,
                  metrics=['accuracy'])

    if summary:
        model.summary()
        tf.keras.utils.plot_model(model, show_shapes=True, dpi=300, to_file='model.png')

    return model