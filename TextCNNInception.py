from keras.layers import *
from keras.models import *
from keras.optimizers import *
import KMaxPooling as KP


class TextCNN1():
    def __init__(self, title_len=25, content_len=1500, seq_len=100, title_matrix=None, content_matrix=None):
        self.title_len = title_len
        self.content_len = content_len
        self.seq_len = seq_len
        self.title_matrix = title_matrix
        self.content_matrix = content_matrix
        self.model = self.CNN1_model()
        self.optimizer = Adadelta(lr=1.0, rho=0.95, epsilon=1e-06)
        self.model.compile(loss='binary_crossentropy', optimizer=self.optimizer, metrics=['accuracy'])

    def CNN1_model(self):
        # title
        input_title = Input((self.title_len,))
        x_title = Embedding(self.title_matrix.shape[0], self.seq_len, weights=[self.title_matrix],
                            input_length=self.title_len, mask_zero=False, trainable=False)(input_title)

        x_title1 = Conv1D(64, 3, strides=1, padding='valid', input_shape=(self.title_len, self.seq_len))(x_title)
        x_title1 = BatchNormalization()(x_title1)
        x_title1 = Activation('relu')(x_title1)
        x_title1 = GlobalMaxPooling1D()(x_title1)

        x_title2 = Conv1D(64, 2, strides=1, padding='valid', input_shape=(self.title_len, self.seq_len))(x_title)
        x_title2 = BatchNormalization()(x_title2)
        x_title2 = Activation('relu')(x_title2)
        x_title2 = GlobalMaxPooling1D()(x_title2)

        x_title3 = Conv1D(64, 1, strides=1, padding='valid', input_shape=(self.title_len, self.seq_len))(x_title)
        x_title3 = BatchNormalization()(x_title3)
        x_title3 = Activation('relu')(x_title3)
        x_title3 = GlobalMaxPooling1D()(x_title3)

        # content
        input_content = Input((self.content_len,))
        x_content = Embedding(self.content_matrix.shape[0], self.seq_len, weights=[self.content_matrix],
                              input_length=self.content_len, mask_zero=False, trainable=False)(input_content)

        x_content1 = Conv1D(128, 4, strides=1, padding='valid', input_shape=(self.content_len, self.seq_len))(x_content)
        x_content1 = BatchNormalization()(x_content1)
        x_content1 = Activation('relu')(x_content1)
        x_content1 = GlobalMaxPooling1D()(x_content1)

        x_content2 = Conv1D(128, 3, strides=1, padding='valid', input_shape=(self.content_len, self.seq_len))(x_content)
        x_content2 = BatchNormalization()(x_content2)
        x_content2 = Activation('relu')(x_content2)
        x_content2 = Conv1D(128, 5, strides=1, padding='valid', input_shape=(self.content_len, self.seq_len))(x_content2)
        x_content2 = BatchNormalization()(x_content2)
        x_content2 = Activation('relu')(x_content2)
        x_content2 = GlobalMaxPooling1D()(x_content2)

        x_content3 = Conv1D(128, 2, strides=1, padding='valid', input_shape=(self.content_len, self.seq_len))(x_content)
        x_content3 = BatchNormalization()(x_content3)
        x_content3 = Activation('relu')(x_content3)
        x_content3 = GlobalMaxPooling1D()(x_content3)

        x_content4 = Conv1D(128, 1, strides=1, padding='valid', input_shape=(self.content_len, self.seq_len))(x_content)
        x_content4 = BatchNormalization()(x_content4)
        x_content4 = Activation('relu')(x_content4)
        x_content4 = Conv1D(128, 3, strides=1, padding='valid', input_shape=(self.content_len, self.seq_len))(x_content4)
        x_content4 = BatchNormalization()(x_content4)
        x_content4 = Activation('relu')(x_content4)
        x_content4 = GlobalMaxPooling1D()(x_content4)

        x = Concatenate()([x_title1, x_title2, x_title3, x_content1, x_content2, x_content3, x_content4])
        x = Dropout(0.25)(x)
        output = Dense(1, activation='sigmoid')(x)
        return Model(inputs=[input_title, input_content], outputs=output, name='Discriminator')