from keras.layers import *
from keras.models import *
from keras.optimizers import *
import KMaxPooling

class TextRCNN():
    def __init__(self, title_len=25,content_len=1500,seq_len=100,title_matrix=None,content_matrix=None):
        self.title_len=title_len
        self.content_len=content_len
        self.seq_len=seq_len
        self.title_matrix=title_matrix
        self.content_matrix=content_matrix
        self.model=self.RCNN_model()
        self.optimizer=Adam(lr=0.001, beta_1=0.9, beta_2=0.999, epsilon=1e-08)
        self.model.compile(loss='binary_crossentropy',optimizer=self.optimizer,metrics=['accuracy'])

    def RCNN_model(self):
        #title
        input_title = Input((self.title_len,))
        x_title_vec = Embedding(self.title_matrix.shape[0],self.seq_len,weights=[self.title_matrix],
                             input_length=self.title_len,mask_zero=False,trainable=False)(input_title)
        x_title = Bidirectional(GRU(64, return_sequences=True, implementation=2), merge_mode=None)(x_title_vec)
        x_title = Concatenate(axis=2)([x_title[0], x_title[1]])
        
        x_title1 = Conv1D(1, 4, strides=1, padding='valid', input_shape=(self.title_len, self.seq_len))(x_title)
        x_title1 = BatchNormalization()(x_title1)
        x_title1 = Activation('relu')(x_title1)
        x_title1 = KMaxPooling.KMaxPooling(k=2)(x_title1)
        
        x_title2 = Conv1D(1, 3, strides=1, padding='valid', input_shape=(self.title_len, self.seq_len))(x_title)
        x_title2 = BatchNormalization()(x_title2)
        x_title2 = Activation('relu')(x_title2)
        x_title2 = KMaxPooling.KMaxPooling(k=2)(x_title2)
        
        x_title3 = Conv1D(1, 2, strides=1, padding='valid', input_shape=(self.title_len, self.seq_len))(x_title)
        x_title3 = BatchNormalization()(x_title)
        x_title3 = Activation('relu')(x_title)
        x_title3 = KMaxPooling.KMaxPooling(k=2)(x_title)

        #content
        input_content = Input((self.content_len,))
        x_content_vec = Embedding(self.content_matrix.shape[0], self.seq_len, weights=[self.content_matrix],
                                input_length=self.content_len, mask_zero=False, trainable=False)(input_content)
        x_content = Bidirectional(GRU(64, return_sequences=True,implementation=2), merge_mode=None)(x_content_vec)
        x_content = Concatenate(axis=2)([x_content[0], x_content[1]])
        
        x_content1 = Conv1D(1, 5, strides=1, padding='valid', input_shape=(self.content_len, self.seq_len))(x_content)
        x_content1 = BatchNormalization()(x_content1)
        x_content1 = Activation('relu')(x_content1)
        x_content1 = KMaxPooling.KMaxPooling(k=2)(x_content1)
        
        x_content2 = Conv1D(1, 4, strides=1, padding='valid', input_shape=(self.content_len, self.seq_len))(x_content)
        x_content2 = BatchNormalization()(x_content2)
        x_content2 = Activation('relu')(x_content2)
        x_content2 = KMaxPooling.KMaxPooling(k=2)(x_content2)
        
        x_content3 = Conv1D(1, 3, strides=1, padding='valid', input_shape=(self.content_len, self.seq_len))(x_content)
        x_content3 = BatchNormalization()(x_content3)
        x_content3 = Activation('relu')(x_content3)
        x_content3 = KMaxPooling.KMaxPooling(k=2)(x_content3)

        x_content4 = Conv1D(1, 2, strides=1, padding='valid', input_shape=(self.content_len, self.seq_len))(x_content)
        x_content4 = BatchNormalization()(x_content4)
        x_content4 = Activation('relu')(x_content4)
        x_content4 = KMaxPooling.KMaxPooling(k=2)(x_content4)

        x = Concatenate()([x_title1,x_title2,x_title3,x_content1,x_content2,x_content3,x_content4])
        x = Dropout(0.25)(x)
        output = Dense(1, activation='sigmoid')(x)
        return Model(inputs=[input_title,input_content], outputs=output, name='Discriminator')