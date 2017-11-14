from keras.layers import *
from keras.models import *
from keras.optimizers import *

class TextRNN_LSTM():
    def __init__(self, title_len=25,content_len=1500,seq_len=100,title_matrix=None,content_matrix=None):
        self.title_len=title_len
        self.content_len=content_len
        self.seq_len=seq_len
        self.title_matrix=title_matrix
        self.content_matrix=content_matrix
        self.model=self.RNN_LSTM_model()
        self.optimizer=Adam(lr=0.001, beta_1=0.9, beta_2=0.999, epsilon=1e-08)
        self.model.compile(loss='binary_crossentropy',optimizer=self.optimizer,metrics=['accuracy'])

    def RNN_LSTM_model(self):
        #title
        input_title = Input((self.title_len,))
        x_title = Embedding(self.title_matrix.shape[0],self.seq_len,weights=[self.title_matrix],
                             input_length=self.title_len,mask_zero=False,trainable=False)(input_title)
        x_title = Bidirectional(LSTM(256, return_sequences=False), merge_mode='concat')(x_title)

        #content
        input_content = Input((self.content_len,))
        x_content = Embedding(self.content_matrix.shape[0], self.seq_len, weights=[self.content_matrix],
                                input_length=self.content_len, mask_zero=False, trainable=False)(input_content)
        x_content = Bidirectional(LSTM(256, return_sequences=False), merge_mode='concat')(x_content)

        x = Concatenate()([x_title,x_content])
        x = Dropout(0.25)(x)
        output = Dense(1, activation='sigmoid')(x)
        return Model(inputs=[input_title,input_content], outputs=output, name='Discriminator')