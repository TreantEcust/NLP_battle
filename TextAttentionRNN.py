from keras.layers import *
from keras.models import *
from keras.optimizers import *

class TextAttentionRNN():
    def __init__(self, title_len=25,content_len=1500,seq_len=100,title_matrix=None,content_matrix=None):
        self.title_len=title_len
        self.content_len=content_len
        self.seq_len=seq_len
        self.title_matrix=title_matrix
        self.content_matrix=content_matrix
        self.model=self.Attention_RNN_model()
        self.optimizer=Adadelta(lr=1.0, rho=0.95, epsilon=1e-06)
        self.model.compile(loss='binary_crossentropy',optimizer=self.optimizer,metrics=['accuracy'])

    def attention_3d_block(self,inputs,SINGLE_ATTENTION_VECTOR=False):
        # inputs.shape = (batch_size, time_steps, input_dim)
        input_dim = int(inputs.shape[2])
        TIME_STEPS=int(inputs.shape[1])
        a = Permute((2, 1))(inputs)
        a = Reshape((input_dim, TIME_STEPS))(a)  # this line is not useful. It's just to know which dimension is what.
        a = Dense(TIME_STEPS, activation='softmax')(a)
        if SINGLE_ATTENTION_VECTOR:
            a = Lambda(lambda x: K.mean(x, axis=1))(a)
            a = RepeatVector(input_dim)(a)
        a_probs = Permute((2, 1))(a)
        output_attention_mul = merge([inputs, a_probs], mode='mul')
        return output_attention_mul

    def Attention_RNN_model(self):
        #title
        input_title = Input((self.title_len,))
        x_title = Embedding(self.title_matrix.shape[0],self.seq_len,weights=[self.title_matrix],
                             input_length=self.title_len,mask_zero=False,trainable=False)(input_title)
        print(x_title.shape)
        x_title = self.attention_3d_block(x_title)
        x_title = Bidirectional(GRU(256, return_sequences=False,implementation=2), merge_mode='concat')(x_title)

        #content
        input_content = Input((self.content_len,))
        x_content = Embedding(self.content_matrix.shape[0], self.seq_len, weights=[self.content_matrix],
                                input_length=self.content_len, mask_zero=False, trainable=False)(input_content)
        x_content = self.attention_3d_block(x_content)
        x_content = Bidirectional(GRU(256, return_sequences=False,implementation=2), merge_mode='concat')(x_content)

        x = Concatenate()([x_title,x_content])
        x = Dropout(0.25)(x)
        output = Dense(1, activation='sigmoid')(x)
        return Model(inputs=[input_title,input_content], outputs=output, name='Discriminator')