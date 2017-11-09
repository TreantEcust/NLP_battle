import h5py
import numpy as np
import keras
import TextCNN
import TextRNN
import pandas as pd

content_matrix=pd.read_csv('data/word2vec/content_matrix.csv',header=None)
title_matrix=pd.read_csv('data/word2vec/title_matrix.csv',header=None)
train_data=h5py.File('data/train/train_data.h5','r')
train_title = train_data['train_title'].value
train_content = train_data['train_content'].value
train_label=train_data['train_label'].value

split_index=np.arange(train_label.shape[0])
np.random.shuffle(split_index)
train_index=split_index[0:int(len(split_index)*0.8)]
eval_index=split_index[int(len(split_index)*0.8):]

X_eval_title=train_title[eval_index,:]
X_eval_content=train_content[eval_index,:]
y_eval=train_label[eval_index]
X_train_title=train_title[train_index,:]
X_train_content=train_content[train_index,:]
y_train=train_label[train_index]
train_data.close()
NLP_model=TextCNN.TextCNN(title_matrix=title_matrix,content_matrix=content_matrix)
NLP_model.model.fit([X_train_title,X_train_content], y_train, batch_size=64, epochs=20, validation_data=([X_eval_title,X_eval_content],y_eval),
                    callbacks=[keras.callbacks.EarlyStopping(monitor='acc', patience=2, verbose=1, mode='auto'),
                    keras.callbacks.ModelCheckpoint('NLP_model_CNN.h5', monitor='acc', verbose=0, save_best_only=True,
                                                    save_weights_only=True, mode='auto', period=1)])

# test_data=h5py.File('data/eval/eval_data.h5','r')
# test_title = test_data['eval_title'].value
# test_content = test_data['eval_content'].value
# test_id=test_data['eval_id'].value
# test_id=np.array(list(map(lambda x:str(x,encoding = "utf-8"),test_id)))
# NLP_model=TextCNN.TextCNN(title_matrix=title_matrix,content_matrix=content_matrix)
# NLP_model.model.load_weights('NLP_model.h5')
# preds=NLP_model.model.predict([test_title,test_content])
# result=pd.DataFrame([test_id,preds],index=['id','pred'])
# result=result.transpose()
# result.loc[:,'result']='NEGATIVE'
# result.loc[(result['pred']>=0.5),'result']='POSITIVE'
# result.drop('pred',axis=1,inplace=True)
# result.to_csv('result.csv',index=False,header=False)