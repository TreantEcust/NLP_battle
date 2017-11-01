import h5py
import numpy as np
import os
import TextCNN
import pandas as pd
import pickle

# title_len=0
# num=0
# content_len=0
# path='data/train/'
# for d in os.listdir(path):
#     file = h5py.File(path+d, 'r')
#     num+=1
#     print(num)
#     title = file['title']
#     content = file['content']
#     label=file['label']
#     title_len+=title.shape[0]
#     content_len+=content.shape[0]
#     file.close()
# print('avg title:'+str(title_len/num))
# print('avg content:'+str(content_len/num))

NLP_model=TextCNN.TextCNN()
NLP_model.fit('data/train',batch_size=64,eval=0.1,patience=2)
NLP_model.model.save_weights('NLP_model.h5')

# NLP_model.model.load_weights('NLP_model.h5')
# ids,preds=NLP_model.predict('data/eval')
# result=np.concatenate((ids,preds),axis=1)
# result=pd.DataFrame(result,columns=['id','pred'])
# result['pred']=result['pred'].astype('float')
# result.loc[:,'result']='NEGATIVE'
# result.loc[(result['pred']>=0.5),'result']='POSITIVE'
# result.drop('pred',axis=1,inplace=True)
# result.to_csv('result.csv',index=False,header=False)