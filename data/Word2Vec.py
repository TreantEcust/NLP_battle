import h5py
import gc
import pickle
from gensim.models import Word2Vec
import numpy as np
import pandas as pd
from tqdm import tqdm

def load_word(word_file):
    print('Loading word from file [%s]' % word_file)
    fp = open(word_file, 'rb')
    dataset = pickle.load(fp)
    fp.close()

    trn_word = dataset['trn']
    evl_word = dataset['evl']
    return [trn_word, evl_word]

word_file = 'vector/word_data.pkl'
trn_word, evl_word = load_word(word_file=word_file)

print('Spliting columns ...')
trn_id = [a[0] for a in trn_word]
trn_title = [a[1] for a in trn_word]
trn_content = [a[2] for a in trn_word]
trn_label = [a[3] for a in trn_word]
del trn_word
gc.collect()

evl_id = [a[0] for a in evl_word]
evl_title = [a[1] for a in evl_word]
evl_content = [a[2] for a in evl_word]
del evl_word
gc.collect()

title_model_name = 'vector/title/title_model'
content_model_name = 'vector/content/content_model'
print('Loading model from [%s]' % title_model_name)
title_model = Word2Vec.load(title_model_name)
title_matrix = np.array([title_model[word] for word in (title_model.wv.vocab)])
title_matrix=np.concatenate((np.zeros((1,title_matrix.shape[1])),title_matrix,np.random.random((1,title_matrix.shape[1]))),axis=0)
default_title=title_matrix.shape[0]-1
title_keys=np.array(list(title_model.wv.vocab.keys()))
title_dict={}
for i in range(title_keys.shape[0]):
    title_dict[title_keys[i]]=i+1
title_df=pd.DataFrame(title_matrix)
# title_df.to_csv('word2vec/title_matrix.csv',header=False,index=False)

del title_model,title_matrix,title_keys,title_df
gc.collect()

print('Loading model from [%s]' % content_model_name)
content_model = Word2Vec.load(content_model_name)
content_matrix = np.array([content_model[word] for word in (content_model.wv.vocab)])
content_matrix=np.concatenate((np.zeros((1,content_matrix.shape[1])),content_matrix,np.random.random((1,content_matrix.shape[1]))),axis=0)
default_content=content_matrix.shape[0]-1
content_keys=np.array(list(content_model.wv.vocab.keys()))
content_dict={}
for i in range(content_keys.shape[0]):
    content_dict[content_keys[i]]=i+1
content_df=pd.DataFrame(content_matrix)
# content_df.to_csv('word2vec/content_matrix.csv',header=False,index=False)

del content_model,content_matrix,content_keys,content_df
gc.collect()

max_title_len=25
max_content_len=1500
#id train
train_id=np.array(trn_id)
train_title=np.zeros((train_id.shape[0],max_title_len))
train_content=np.zeros((train_id.shape[0],max_content_len))
train_label=[]
for i in tqdm(range(train_id.shape[0])):
    for j in range(len(trn_title[i])):
        temp = trn_title[i][j]
        if temp in title_dict:
            temp_str = str(title_dict[temp])
        else:
            temp_str = str(default_title)
        train_title[i,j]=temp_str
        if j==max_title_len-1:
            break
    for j in range(len(trn_content[i])):
        temp=trn_content[i][j]
        if temp in content_dict:
            temp_str=str(content_dict[temp])
        else:
            temp_str=str(default_content)
        train_content[i,j]=temp_str
        if j==max_content_len-1:
            break
    if trn_label[i]=='POSITIVE':
        train_label.append(1)
    else:
        train_label.append(0)
train_label=np.array(train_label)
train_id=[np.string_(i) for i in train_id]
file = h5py.File('train/train_data.h5','w')
file.create_dataset('train_id', data = train_id)
file.create_dataset('train_title', data = train_title)
file.create_dataset('train_content',data = train_content)
file.create_dataset('train_label',data = train_label)
file.close()

#id eval
eval_id=np.array(evl_id)
eval_title=np.zeros((eval_id.shape[0],max_title_len))
eval_content=np.zeros((eval_id.shape[0],max_content_len))
for i in tqdm(range(eval_id.shape[0])):
    for j in range(len(evl_title[i])):
        temp = evl_title[i][j]
        if temp in title_dict:
            temp_str = str(title_dict[temp])
        else:
            temp_str = str(default_title)
        eval_title[i,j]=temp_str
        if j==max_title_len-1:
            break
    for j in range(len(evl_content[i])):
        temp=evl_content[i][j]
        if temp in content_dict:
            temp_str=str(content_dict[temp])
        else:
            temp_str=str(default_content)
        eval_content[i,j]=temp_str
        if j==max_content_len-1:
            break
eval_id=[np.string_(i) for i in eval_id]
file = h5py.File('eval/eval_data.h5','w')
file.create_dataset('eval_id', data = eval_id)
file.create_dataset('eval_title', data = eval_title)
file.create_dataset('eval_content',data = eval_content)
file.close()