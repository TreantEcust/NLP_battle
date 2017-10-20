# -*- coding: UTF-8 -*-
#读取样本
import pickle
import pandas as pd
from getFeature import *
from tqdm import tqdm

def get_load(file_address, max_line=200000,type='train'):
    count_line = 0
    raw_data = []
    with open(file_address, 'r',encoding='UTF-8') as f:
        for line in f.readlines():
            if line==None:
                break
            text = line.strip().split('\t')
            id = text[0]
            title = text[1]
            if type=='train':
                label = text[-1]
                content = ''.join(text[2:-1]).replace(' ', '\t')
            elif type=='test':
                content= ''.join(text[2:]).replace(' ', '\t')
            count_line += 1
            if count_line >= max_line:
                break
            if content:
                if type=='train':
                    raw_data.append([id,title,content,label])
                elif type=='test':
                    raw_data.append([id,title,content])
            elif type=='test':
                raw_data.append([id,-9999,''])
            else:
                continue
    return raw_data

def get_features(raw_data,type='train'):
    # raw_data = [doc1, doc2, ..., docN], where doc = [id,title,content,label]
    # new_data = [doc1, doc2, ..., docN], where doc = [id, fea1, fea2, ..., feaD, label(or not)]
    new_data = []
    i = 0
    feature_names=['id']
    for doc in tqdm(raw_data):
        new_doc = []
        new_doc.append(doc[0]) # get id
        # get features
        if doc[1]==-9999:
            new_doc.extend(list(np.ones(len(feature_names)-1)*(-9999)))
        else:
            dict_text_info = get_text_info(doc[2])
            if len(feature_names)==1:
                feature_names.extend(list(dict_text_info.keys()))
                if type=='train':
                    feature_names.extend(['label'])
            new_doc.extend([dict_text_info[key] for key in dict_text_info])
            if len(doc) == 4: # get label if exists
                new_doc.append(doc[-1])
        # add the document to the new dataset
        new_data.append(new_doc)
        i += 1
    return new_data,feature_names



if __name__ == '__main__':
    #train
    file_address = 'data/train.tsv'
    raw_data = get_load(file_address)
    new_data,feature_names = get_features(raw_data)
    new_data=np.array(new_data)
    new_data=pd.DataFrame(new_data,columns=feature_names)
    new_data.to_csv(file_address.split('.')[0]+'.csv',index=False)

    #eval
    file_address = 'data/evaluation_public.tsv'
    raw_data = get_load(file_address,type='test')
    new_data, feature_names = get_features(raw_data,type='test')
    new_data = np.array(new_data)
    new_data = pd.DataFrame(new_data, columns=feature_names)
    new_data.to_csv(file_address.split('.')[0] + '.csv', index=False)