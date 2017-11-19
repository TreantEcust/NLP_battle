# -*- coding: utf-8 -*-
"""
Created on Mon Oct 30 12:57:37 2017

@author: ZWang
"""
from gensim.models import Word2Vec
import pickle
import copy
import gc
from tqdm import tqdm


def load_word(word_file):
    print('Loading word from file [%s]' % word_file)
    fp = open(word_file, 'rb')
    dataset = pickle.load(fp)
    fp.close()

    trn_word = dataset['trn']
    evl_word = dataset['evl']
    return [trn_word, evl_word]


def model_trn(words, min_count, window, dim, save_name):
    print('Training word2vec model ...')
    print('len_words = %d' % len(words))
    model = Word2Vec(words, size=dim, min_count=min_count, window=window, workers=24)
    print('Saving word2vec model to file [%s]' % save_name)
    model.save(save_name)
    return model

def split_sentence(x):
    doc=[]
    sen=[]
    for x1 in x:
        if x1=='。' or x1=='！' or x1=='.' or x1=='!' or x1=='?' or x1=='？':
            sen.append(x1)
            doc.append(sen)
            sen=[]
        else:
            sen.append(x1)
    if len(sen)!=0:
        doc.append(sen)
    return doc

if not ('trn_word' in dir() and 'evl_word' in dir()):
    word_file = 'word_data.pkl'
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

    trn_title.extend(evl_title)
    del evl_title
    gc.collect()

    trn_content.extend(evl_content)
    del evl_content
    gc.collect()


# print('Title word2vec model ...')
# min_count = 2
# window = 5
# dim = 256
# save_name = 'title256/title_model256'
# title_model = model_trn(words=trn_title, min_count=min_count, window=window , dim=dim, save_name=save_name)

print('Content word2vec model ...')
min_count = 5
window = 5
dim = 256
save_name = 'content256/content_model256'
content_model = model_trn(words=trn_content, min_count=min_count, window=window, dim=dim, save_name=save_name)

