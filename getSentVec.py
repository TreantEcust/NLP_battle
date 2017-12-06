# -*- coding: UTF-8 -*-


# 通过sentence_vector = mean(tf-idf*word_vector)的方式生成每篇文章的句向量
import gc
# import os
# import re
import h5py
import pickle
import codecs
import jieba
import jieba.posseg as pseg
from jieba import analyse
from gensim.models import Word2Vec
import numpy as np
from tqdm import tqdm


def load_data(file_address, tag_label, max_line):
    # 载入数据，tag_label=no是无类别标记数据，tag_label=yes是有类别标记数据，label放最后
    print('Load the data...')
    count_line = 0
    data = []
    with codecs.open(file_address, 'r', 'utf-8') as f:
        for line in f.readlines():
            text = line.strip().split('\t')
            content = ''.join(text[2:-1]).replace(' ', '\t') if tag_label == 'yes' else ''.join(text[2:]).replace(' ', '\t')
            count_line += 1
            if count_line >= max_line:
                break
            if len(content)!=0:
                if tag_label == 'yes':  # train
                    data.append([text[0], text[1], content, text[-1]])
                else:  # test
                    data.append([text[0], text[1], content])
            else:
                # print count_line-1
                continue
    print('Succeed to load the data.')
    return data


def save_data(data, save_path):
    with h5py.File(save_path, 'w') as f:
        f.create_dataset('data', data=data)
    print('Succeed save .h5 file.')
    return None


def save_w2v_model(model, save_path):
    model.save(save_path)
    print('Successfully saving word2vec model to [%s]' % save_path)
    return None


def load_w2v_model(load_path):
    model = Word2Vec.load(load_path)
    print('Successfully loading word2vec model from [%s]' % load_path)
    return model


def get_break_punctuation(tag='string'):
    # 自定义断句符号
    if tag == 'string':
        return r'[|。…；：！？\.;:!?\\t]+'
    elif tag == 'list':
        return ['。', '……', '；', '：', '！', '？', '.', ';', ':', '!', '?', '\t', '|']
    else:
        print('Wrong tag!')
        return None


def get_function_words():
    # 生成词向量前一个句子里需要去掉的词的词性，jieba词性tag
    # 包括虚词、标点
    return ['c', 'e', 'o', 'p', 'u', 'w', 'x', 'y']


def add_words_dict(dict_path):
    # 添加用户自定义词典
    jieba.load_userdict(dict_path)
    print('Successfully load self-defined dictionary!')
    return None


def get_word_tag(str):
    # 给一篇文章切词并标注词性
    # input: str is a string
    # output: lst = [tuple, tuple, ..., tuple],
    #         where tuple = (word, tag)
    # add_words_dict('This is a path.') # 加载用户自定义词典
    return list(pseg.cut(str))


def get_word_clean(lst):
    # 根据词性过滤虚词与标点等
    # input: lst = [tuple, tuple, ..., tuple],
    #         where tuple = (word, tag)
    # output: lst = [word, word, ..., word],
    #         where function words except break punctuations are filtered
    lst_new = []
    for i, j in lst:
        if (i not in get_break_punctuation(tag='list')) and (j in get_function_words()):
            continue
        else:
            lst_new.append(i)
    return lst_new


def get_sentence(lst):
    # 根据断句标点断句
    # input: lst = [word, word, ..., word]
    # output: lst_sentence = [sentence, sentence, ..., sentence],
    #         where sentence = [word, word, ..., word]
    lst_sentence = []
    lst_tempor = []
    for word in lst:
        if len(lst_tempor) != 0:
            if word in get_break_punctuation(tag='list'):
                lst_sentence.append(lst_tempor)
                lst_tempor = []
            else:
                lst_tempor.append(word)
        else:
            if word in get_break_punctuation(tag='list'):
                continue
            else:
                lst_tempor.append(word)
    del lst_tempor
    gc.collect()
    return lst_sentence


def get_score_dict(str, tag='tfidf', k=20):
    # 得到一篇文章的前k个关键词及其分数，分数是tfidf或textrank的取值
    # input: str is a string
    # output: dict_score = {keyword: score}
    extractor = analyse.extract_tags if tag == 'tfidf' else analyse.textrank
    keywords = extractor(str, withWeight=True, topK=k)
    dict_score = {}
    for keyword, score in keywords:
        dict_score[keyword] = score
    return dict_score


def get_wordvector(lst_sentence, size, window, min_count):
    # 根据word2vector算法生成词向量
    # input: lst_sentence = [sentence, sentence, ..., sentence],
    #        where sentence = [word, word, ..., word]
    # output: w2v model
    try:
        model = Word2Vec(lst_sentence, size, window, min_count)
        print('Success: get_model with {}size, {}window, {}min_count!'.format(size, window, min_count))
        return model
    except:
        if len(lst_sentence) != 0:
            print('Incorrect parameters!')
        else:
            print('Text does not exist!')
        return None


def get_sentence_vector(model, lst_sentence, dict_score, size):
    # 把一篇文章里每一句的词转为词向量，再合并为句向量
    # input: lst_sentence = [sentence, sentence, ..., sentence],
    #        where sentence = [word, word, ..., word]
    # output: lst_vector = [sentvec, sentvec, ..., sentvec]
    #        where sentvec is the vector of a sentence
    lst_vector = []
    for sentence in lst_sentence:
        lst_tempor = []
        for word in sentence:
            try:
                wordvec = model[word]
            except:
                wordvec = np.random.random([size, ])*0.5-0.25
            try:
                wordscore = dict_score[word]
            except:
                wordscore = 0.000001
            lst_tempor.append(wordvec*wordscore)
        sentvec = np.mean(lst_tempor, axis=0)
        lst_vector.append(sentvec)
    return lst_vector


def get_padding(lst_vector, vec_shape, size_padding):
    # 把一个list内的item个数限定为size_padding，超过截断，不足补0
    # input: lst_vector = [sentvec, sentvec, ..., sentvec]
    #        where sentvec is the vector of a sentence
    #        size_padding is the pre-defined output length for lst_vector
    #        vec_shape is the shape of sentvec
    # output: lst_vector = [sentvec, sentvec, ..., sentvec, paddingvec, ..., paddingvec]
    #        where sentvec is the vector of a sentence, paddingvec is a zero-vector
    if len(lst_vector) > size_padding:
        return lst_vector[:size_padding]
    else:
        paddingvec = np.zeros(vec_shape)
        lst_vector += [paddingvec] * (size_padding - len(lst_vector))
        return lst_vector


def get_padding_size(lst_length):
    # input： lst_length = [int, int, ..., int]
    # output: a integer
    return int((np.max(lst_length) + np.mean(lst_length))*0.25)


def get_main_workflow(data, dict_par):
    # 从读入数据到生成句向量的主要流程
    # input: data = [doc, doc, ..., doc]
    #        where doc = [id, title, content[, label]]
    # output: data_new = [doc, doc, ..., doc]
    #         where doc = [id, padded_vectorized_content[, label]]
    lst_all_sentence = []
    data_new = []
    lst_length = []
    # 1. 处理每一篇文章
    print('Pre-processing...')
    for doc in tqdm(data):
        # 1-1. 取当前文章
        if dict_par['tag_text'] == 'content':
            text = doc[2]
        elif dict_par['tag_text'] == 'title':
            text = doc[1]
        else:
            print('Wrong tag_text!')
            return None

        # 1-2. 提取当前文章的关键词及对应词值(score)
        dict_score = get_score_dict(text, dict_par['tag_dict_score'], dict_par['k'])

        # 1-3. 标注当前文章词性
        lst_text = get_word_tag(text)
        # 去掉当前文章的虚词和冗余符号
        lst_text_clean = get_word_clean(lst_text)
        # 划分当前文章的句子
        lst_text_sentence = get_sentence(lst_text_clean)

        # 1-4. 统计当前文章的句子个数
        lst_length.append(len(lst_text_sentence))

        # 1-5. 合并句子，存相关信息到新data内
        if len(lst_text_sentence) != 0:
            lst_all_sentence.append(lst_text_sentence)
            if dict_par['tag_label'] == 'yes':
                data_new.append([doc[0], [lst_text_sentence, dict_score], doc[-1]])
            elif dict_par['tag_label'] == 'no':
                data_new.append([doc[0], [lst_text_sentence, dict_score]])
            else:
                print('Wrong tag_label!')
                return None
    print('Succeed to pre-process the data.')

    # 2. 训练w2v_model
    print('Generate w2v_model...')
    w2v_model = get_wordvector(lst_all_sentence, dict_par['size'], dict_par['window'], dict_par['min_count'])
    print('Succeed to generate the w2v_model.')

    # 3. 生成句向量
    size_padding = get_padding_size(lst_length)
    del lst_length
    gc.collect()
    print('Generate sentence-vector-based data...')
    for i_doc_vec, doc_vec in enumerate(tqdm(data_new)):
        now_vector = get_sentence_vector(w2v_model, doc_vec[1][0], doc_vec[1][1], dict_par['size'])
        now_vector = get_padding(now_vector, now_vector[0].shape, size_padding)
        data_new[i_doc_vec][1] = now_vector
        data_new[i_doc_vec][0] = np.string_(data_new[i_doc_vec][0])
        if dict_par['tag_label'] == 'yes':
            data_new[i_doc_vec][-1] = 1 if data_new[i_doc_vec][-1] == 'POSITIVE' else 0
    print('Succeed to generate sentence-vector-based data.')

    return data_new


if __name__ == '__main__':
    # 设置超参数
    dict_par = {
        'max_line': float('Inf'),
        'tag_label': 'yes',
        'tag_text': 'content',
        'tag_dict_score': 'tfidf',
        'k': 20,
        'size': 50,
        'window': 5,
        'min_count': 2
    }

    # 载入数据集
    load_path = './Datasets/train_semi.tsv'
    data = load_data(load_path, dict_par['tag_label'], dict_par['max_line'])

    # 生成句向量data
    new_data = get_main_workflow(data, dict_par)
    print('Succeed to get the sentence-vector-based data!')

    # array化
    new_id = np.array([now_data[0] for now_data in new_data])
    new_mat = np.array([now_data[1] for now_data in new_data])
    if dict_par['tag_label'] == 'yes':
        new_label = np.array([now_data[-1] for now_data in new_data])
    del new_data
    gc.collect()

    # 存储句向量data
    save_path = './Datasets/train_semi_sentvec.h5'
    # save_data(data, save_path)
    with h5py.File(save_path, 'w') as f_w:
        f_w.create_dataset('ID', data=new_id)
        f_w.create_dataset('Data', data=new_mat)
        if dict_par['tag_label'] == 'yes':
            f_w.create_dataset('LABEL', data=new_label)
    print('Succeed to save the data!')


