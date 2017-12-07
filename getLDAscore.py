# -*- coding: UTF-8 -*-


# 把每篇文章当一个corpus，文章中的句子当一篇文章，计算一个corpus内doc的lda矩阵。
# 这里断句加入了逗号，因为发现逗号隔开的句子也可能存在不同主题。
import gc
# import os
# import re
import h5py
import codecs
import jieba
import jieba.posseg as pseg
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.decomposition import LatentDirichletAllocation
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
            if len(content) != 0:
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


def get_break_punctuation(tag='string'):
    # 自定义断句符号
    if tag == 'string':
        return r'[|。…；：！？\.;:!?\\t]+'
    elif tag == 'list':
        return  ['。', '，', '……', '；', '：', '！', '？', '.', ',', ';', ':', '!', '?', '\t', '|']
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


def get_sentence_lda(lst_sentence, k_lda):
    # input: lst_sentence = [sentence, sentence, ..., sentence],
    #         where sentence = [word, word, ..., word]
    #         k_lad: the number of topics
    # output: arr_lda = [lda_score, lda_score, ..., lda_score]
    #         where lda_score = [topic_prob(1), topic_prob(2), ..., topic_prob(k_lda)]
    lst_corpus = [' '.join(sent) for sent in lst_sentence]
    word_freq = CountVectorizer().fit_transform(lst_corpus)
    lda_model = LatentDirichletAllocation(n_topics=k_lda, random_state=0)
    return lda_model.fit_transform(word_freq)


def get_arr_padding(arr, size_padding):
    # 把一个arr内的行数限定为size_padding，超过截断，不足补0
    # input: old_array = np.array([ldavec, ldavec, ..., ldavec])
    #        size_padding is the number of self-defined rows
    # output: new_array = np.array([ldavec, ldavec, ..., ldavec])
    if arr.shape[0] > size_padding:
        return arr[:size_padding, :]
    else:
        paddingmat = np.zeros((size_padding-arr.shape[0], arr.shape[1]))
        return np.concatenate((arr, paddingmat), axis=0)


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
    data_new = []
    lst_length = []
    lst_exception = []
    # 1. 处理每一篇文章
    print('Get lda-score-based data_new...')
    for doc in tqdm(data):
        # 读取当前文章内容
        if dict_par['tag_text'] == 'content':
            text = doc[2]
        elif dict_par['tag_text'] == 'title':
            text = doc[1]
        else:
            print('Wrong tag_text!')
            return None
        # 清洗当前文章
        lst_text = get_word_tag(text)  # 标注当前文章词性
        lst_text_clean = get_word_clean(lst_text)  # 去掉当前文章的虚词和冗余符号
        lst_text_sentence = get_sentence(lst_text_clean)  # 划分当前文章的句子
        if len(lst_text_sentence) == 0:
            lst_exception.append(np.string_(doc[0]))
            print(doc[0])
            continue
        # 统计当前文章的句子个数
        lst_length.append(len(lst_text_sentence))
        # 给当前文章的每个句子算一个主题，以主题标号代替句子
        lda = get_sentence_lda(lst_text_sentence, dict_par['k_lda'])
        # 存储到data_new中
        if dict_par['tag_label'] == 'yes':
            label = 1 if doc[-1] == 'POSITIVE' else 0
            data_new.append([np.string_(doc[0]), lda, label])
        else:
            data_new.append([np.string_(doc[0]), lda])
    print('Succeed to get lda-score-based data_new.')

    # 2. 补齐文章向量并存入data_new
    size_padding = get_padding_size(lst_length)  # 让文章对齐
    del lst_length
    gc.collect()
    for i_doc, doc in enumerate(tqdm(data_new)):
        data_new[i_doc][1] = get_arr_padding(doc[1], size_padding)
    print('Succeed to generate lda-score-based data.')

    return data_new, lst_exception


if __name__ == '__main__':
    # 设置超参数
    dict_par = {
        'max_line': float('Inf'),
        'tag_label': 'yes',
        'tag_text': 'content',
        'k_lda': 5
    }

    # 载入数据集
    load_path = './Datasets/train_semi.tsv'
    data = load_data(load_path, dict_par['tag_label'], dict_par['max_line'])

    # 生成句向量data
    new_data, lst_exception = get_main_workflow(data, dict_par)
    print('Succeed to get the sentence-vector-based data!')

    # array化
    new_id = np.array([now_data[0] for now_data in new_data])
    new_mat = np.array([now_data[1] for now_data in new_data])
    new_exception = np.array(lst_exception)
    if dict_par['tag_label'] == 'yes':
        new_label = np.array([now_data[-1] for now_data in new_data])
    del new_data
    gc.collect()

    # 存储句向量data
    save_path = './Datasets/train_semi_sentvec.h5'
    # save_data(data, save_path)
    with h5py.File(save_path, 'w') as f_w:
        f_w.create_dataset('ID', data=new_id)
        f_w.create_dataset('LDA', data=new_mat)
        f_w.create_dataset('EXP', data=new_exception)
        if dict_par['tag_label'] == 'yes':
            f_w.create_dataset('LABEL', data=new_label)
    print('Succeed to save the data!')


