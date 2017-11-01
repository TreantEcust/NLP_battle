# -*- coding: UTF-8 -*-

import numpy as np

def get_end_word():
    lst_end_word = ['，', '、', '！', '？', '。', '：', ' ', '；']
    return lst_end_word

def get_stop_word():
    stop_word_address = '../Datasets/stop_words.txt'
    lst_stop_word = []
    with open(stop_word_address, 'rb') as f:
        for word in f.readlines():
            lst_stop_word.append(word.strip())
    return lst_stop_word

def get_text_info(text):
    # text = long string
    # 原始文章/去掉停用词文章的长度
    dict_sent = {}
    dict_sent['old_len'] = len(text)
    # new_text = filter(lambda x:x not in get_stop_word(), text)
    # dict_sent['new_len'] = len(new_text)
    # dict_sent['ratio_len'] = len(new_text)/float(len(text))

    # 生成原始分句文章
    for i_end in get_end_word():
        text = text.replace(i_end, '。')
    cut_text = text.split('。')
    dict_sent['num_sent'] = len(cut_text)

    # 原始分句文章句子信息
    sentences_length = np.array(list(map(lambda x:len(x), cut_text))) # sentences_length = [len(sent1), len(sent2), ..., len(sentN)]
    dict_sent['max_sent'] = max(sentences_length)
    dict_sent['min_sent'] = min(sentences_length)
    dict_sent['range_sent'] = max(sentences_length) - min(sentences_length)
    dict_sent['mean_sent'] = np.mean(sentences_length)

    sentences_diff = np.diff(sentences_length) # sentences_diff = [len(sent2)-len(sent1), len(sent3)-len(sent2), ..., len(sentN)-len(sentN-1)]
    if len(sentences_diff) == 0:
        sentences_diff = [0]
    dict_sent['max_diff_sent'] = max(sentences_diff)
    dict_sent['min_diff_sent'] = min(sentences_diff)
    dict_sent['range_diff_sent'] = max(sentences_diff) - min(sentences_diff)
    dict_sent['mean_diff_sent'] = np.mean(sentences_diff)

    set_cut_text = set(cut_text)
    dict_sent['diff_repeat_sent'] = len(cut_text) - len(set_cut_text)
    dict_sent['ratio_repeat_sent'] = (len(cut_text) - len(set_cut_text)) / float(len(cut_text))
    # sentences_repeat = [] # sentences_repeat = [repeat_sent1, repeat_sent2, ..., repeat_sentM]
    sentences_repeat_length = [] # sentences_repeat = [len(repeat_sent1), len(repeat_sent2), ..., len(repeat_sentM)]
    for sent in set_cut_text:
        if cut_text.count(sent) > 1:
            # sentences_repeat.append(sent)
            sentences_repeat_length.append(cut_text.count(sent))
    if sentences_repeat_length == []:
        sentences_repeat_length = [0]
    dict_sent['max_repeat_sent'] = max(sentences_repeat_length)
    dict_sent['min_repeat_sent'] = min(sentences_repeat_length)
    dict_sent['range_repeat_sent'] = max(sentences_repeat_length) - min(sentences_repeat_length)
    dict_sent['mean_repeat_sent'] = np.mean(sentences_repeat_length)

    return dict_sent
