# -*- coding: UTF-8 -*-
import pickle
import gc
from tqdm import tqdm
import jieba as jb
from gensim.models import Word2Vec


def load_data(file_address, tag, max_line='Inf'):
    count_line = 0
    data = []
    with open(file_address, 'rb') as f:
        for line in f.readlines():
            text = line.strip().split('\t')
            content = ''.join(text[2:-1]).replace(' ', '\t') if tag == 0 else ''.join(text[2:]).replace(' ', '\t')
            count_line += 1
            if count_line >= max_line:
                break
            if len(content)!=0:
                if tag == 0:  # train
                    data.append([text[0], text[1], content, text[-1]])
                else:  # test
                    data.append([text[0], text[1], content])
            else:
                # print count_line-1
                continue
    print('Success: load_data!')
    return data


def save_pkl(data, data_name):
    file_name = './temp_data/' + data_name + '.pkl'
    with open(file_name, 'wb') as f:
        pickle.dump(data, f)
    print('Success: save_pkl!')
    return None


print('Now load data...')
train = load_data('../Datasets/train_50.tsv', tag=0, max_line='Inf')
test = load_data('../Datasets/evaluation_public_40.tsv', tag=1, max_line='Inf')
print 'train samples:{}'.format(len(train)), 'test samples:{}'.format(len(test))


def get_end_word():
    lst_end_word = ['，', '、', '！', '？', '。', '：', ' ', '；']
    return lst_end_word


def cut_sent(document):
    # document is a long string
    # sentences = [sent1, sent2, ..., sentN], where senti is a string
    for i_end in get_end_word():
        document = document.replace(i_end, '。')
    sentences = filter(lambda x:len(x)!=0, document.split('。'))
    return sentences


def get_sentences(lst_documents):
    # 断句
    # old lst_documents = [doc1, doc2, ..., docN], where doci = [id, title, content, label(or not)]
    # new lst_documents = [doc1, doc2, ..., docN], where doci = [id, title, sentences, label(or not)]
    # sentences = [sent1, sent2, ..., sentN]
    for document in tqdm(lst_documents):
        now_sentences = cut_sent(document[2])
        document[2] = now_sentences
    print('Success: get_sentences!')
    return lst_documents


print('Now get sentences...')
train = get_sentences(train)
test = get_sentences(test)


def get_words(lst_documents):
    # 切词
    # old lst_documents = [doc1, doc2, ..., docN], where doci = [id, title, sentences, label(or not)]
    # old lst_documents = [doc1, doc2, ..., docN], where doci = [id, title, words, label(or not)]
    # words = [sent1, sent2, ..., sentN], where senti = [word1, word2, ..., wordN]
    num_words = []  # 记录每句话的词数
    for document in tqdm(lst_documents):
        sentences = document[2]
        now_sentences = []
        for sent in sentences:
            if len(sent)!=0:
                now_sent = list(jb.cut(sent))
                num_words.extend([len(now_sent)])
                now_sentences.append(now_sent)
            else:
                continue
        if len(now_sentences)!=0:
            document[2] = now_sentences
        else:
            print('Fail to cut words!')
    print('Success: get_words!')
    return lst_documents, num_words


print('Now get words...')
train, num_words_train = get_words(train)
test, num_words_test = get_words(test)
num_words_all = num_words_train + num_words_test
save_pkl(train, 'words_train')
save_pkl(test, 'words_test')
save_pkl(num_words_all, 'num_words_all')
del num_words_train, num_words_test, num_words_all
gc.collect()

# 合成用于w2v的长句
# word_all = [train_doc1, train_doc2, ..., train_docN, test_doc1, test_doc2, ..., test_docN ]
# train_doci = [id, title, words, label], test_doci = [id, title, words]
both = train + test
texts = []
for i in tqdm(both):
    temp_text = []
    for j in i[2]:
        temp_text.append(j)
    texts.extend(temp_text)
print 'There are {} sentences now!'.format(len(texts))


def get_model(texts, size=100, window=5, min_count=2):
    # texts = [sent1, sent2, ..., sentN], where senti = [word1, word2, ..., wordN]
    try:
        model = Word2Vec(texts, size, window, min_count)
        print('Success: get_model with {}size, {}window, {}min_count!'.format(size, window, min_count))
        return model
    except:
        if len(texts)!=0:
            print('Incorrect parameters!')
        else:
            print('Text does not exist!')
        return None


print('Now generate model...')
now_model = get_model(texts)
save_pkl(now_model, 'Model_w2v')