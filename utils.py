import numpy as np
import os


def get_data(opt):
    datas = np.load(os.path.join(opt.data_path, opt.filename), allow_pickle=True)
    data, word2ix, ix2word = datas['data'], datas['word2ix'].item(), datas['ix2word'].item()
    return data, word2ix, ix2word


def trans_ix2word(data, ix2word):
    words = []
    for i in data:
        words.append(ix2word[i])

    return words


def fPrint(wordList):
    for i in wordList:
        if i == '，':
            print(i, end='\t')
        elif i in ['。', '？', '！']:
            print(i, end='\n')
        else:
            print(i, end='')

