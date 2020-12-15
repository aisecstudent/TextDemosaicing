# -*- coding: utf-8 -*-

import json, re, torch, os
import pandas as pd
from text2image import txt2im
from tqdm import tqdm
from sklearn.model_selection import train_test_split
import numpy as np
from gensim.models import KeyedVectors

input_dir = '../input/'
resource_dir = '../resource/'

def load_corpus():
    waimai = pd.read_csv(input_dir + "text_set/waimai_10k.csv")
    ChnSentiCorp_htl_all = pd.read_csv(input_dir + "text_set/ChnSentiCorp_htl_all.csv")
    review = waimai.review.drop_duplicates().tolist() + ChnSentiCorp_htl_all.review.drop_duplicates().tolist()
    review = [x for x in review if str(x) != 'nan']
    return review

line_pattern = re.compile((u""
    + u"(%(prefix)s+\S%(postfix)s+)" # 标点
    + u"|(%(prefix)s*\w+%(postfix)s*)" # 单词
    + u"|(%(prefix)s+\S)|(\S%(postfix)s+)" # 标点
    + u"|(\d+%%)" # 百分数
    ) % {
    "prefix": u"['\(<\[\{‘“（《「『]",
    "postfix": u"[:'\)>\]\}：’”）》」』,;\.\?!，、；。？！]",
})

def split_line(line, max_len=64):
    lst = ''
    lines = []
    while line:
        ro = line_pattern.match(line)
        end = 1 if not ro else ro.end()
        if len(lst) + len(line[:end]) < max_len:
            lst += line[:end]
        else:
            lines.append(lst)
            lst = line[:end][:max_len]
        line = line[end:]
    if lst != '':
        lines.append(lst)
    return lines

def get_vocab(review):
    vocab = {'[PAD]': 0, '[BOS]': 1, '[EOS]': 2, '[CLS]': 3, '[SEP]': 4, '[UNK]': 5}
    for line in review:
        for char in line:
            if char not in vocab:
                vocab[char] = len(vocab)
    return vocab

def save_w2v():
    vocab = json.load(open(input_dir + 'vocab.json', 'r'))
    model = KeyedVectors.load_word2vec_format(resource_dir + 'sgns.merge.char')
    w2v = []
    vocab_inverse = {vocab[k]:k for k in vocab}
    zeros = np.zeros((300), dtype=np.float32)
    for i in range(4000):
        word = vocab_inverse.get(i, 0)
        if word in model:
            w2v.append(model.get_vector(word))
        else:
            w2v.append(zeros)
    w2v = torch.from_numpy(np.asarray(w2v, dtype=np.float32))
    torch.save(w2v, input_dir + 'word2vec.torch')

def main():
    new_text1 = "蚂蚁准备上市前，杭州一栋大楼的员工都沸腾了。好多员工要变成千万富翁，基本无心工作。"
    new_text2 = "蚂蚁暂缓上市后，员工失望至极，退订豪车豪宅，据说杭州房价连夜调降15％。"

    review = load_corpus()
    review += [new_text1, new_text2]
    
    if not os.path.exists(input_dir + 'vocab.json'):
        vocab = get_vocab(review)
        json.dump(vocab, open(input_dir + 'vocab.json', 'w'), ensure_ascii=False)
    else:
        vocab = json.load(open(input_dir + 'vocab.json', 'r'))

    count = 0
    for lines in tqdm(review):
        for text in split_line(lines):
            if text != '':
                txt2im(text, "image_%d.png" % count, neighbor=5)
                count += 1
    
    txt2im(new_text1, "image_%d.png" % count, neighbor=5)
    count += 1
    txt2im(new_text2, "image_%d.png" % count, neighbor=5)

def train_w2v():
    import gensim
    class SentenceIterator: 
        def __init__(self): 
            self.filepath = input_dir + 'meta_data.jsonl'

        def __iter__(self): 
            for line in open(self.filepath, 'r'): 
                yield [i for i in orjson.loads(line.strip())['text']]
    sentences = SentenceIterator() 
    w2v_model = gensim.models.Word2Vec(sentences, size=32, window=20, min_count=1, workers=6, iter=10, sg=1, negative=20, sample=1e-3)
    w2v_model.save(input_dir + 'review.w2v')

    vocab = json.load(open(input_dir + 'vocab.json', 'r'))
    w2v = []
    vocab_inverse = {vocab[k]:k for k in vocab}
    zeros = np.zeros((32), dtype=np.float32)
    for i in range(4000):
        word = vocab_inverse.get(i, '[PAD]')
        if word in w2v_model:
            w2v.append(w2v_model.wv.get_vector(word))
        else:
            w2v.append(zeros)
    w2v = torch.from_numpy(np.asarray(w2v, dtype=np.float32))
    torch.save(w2v, input_dir + 'review.w2v.torch')


if __name__ == '__main__':
    main()
    #train_w2v()
    save_w2v()






