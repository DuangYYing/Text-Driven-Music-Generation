import os
import torch
import numpy as np
#from nltk import word_tokenize
from collections import Counter
from torch.autograd import Variable
from parser import args
from utils import seq_padding, subsequent_mask

class PrepareData:
    def __init__(self):

        # 读取数据 并分词
        self.train_en = self.load_data(args.trainword_file)
        self.train_cn = self.load_data(args.trainmidi_file)
        self.dev_en = self.load_data(args.devword_file)
        self.dev_cn = self.load_data(args.devmidi_file)

        # 构建单词表
        self.en_word_dict, self.en_total_words, self.en_index_dict = self.build_dict(self.train_en)
        self.cn_word_dict, self.cn_total_words, self.cn_index_dict = self.build_dict(self.train_cn)

        # id化
        self.train_en, self.train_cn = self.wordToID(self.train_en, self.train_cn, self.en_word_dict, self.cn_word_dict)
        self.dev_en, self.dev_cn = self.wordToID(self.dev_en, self.dev_cn, self.en_word_dict, self.cn_word_dict)

        # 划分batch + padding + mask
        self.train_data = self.splitBatch(self.train_en, self.train_cn, args.batch_size)
        self.dev_data = self.splitBatch(self.dev_en, self.dev_cn, args.batch_size)

    def load_data(self, path):
        en = []
        #cn = []
        with open(path, 'r') as f:
            for line in f:                               #一行行读，每行小写，每个加进去en里
                line = line.strip().split('\t')          
                en=en+line
               # en = en + ["BOS"] + line + ["EOS"]
                #en.append(["BOS"] + word_tokenize(line[0].lower()) + ["EOS"])       # 分词！改word_tokenize！english 
                #cn.append(["BOS"] + word_tokenize(" ".join([w for w in line[1]])) + ["EOS"]) # 分词！改！chinese

        return en
    
    def build_dict(self, sentences, max_words = 50000):
        word_count = Counter()

        for sentence in sentences:
            #print("sentence:"+sentence)
            for s in sentence.split(','):
                #print("s:"+ s)
                word_count[s] += 1   #统计每个词出现的个数

        ls = word_count.most_common(max_words)      # word_counts.most_common() 输入整数时C 返回的是排名前C个的数据  不输入是按照出现次数对所有数据排序
        total_words = len(ls) + 2                  # word_counts.most_common()   返回值的类型是list[]   第一个参数：字符   第二个参数是：字符出现的次数

        word_dict = {w[0]: index + 2 for index, w in enumerate(ls)}
        word_dict['UNK'] = args.UNK
        word_dict['PAD'] = args.PAD

        index_dict = {v: k for k, v in word_dict.items()}       

        return word_dict, total_words, index_dict

    def wordToID(self, en, cn, en_dict, cn_dict, sort=True):    #应该只要en_id
        length = len(en)

        out_en_ids = [[en_dict.get(w, 0) for w in sent.split(',')] for sent in en]
        out_cn_ids = [[cn_dict.get(w, 0) for w in sent.split(',')] for sent in cn]
        
        # sort sentences by english lengths
        def len_argsort(seq):
            return sorted(range(len(seq)), key=lambda x: len(seq[x]))

        # 把中文和英文按照同样的顺序排序
        if sort:
            sorted_index = len_argsort(out_en_ids)
            out_en_ids = [out_en_ids[i] for i in sorted_index]
            out_cn_ids = [out_cn_ids[i] for i in sorted_index]
            
        return out_en_ids, out_cn_ids

    def splitBatch(self, en, cn, batch_size, shuffle=True):
        idx_list = np.arange(0, len(en), batch_size)
        if shuffle:
            np.random.shuffle(idx_list)
        batch_indexs = []
        for idx in idx_list:
            batch_indexs.append(np.arange(idx, min(idx + batch_size, len(en))))
        
        batches = []
        for batch_index in batch_indexs:
            batch_en = [en[index] for index in batch_index]  
            batch_cn = [cn[index] for index in batch_index]
            batch_cn = seq_padding(batch_cn)
            batch_en = seq_padding(batch_en)
            batches.append(Batch(batch_en, batch_cn))

        return batches


class Batch:
    "Object for holding a batch of data with mask during training."
    def __init__(self, src, trg=None, pad=0):

        src = torch.from_numpy(src).to(args.device).long()
        trg = torch.from_numpy(trg).to(args.device).long()

        self.src = src
        self.src_mask = (src != pad).unsqueeze(-2)
        if trg is not None:
            self.trg = trg[:, :-1]
            self.trg_y = trg[:, 1:]
            self.trg_mask = \
                self.make_std_mask(self.trg, pad)
            self.ntokens = (self.trg_y != pad).data.sum()

    @staticmethod
    def make_std_mask(tgt, pad):
        "Create a mask to hide padding and future words."
        tgt_mask = (tgt != pad).unsqueeze(-2)
        tgt_mask = tgt_mask & Variable(
            subsequent_mask(tgt.size(-1)).type_as(tgt_mask.data))
        return tgt_mask
        
        
        
