from tqdm import tqdm
import torch
from transformers import BartTokenizer
from torch.utils.data import Dataset, DataLoader
import random
import numpy as np
import os

from others.utils import pad_sents, save, get_mask, fix_random_seed

class BartDataset(Dataset):
    '''
    Attributes:
        src: it's a list, each line is a sample for source text.
        tgt: it's a list, each line is a sample for target text.
        src_ids: it's a list, each line is a sample for source index after tokenized.
        tgt_ids: it's a list, each line is a sample for target index after tokenized.
    '''
    def __init__(self, tokenizer, multi_news_reader, args):
        self.tokenizer = tokenizer
        self.multi_news_reader = multi_news_reader
        self.src = multi_news_reader.data_src
        self.tgt = multi_news_reader.data_tgt
        self.src = [i.strip('\n') for i in self.src]
        self.tgt = [i.strip('\n') for i in self.tgt]
        self.src_ids = self.tokenize(self.src)
        self.tgt_ids = self.tokenize(self.tgt)

    def __len__(self):
        return len(self.src)

    def __getitem__(self, idx):
        return self.src_ids[idx], self.tgt_ids[idx]

    def tokenize(self, data):
        tokenized_text = [self.tokenizer.encode(i, add_special_tokens=False) for i in tqdm(data)]
        return tokenized_text

    def collate_fn(self, data):
        # rebuld the raw text and truncate to max length
        max_input_len = 1024
        max_output_len = 256
        raw_src = [pair[0] for pair in data]
        raw_tgt = [pair[1] for pair in data]
        raw_src = [i[:max_input_len-1] for i in raw_src]
        raw_tgt = [i[:max_output_len-1] for i in raw_tgt]
        src = []
        tgt = []
        # remove blank data
        for i in range(len(raw_src)):
            if (raw_src[i] != []) and (raw_tgt[i] != []):
                src.append(raw_src[i])
                tgt.append(raw_tgt[i])
        # make input mask
        mask = torch.tensor(get_mask(src, max_len=max_input_len))
        # make input ids
        src_ids = torch.tensor(pad_sents(src, 1, max_len=max_input_len)[0])
        # make output ids
        decoder_ids = [[0]+i for i in tgt]
        # make output labels
        label_ids = [i+[2] for i in tgt]
        decoder_ids = torch.tensor(pad_sents(decoder_ids, 1, max_len=max_output_len)[0])
        label_ids = torch.tensor(pad_sents(label_ids, -100, max_len=max_output_len)[0])

        return src_ids, decoder_ids, mask, label_ids


class DataReader(object):
    '''
    Attributes:
        data_src: source text
        data_tgt: target text
    '''
    def __init__ (self, args):
        self.args = args
        self.raw_data = self.load_multinews_data(self.args)
        self.data_src = self.raw_data[0]
        self.data_tgt = self.raw_data[1]

    def file_reader(self, file_path):
        file = open(file_path, 'r')
        lines = file.readlines()
        return lines

    def load_multinews_data(self, args):
        train_src_path = args.data_path + args.data_name + '/' + args.mode + '.source'
        train_tgt_path = args.data_path + args.data_name + '/' + args.mode + '.target'
        train_src_lines = self.file_reader(train_src_path)
        train_tgt_lines = self.file_reader(train_tgt_path)
        return (train_src_lines, train_tgt_lines)

def data_builder(args):
    save_path = args.data_path + args.data_name + '/' + args.mode + 'loader' + '.pt'
    data_reader = DataReader(args)
    tokenizer = BartTokenizer.from_pretrained('facebook/bart-base')
    train_set = BartDataset(tokenizer, data_reader, args)
    if args.mode == 'train':
        data_loader = DataLoader(dataset=train_set,
                                batch_size=args.batch_size,
                                shuffle=True,
                                collate_fn=train_set.collate_fn)
    else:
        data_loader = DataLoader(dataset=train_set,
                                batch_size=args.batch_size,
                                shuffle=False,
                                collate_fn=train_set.collate_fn)
    save(data_loader, save_path)

if __name__ == '__main__':
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('-data_path', default='dataset/', type=str)
    parser.add_argument('-data_name', default='debate', type=str)
    parser.add_argument('-mode', default='train', type=str)
    parser.add_argument('-batch_size', default=4, type=int)
    parser.add_argument('-random_seed', type=int, default=0)
    args = parser.parse_args()

    # set random seed
    fix_random_seed(args.random_seed)
    data_builder(args)

