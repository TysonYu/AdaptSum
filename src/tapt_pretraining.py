
import torch
from torch.utils.data import Dataset, DataLoader
from transformers import BartForConditionalGeneration, BartTokenizer, get_linear_schedule_with_warmup
from others.logging import logger
from others.utils import pad_sents, get_mask
from others.optimizer import build_optim
from tqdm import tqdm
import numpy as np
import argparse
import random
import os
from nltk.tokenize import sent_tokenize


def text_infilling(sent, mask_probability=0.05, lamda=3):
    '''
    inputs:
        sent: a sentence string
        mask_probability: probability for masking tokens
        lamda: lamda for poission distribution
    outputs:
        sent: a list of tokens with masked tokens
    '''
    sent = sent.split()
    length = len(sent)
    mask_indices = (np.random.uniform(0, 1, length) < mask_probability) * 1
    span_list = np.random.poisson(lamda, length)  # lamda for poission distribution
    nonzero_idx = np.nonzero(mask_indices)[0]
    for item in nonzero_idx:
        span = min(span_list[item], 5)    # maximum mask 5 continuous tokens
        for i in range(span):
            if item+i >= length:
                continue
            mask_indices[item+i] = 1
    for i in range(length):
        if mask_indices[i] == 1:
            sent[i] = '<mask>'

    # merge the <mask>s to one <mask>
    final_sent = []
    mask_flag = 0
    for word in sent:
        if word != '<mask>':
            mask_flag = 0
            final_sent.append(word)
        else:
            if mask_flag == 0:
                final_sent.append(word)
            mask_flag = 1
    return final_sent

def sent_permutation(sent):
    '''
    inputs:
        sent: a sentence string
    outputs:
        shuffle_sent: a string after sentence permutations
    '''
    # split sentences based on '.'
    splits = sent_tokenize(sent)
    random.shuffle(splits)

    return " ".join(splits)


def add_noise(sents, mask_probability):
    noisy_sent_list = []
    for sent in sents:
        noisy_sent = sent_permutation(sent)
        noisy_sent = text_infilling(noisy_sent, mask_probability)

        noisy_sent = " ".join(noisy_sent)
        noisy_sent_list.append(noisy_sent)

    return noisy_sent_list


class CorpusDataset(Dataset):
    def __init__(self, data_path, denoising_flag=False):
        self.data = []
        with open(data_path, "r", ) as f:
            for i, line in enumerate(f):
                line = line.strip()
                if denoising_flag:
                    line = "denoising: " + line
                self.data.append(line)  # append a list of tokens each time

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        return self.data[idx]


class BartLMTrainer(object):
    def __init__(self, model, dataloader, tokenizer, args, pretrained_model=None):
        self.args = args
        self.model = model
        self.pretrained_model = pretrained_model
        self.optimizer = build_optim(args, model, None, pretrained_model)
        self.dataloader = dataloader
        self.tokenizer = tokenizer
        self.epoch = args.epoch
        self.mask_probability = args.mask_prob
        self.accumulation_steps = args.accum_step
        self.clip = args.clip
        self.domain = args.dm
        self.path = args.path
        if args.recadam:
            if args.max_steps > 0:
                t_total = args.max_steps
                self.epoch = args.max_steps // (len(self.dataloader) // self.accumulation_steps) + 1
            else:
                t_total = len(self.dataloader) // self.accumulation_steps * self.epoch
            self.scheduler = get_linear_schedule_with_warmup(self.optimizer, num_warmup_steps=args.warmup_steps, num_training_steps=t_total)

    def train(self):
        print('Start finetuning BART language model')
        iteration = 0
        for epoch_i in range(self.epoch):
            self.model.train()
            if self.pretrained_model is not None:
                self.pretrained_model.eval()
            print('[ Epoch : {}]'.format(epoch_i))
            loss_list = []
            dist_sum, dist_num = 0.0, 0
            pbar = tqdm(self.dataloader, total=len(self.dataloader))
            for sents in pbar:
                sents = [self.shorten_sent(sent) for sent in sents]
                iteration += 1
                tokenized_sents = self.tokenize(sents)
                decoder_ids = [[self.tokenizer.bos_token_id] + item for item in tokenized_sents]
                label_ids = [item + [self.tokenizer.eos_token_id] for item in tokenized_sents]
                # print("before:")
                # print(sents[0])
                # print("tokenized sents:")
                # print(tokenized_sents[0])
                # sents: a list of sentence, each item inside is a string
                noisy_text = add_noise(sents, self.mask_probability)
                # noisy_text: a list of sentence, each item inside is a string
                # print("after:")
                # print(noisy_text[0])
                inputs_ids = self.tokenize(noisy_text)
                # print("tokenized noisy text:")
                # print(inputs_ids[0])

                # prepare data for training
                mask = torch.tensor(get_mask(inputs_ids, max_len=512)).cuda()
                inputs_ids = torch.tensor(pad_sents(inputs_ids, pad_token=self.tokenizer.pad_token_id, max_len=512)[0]).cuda()
                decoder_ids = torch.tensor(pad_sents(decoder_ids, pad_token=self.tokenizer.pad_token_id, max_len=512)[0]).cuda()
                label_ids = torch.tensor(pad_sents(label_ids, pad_token=-100, max_len=512)[0]).cuda()
                #optimize model
                loss = self.model(input_ids=inputs_ids, attention_mask=mask, decoder_input_ids=decoder_ids, labels=label_ids)[0]
                loss_list.append(loss.item())
                loss = loss / self.accumulation_steps
                loss.backward()
                if self.args.logging_Euclid_dist:
                    dist = torch.sum(torch.abs(torch.cat(
                        [p.view(-1) for n, p in self.model.named_parameters()]) - torch.cat(
                        [p.view(-1) for n, p in self.pretrained_model.named_parameters()])) ** 2).item()

                    dist_sum += dist
                    dist_num += 1

                if iteration % self.accumulation_steps == 0:
                    if self.args.recadam:
                        torch.nn.utils.clip_grad_norm_(self.model.parameters(), 1.0)
                    self.optimizer.step()
                    if self.args.recadam:
                        self.scheduler.step()
                    self.model.zero_grad()
                    loss_list = [np.mean(loss_list)]

                if self.args.logging_Euclid_dist:
#                     pbar.set_description("(Epoch {}) LOSS: {:.6f} Euclid dist: {:.6f} LR: {:.6f}".format(epoch_i, np.mean(loss_list), dist_sum / dist_num, self.scheduler.get_last_lr()[0]))
                    pbar.set_description("(Epoch {}) LOSS: {:.6f} Euclid dist: {:.6f}".format(epoch_i, np.mean(loss_list), dist_sum / dist_num))
                else:
                    pbar.set_description("(Epoch {}) LOSS: {:.6f} LearningRate: {:.10f}".format(epoch_i, np.mean(loss_list), self.optimizer.learning_rate))
                if iteration % args.save_interval == 0:
                    self.save_model(iteration)

    def shorten_sent(self, sent):
        split_sent = sent.split()
        if len(split_sent) > 400:
            sent = ' '.join(split_sent[:400])
        return sent

    def tokenize(self, sents):
        tokenized_text = [self.tokenizer.encode(sent, add_special_tokens=False) for sent in sents]
        return tokenized_text

    def save_model(self, iter_num):
        print("saving model")
        saved_path = os.path.join('TAPT_save/{}_{}.chkpt'.format(args.dm, iter_num))
        torch.save(self.model, saved_path)

if __name__ == "__main__":
    # configuration
    parser = argparse.ArgumentParser()
    parser.add_argument('-visible_gpu', default='1', type=str)
    parser.add_argument('-bsz', type=int, default=4, help="batch size")
    parser.add_argument('-path', type=str, default="", help="data path")
    parser.add_argument('-epoch', type=int, default=10, help="epoch size")
    parser.add_argument('-mask_prob', type=float, default=0.15, help="mask probability")
    parser.add_argument('-dm', type=str, default="", help="domain name")
    parser.add_argument('-random_seed', type=int, default=0)
    parser.add_argument('-save_interval', default=10000, type=int)
    # optimizer configuration
    parser.add_argument('-lr', default=0.05, type=float)
    parser.add_argument('-optim', default='adam', type=str)
    parser.add_argument('-max_grad_norm', default=0, type=float)
    parser.add_argument('-beta1', default=0.9, type=float)
    parser.add_argument('-beta2', default=0.998, type=float)
    parser.add_argument('-warmup_steps', default=10000, type=int)
    parser.add_argument('-decay_method', default='noam', type=str)
    parser.add_argument('-enc_hidden_size', default=768, type=int)
    parser.add_argument('-clip', type=float, default=1.0, help="gradient clip")
    parser.add_argument('-accum_step', type=int, default=10, help="accumulation steps")
    parser.add_argument('-train_from', default='', type=str)
    # using RecAdam
    parser.add_argument("-adam_epsilon", default=1e-8, type=float, help="Epsilon for Adam optimizer.")
    parser.add_argument('-recadam', default=False, action='store_true')
    parser.add_argument("-weight_decay", default=0.0, type=float, help="Weight decay if we apply some.")
    parser.add_argument("-anneal_w", type=float, default=1.0, help="Weight for the annealing function in RecAdam. Default 1.0.")
    parser.add_argument("-anneal_fun", type=str, default='sigmoid', choices=["sigmoid", "linear", 'constant'], help="the type of annealing function in RecAdam. Default sigmoid")
    parser.add_argument("-anneal_t0", type=int, default=1000, help="t0 for the annealing function in RecAdam.")
    parser.add_argument("-anneal_k", type=float, default=0.1, help="k for the annealing function in RecAdam.")
    parser.add_argument("-pretrain_cof", type=float, default=5000.0, help="Coefficient of the quadratic penalty in RecAdam. Default 5000.0.")
    parser.add_argument("-logging_Euclid_dist", action="store_true", help="Whether to log the Euclidean distance between the pretrained model and fine-tuning model")
    parser.add_argument("-max_steps", default=-1, type=int, help="If > 0: set total number of training steps to perform. Override num_train_epochs.")
    parser.add_argument("-model_type", type=str, default="layers")

    args = parser.parse_args()

    # set random seed
    random.seed(args.random_seed)
    np.random.seed(args.random_seed)
    torch.manual_seed(args.random_seed)
    torch.cuda.manual_seed(args.random_seed)
    torch.backends.cudnn.deterministic = True

    # set gpu
    os.environ["CUDA_VISIBLE_DEVICES"] = args.visible_gpu

    print("Loading datasets ...")
    dataset = CorpusDataset(args.path)
    dataloader = DataLoader(dataset=dataset, batch_size=args.bsz, shuffle=True)

    if args.train_from:
        model = torch.load(args.train_from, map_location='cpu')
    else:
        model = BartForConditionalGeneration.from_pretrained('facebook/bart-base')
    model.cuda()

    tokenizer = BartTokenizer.from_pretrained('facebook/bart-base')

    if args.recadam:
        pretrained_model = BartForConditionalGeneration.from_pretrained('facebook/bart-base')
        pretrained_model.cuda()
    else:
        pretrained_model = None

    bart_lm_trainer = BartLMTrainer(model, dataloader, tokenizer, args, pretrained_model)

    bart_lm_trainer.train()