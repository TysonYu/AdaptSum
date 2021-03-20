import os
import argparse
import torch
from tqdm import tqdm
from transformers import BartTokenizer
from transformers import BartForConditionalGeneration
from others.logging import init_logger, logger
from others.utils import load, count_parameters, initialize_weights, fix_random_seed
from preprocessing import BartDataset, DataReader
from others.optimizer import build_optim
from trainer import train

def tokenize(data, tokenizer):
    tokenized_text = [tokenizer.encode(i) for i in data]
    return tokenized_text

if __name__ == '__main__':
    # for training
    parser = argparse.ArgumentParser()
    parser.add_argument('-visible_gpu', default='1', type=str)
    parser.add_argument('-log_file', default='./logs/inference/', type=str)
    parser.add_argument('-train_from', default='', type=str)
    parser.add_argument('-random_seed', type=int, default=199744)
    parser.add_argument('-lr', default=0.001, type=float)
    parser.add_argument('-max_grad_norm', default=0, type=float)
    parser.add_argument('-epoch', type=int, default=50)
    parser.add_argument('-saving_path', default='./save/', type=str)
    parser.add_argument('-data_name', default='debate', type=str)
    # for learning, optimizer
    parser.add_argument('-optim', default='adam', type=str)
    parser.add_argument('-beta1', default=0.9, type=float)
    parser.add_argument('-beta2', default=0.998, type=float)
    parser.add_argument('-warmup_steps', default=1000, type=int)
    parser.add_argument('-decay_method', default='noam', type=str)
    parser.add_argument('-enc_hidden_size', default=768, type=int)
    parser.add_argument('-clip', default=1.0, type=float)
    parser.add_argument('-accumulation_steps', default=10, type=int)

    args = parser.parse_args()

    # initial logger
    init_logger(args.log_file+args.data_name+'.log')
    logger.info(args)

    # set gpu
    os.environ["CUDA_VISIBLE_DEVICES"] = args.visible_gpu

    # set random seed
    fix_random_seed(args.random_seed)

    # loading data
    # it's faster to load data from pre_build data
    logger.info('starting to read dataloader')
    data_file_name = './dataset/' + args.data_name + '/testloader.pt'
    train_loader = load(data_file_name)

    # initial tokenizer
    tokenizer = BartTokenizer.from_pretrained('facebook/bart-base')

    # initial model
    logger.info('starting to build model')
    if args.train_from != '':
        logger.info("train from : {}".format(args.train_from))
        # checkpoint = torch.load(args.train_from, map_location='cpu')
        # model.load_state_dict(checkpoint['model'])
        model = torch.load(args.train_from)
    else:
        model = BartForConditionalGeneration.from_pretrained('facebook/bart-base')
    model.cuda()
    model.eval()

    # inference and save model
    save_file_name = './dataset/' + args.data_name + '/summaries'
    summaries = open(save_file_name, 'w')
    outputs = []
    for src_ids, decoder_ids, mask, label_ids in tqdm(train_loader):
        src_ids = src_ids.cuda()
        summary_ids = model.generate(src_ids, num_beams=4, max_length=256, early_stopping=True)
        output = [tokenizer.decode(g, skip_special_tokens=True, clean_up_tokenization_spaces=False) for g in summary_ids]
        outputs += output
    outputs = [i+'\n' for i in outputs]
    print(outputs[:4])
    summaries.writelines(outputs)
    summaries.close()

