import os
import argparse
import torch
from torch.utils.data import DataLoader
from transformers import BartForConditionalGeneration
from others.logging import init_logger, logger
from others.utils import load, count_parameters, initialize_weights, batch_generator, fix_random_seed
from preprocessing import BartDataset, DataReader
from others.optimizer import build_optim
from trainer import train, multitask_train
from dapt_pretraining import CorpusDataset
import random
import numpy as np

def make_log_file_name(args):
    if args.pre_trained_lm != '':
        log_file_name = args.log_file + args.data_name + '/train_'  + args.pre_trained_lm.split('/')[-1][:-3] + '_' + args.percentage + '%_pretrain_lm.log'
    elif args.pre_trained_src:
        log_file_name = args.log_file + args.data_name + '/train_' + args.train_from.split('/')[-1][:-6] + '_' + args.percentage + '_' + '%_pretrain_src.log'
    else:
        log_file_name = args.log_file + args.data_name + '/train_' + args.percentage + '%.log'
    return log_file_name

def load_dataloader(args):
    train_file_name = './dataset/' + args.data_name + '/trainloader.pt'
    train_loader = load(train_file_name)
    valid_file_name = './dataset/' + args.data_name + '/validloader.pt'
    valid_loader = load(valid_file_name)
    logger.info('train loader has {} samples'.format(len(train_loader.dataset)))
    return train_loader, valid_loader

def load_dataloader_for_domain_corpus(args):
    corpus_dataset = CorpusDataset(args.corpus_path, denoising_flag=True)
    dataloader = DataLoader(dataset=corpus_dataset, batch_size=args.bsz, shuffle=True)
    data_generator = batch_generator(dataloader)
    return data_generator

def load_model(args):
    model = BartForConditionalGeneration.from_pretrained('facebook/bart-base')
    if args.pre_trained_lm != '':
        model = torch.load(args.pre_trained_lm, map_location='cpu')
    # load from saved model
    if args.train_from != '':
        logger.info("train from : {}".format(args.train_from))
        if "mtl_pre_trained_lm" in args.train_from:
            checkpoint = torch.load(args.train_from, map_location='cpu')
            model.load_state_dict(checkpoint['model_lm'])
        elif "xsum" in args.train_from:
            checkpoint = None
            model = BartForConditionalGeneration.from_pretrained('VictorSanh/bart-base-finetuned-xsum')
        else:
            checkpoint = torch.load(args.train_from, map_location='cpu')
            model.load_state_dict(checkpoint['model'])
    if args.train_from == '':
        checkpoint = None
    if args.mtl:
        model_lm = BartForConditionalGeneration.from_pretrained('facebook/bart-base')
        if args.pre_trained_lm != '':
            model_lm = torch.load(args.pre_trained_lm, map_location='cpu')
        model_cnn = BartForConditionalGeneration.from_pretrained('facebook/bart-base')
        # shared part
        model_cnn.model.shared = model_lm.model.shared
        # encoder part
        model_cnn.model.encoder = model_lm.model.encoder

        print("dont share decoder!")
        model = None
        return model_lm, model_cnn, checkpoint
    return model, checkpoint

if __name__ == '__main__':
    # for training
    parser = argparse.ArgumentParser()
    parser.add_argument('-visible_gpu', default='1', type=str)
    parser.add_argument('-log_file', default='./logs/', type=str)
    parser.add_argument('-train_from', default='', type=str)
    parser.add_argument('-random_seed', type=int, default=0)
    parser.add_argument('-lr', default=0.05, type=float)
    parser.add_argument('-max_grad_norm', default=0, type=float)
    parser.add_argument('-epoch', type=int, default=50)
    parser.add_argument('-max_iter', type=int, default=800000)
    parser.add_argument('-saving_path', default='./save/', type=str)
    parser.add_argument('-data_name', default='debate', type=str)
    parser.add_argument('-pre_trained_lm', default='', type=str)
    parser.add_argument('-pre_trained_src', action='store_true')
    parser.add_argument('-break_point_continue', action='store_true')
    parser.add_argument('-percentage', default='100', type=str)
    parser.add_argument('-corpus_path', type=str, default="", help="target domain corpus path")
    parser.add_argument('-mask_prob', type=float, default=0.15, help="mask probability")
    # for learning, optimizer
    parser.add_argument('-mtl', action='store_true', help='multitask learning')
    parser.add_argument('-optim', default='adam', type=str)
    parser.add_argument('-beta1', default=0.9, type=float)
    parser.add_argument('-beta2', default=0.998, type=float)
    parser.add_argument('-warmup_steps', default=1000, type=int)
    parser.add_argument('-decay_method', default='noam', type=str)
    parser.add_argument('-enc_hidden_size', default=768, type=int)
    parser.add_argument('-clip', default=1.0, type=float)
    parser.add_argument('-accumulation_steps', default=10, type=int)
    parser.add_argument('-bsz', default=4, type=int, help='batch size')
    # for evaluation
    parser.add_argument('-process_num', default=4, type=int)
    parser.add_argument('-save_interval', default=100, type=int)
    parser.add_argument('-start_to_save_iter', default=100, type=int)
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

    # initial logger
    init_logger(make_log_file_name(args))
    logger.info(args)
    # set gpu
    os.environ["CUDA_VISIBLE_DEVICES"] = args.visible_gpu
    # set random seed
    fix_random_seed(args.random_seed)

    # loading data
    # it's faster to load data from pre_build data
    logger.info('starting to read dataloader')
    train_loader, valid_loader = load_dataloader(args)

    # initial model and optimizer
    logger.info('starting to build model')
    if not args.mtl:
        model, checkpoint = load_model(args)
        model.cuda()
        optim = build_optim(args, model, None, model)
        pretrained_model = BartForConditionalGeneration.from_pretrained('facebook/bart-base')
        if args.recadam:
            pretrained_model.cuda()
            optim = build_optim(args, model, None, pretrained_model)
    else:
        model_lm, model_cnn, checkpoint = load_model(args)
        model_lm.cuda()
        model_cnn.cuda()
        optim_lm = build_optim(args, model_lm, checkpoint)
        optim_cnn = build_optim(args, model_cnn, checkpoint)

    # training
    if args.mtl:
        assert args.data_name == "cnn_dm" or args.data_name == "xsum"
        tgtdomain_data = load_dataloader_for_domain_corpus(args)
        cnn_train_data = batch_generator(train_loader)
        cnn_valid_data = batch_generator(valid_loader)
        multitask_train(model_lm, model_cnn, cnn_train_data, cnn_valid_data, tgtdomain_data, optim_lm, optim_cnn, checkpoint, args)
    else:
        train(model, train_loader, valid_loader, optim, checkpoint, args, pretrained_model)
