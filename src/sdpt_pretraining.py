import os
import argparse
import torch
from transformers import BartForConditionalGeneration, get_linear_schedule_with_warmup
import numpy as np
from preprocessing import BartDataset, DataReader
from others.logging import init_logger, logger
from others.utils import load, fix_random_seed
from others.optimizer import build_optim



def load_dataloader(args):
    train_file_name = './dataset/' + args.data_name + '/trainloader.pt'
    train_loader = load(train_file_name)
    logger.info('train loader has {} samples'.format(len(train_loader.dataset)))
    return train_loader

def train(model, training_data, optimizer, checkpoint, args, pretrained_model):
    ''' Start training '''
    if args.logging_Euclid_dist:
        t_total = len(training_data) // args.accumulation_steps * 10
        scheduler = get_linear_schedule_with_warmup(optimizer, num_warmup_steps=args.warmup_steps, num_training_steps=t_total)
    logger.info('Start training')
    iteration = 0
    if args.break_point_continue:
        iteration = checkpoint['iteration']
    total_loss = 0
    F1 = 0
    for epoch_i in range(args.epoch):
        logger.info('[ Epoch : {}]'.format(epoch_i))
        dist_sum, dist_num = 0.0, 0
        # training part
        model.train()
        for src_ids, decoder_ids, mask, label_ids in training_data:
            iteration += 1
            src_ids = src_ids.cuda()
            decoder_ids = decoder_ids.cuda()
            mask = mask.cuda()
            label_ids = label_ids.cuda()
            # forward
            # optimizer.optimizer.zero_grad()
            loss = model(input_ids=src_ids, attention_mask=mask, decoder_input_ids=decoder_ids, labels=label_ids)[0]
            total_loss += loss.item()
            loss = loss / args.accumulation_steps
            # backward
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), args.clip)
            # loss accumulation
            if (iteration+1) % args.accumulation_steps == 0:
                optimizer.step()
                if args.recadam:
                    scheduler.step()
                model.zero_grad()

            if args.logging_Euclid_dist:
                dist = torch.sum(torch.abs(torch.cat(
                    [p.view(-1) for n, p in model.named_parameters()]) - torch.cat(
                    [p.view(-1) for n, p in pretrained_model.named_parameters()])) ** 2).item()

                dist_sum += dist
                dist_num += 1
            # write to log file
            if iteration % 20 == 0:
                if args.logging_Euclid_dist:
                    logger.info("iteration: {} loss_per_word: {:4f} Euclid dist: {:.6f}".format(iteration, total_loss/20, dist_sum / dist_num))
                else:
                    logger.info("iteration: {} loss_per_word: {:4f} learning rate: {:4f} ".format(iteration, total_loss/20, optimizer.learning_rate))
                total_loss = 0
            # save model
            if iteration % args.save_interval == 0 and iteration > args.start_to_save_iter:
                print('=====Saving checkpoint=====')
                model_name = args.saving_path + "/{}_{}.chkpt".format(args.data_name, iteration)
                torch.save(model, model_name)
            else:
                pass

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
    parser.add_argument('-minor_data', action='store_true')
    parser.add_argument('-pre_trained_lm', default='', type=str)
    parser.add_argument('-pre_trained_src', action='store_true')
    parser.add_argument('-break_point_continue', action='store_true')
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
    parser.add_argument('-start_to_save_iter', default=3000, type=int)
    parser.add_argument('-save_interval', default=10000, type=int)
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
    if not os.path.exists(args.log_file + args.data_name):
        os.makedirs(args.log_file + args.data_name)
    init_logger(args.log_file + args.data_name + '/DAPT_pretraining.log')
    logger.info(args)
    # set gpu
    os.environ["CUDA_VISIBLE_DEVICES"] = args.visible_gpu
    # set random seed
    fix_random_seed(args.random_seed)

    # loading data
    # it's faster to load data from pre_build data
    logger.info('starting to read dataloader')
    train_loader = load_dataloader(args)

    # initial model and optimizer
    logger.info('starting to build model')
    model = BartForConditionalGeneration.from_pretrained('facebook/bart-base')
    model.cuda()
    checkpoint = None
    optim = build_optim(args, model, None, model)
    pretrained_model = BartForConditionalGeneration.from_pretrained('facebook/bart-base')
    if args.recadam:
        pretrained_model.cuda()
        optim = build_optim(args, model, None, pretrained_model)

    # training
    train(model, train_loader, optim, checkpoint, args, pretrained_model)
