import torch
from tqdm import tqdm
import os
from transformers import BartTokenizer, get_linear_schedule_with_warmup
from others.logging import logger
from others.utils import pad_sents, get_mask
from cal_rouge import test_rouge, rouge_results_to_str
from dapt_pretraining import text_infilling, sent_permutation, add_noise

def train(model, training_data, validation_data, optimizer, checkpoint, args, pretrained_model):
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
                temp_F1 = evaluation(model, validation_data, args)
                model.train()
                if temp_F1 > F1:
                    logger.info("saving model")
                    if not os.path.exists(args.saving_path + args.data_name):
                        os.makedirs(args.saving_path + args.data_name)
                    model_name = make_file_name(args, iteration)
#                     checkpoint = {'iteration': iteration, 'settings': args, 'optim': optimizer.optimizer.state_dict(), 'model': model.state_dict()}
                    torch.save(model, model_name)
                    F1 = temp_F1
                else:
                    pass


def multitask_train(model_lm, model_cnn, cnn_train_data, cnn_valid_data, tgtdomain_data, optimizer_lm, optimizer_cnn, checkpoint, args):
    tokenizer = BartTokenizer.from_pretrained('facebook/bart-base')
    ''' Start training '''
    logger.info('Start multitask training')
    iteration = 0
    if args.train_from!= '':
        iteration = checkpoint['iteration']
    cnn_loss = 0
    lm_loss = 0
    F1 = 0
    while iteration < args.max_iter:
        iteration += 1
        model_lm.train()
        model_cnn.train()

        ## cnn news summarization training part
        src_ids, decoder_ids, mask, label_ids = next(cnn_train_data)
        src_ids = src_ids.cuda()
        decoder_ids = decoder_ids.cuda()
        mask = mask.cuda()
        label_ids = label_ids.cuda()

        loss = model_cnn(input_ids=src_ids, attention_mask=mask, decoder_input_ids=decoder_ids, labels=label_ids)[0]
        cnn_loss += loss.item()
        loss = loss / args.accumulation_steps
        # backward
        loss.backward()
        # torch.nn.utils.clip_grad_norm_(model_cnn.parameters(), args.clip)

        ## denoising language modeling part
        sents = next(tgtdomain_data)
        tokenized_sents = [tokenizer.encode(sent, add_special_tokens=False) for sent in sents]
        decoder_ids = [[tokenizer.bos_token_id] + item for item in tokenized_sents]
        label_ids = [item + [tokenizer.eos_token_id] for item in tokenized_sents]

        noisy_text = add_noise(sents, args.mask_prob)
        inputs_ids = [tokenizer.encode(sent, add_special_tokens=False) for sent in noisy_text]

        # prepare data for training
        inputs_ids = torch.tensor(pad_sents(inputs_ids, pad_token=tokenizer.pad_token_id)[0]).cuda()
        mask = torch.tensor(get_mask(inputs_ids)).cuda()
        decoder_ids = torch.tensor(pad_sents(decoder_ids, pad_token=tokenizer.pad_token_id)[0]).cuda()
        label_ids = torch.tensor(pad_sents(label_ids, pad_token=-100)[0]).cuda()

        # optimize model
        loss = model_lm(input_ids=inputs_ids, attention_mask=mask, decoder_input_ids=decoder_ids, labels=label_ids)[0]
        lm_loss += loss.item()
        loss = loss / args.accumulation_steps
        loss.backward()
        # torch.nn.utils.clip_grad_norm_(model_lm.parameters(), args.clip)

        # loss accumulation
        if (iteration+1) % args.accumulation_steps == 0:
            optimizer_lm.step()
            optimizer_cnn.step()
            model_lm.zero_grad()
            model_cnn.zero_grad()
        # write to log file
        if iteration % 20 == 0:
            logger.info("iteration: {} loss_per_word: {:.6f} loss_lm: {:.6f} learning rate lm: {:.9f} learning rate cnn: {:.9f}".format(iteration, cnn_loss/20, lm_loss/20, optimizer_lm.learning_rate, optimizer_cnn.learning_rate))
            cnn_loss = 0
            lm_loss = 0

        if iteration % 50000 == 0:
            # eval_F1 = evaluation(model, cnn_valid_data, args)
            # logger.info("Iteration: {}. F1 score: {:.4f}".format(iteration, eval_F1))
            logger.info("saving model")
            if not os.path.exists(args.saving_path + args.data_name):
                os.makedirs(args.saving_path + args.data_name)
            model_name = make_file_name(args, iteration)
#             checkpoint_1 = {'iteration': iteration, 'settings': args, 'optim': optimizer_lm.optimizer.state_dict(), 'model_lm': model_lm.state_dict()}
#             checkpoint_2 = {'iteration': iteration, 'settings': args, 'optim': optimizer_cnn.optimizer.state_dict(), 'model': model_cnn.state_dict()}
            torch.save(model_lm, model_name[0])
            torch.save(model_cnn, model_name[1])


def evaluation(model, validation_data, args):
    model.eval()
    valid_reference_path = './dataset/' + args.data_name + '/valid.target'
    valid_data = open(valid_reference_path,'r')
    valid_list = valid_data.readlines()
    valid_list = [i.strip('\n') for i in valid_list]
    # inference
    tokenizer = BartTokenizer.from_pretrained('facebook/bart-base')
    outputs = []
    for src_ids, decoder_ids, mask, label_ids in tqdm(validation_data):
        src_ids = src_ids.cuda()
        summary_ids = model.generate(src_ids, num_beams=4, max_length=256, early_stopping=True)
        output = [tokenizer.decode(g, skip_special_tokens=True, clean_up_tokenization_spaces=False) for g in summary_ids]
        outputs += output
    # calculate rouge
    final_results = test_rouge(outputs, valid_list, args.process_num)
    R1_F1 = final_results["rouge_1_f_score"] * 100
    logger.info('[ Validation ]')
    logger.info(rouge_results_to_str(final_results))
    return R1_F1

def make_file_name(args, iteration):
    model_name = args.saving_path + "{}/{}_75%sample.chkpt".format(args.data_name,iteration,args.percentage)
    if args.pre_trained_lm != '':
        # model_name = args.saving_path + "{}/{}_{}%DAPT_REG_EPOCH_1.chkpt".format(args.data_name,iteration,args.percentage)
        model_name = args.saving_path + "{}/{}_{}%DAPT_reg_160000.chkpt".format(args.data_name,iteration,args.percentage)
    if args.pre_trained_src:
        model_name = args.saving_path + "{}/{}_{}%cnn_pretrain_400000.chkpt".format(args.data_name,iteration,args.percentage)
#         model_name = args.saving_path + "{}/{}_{}%_pre_trained_src.chkpt".format(args.data_name,iteration,args.percentage)
    if args.mtl:
        model_name_1 = args.saving_path + "{}/{}_mtl_pre_trained_lm.chkpt".format('social_media_short',iteration)
        model_name_2 = args.saving_path + "{}/{}_mtl_pre_trained_src.chkpt".format('social_media_short',iteration)
        model_name = [model_name_1, model_name_2]
    return model_name

def evaluation_loss(model, validation_data, args):
    model.eval()
    total_loss = 0
    for src_ids, decoder_ids, mask, label_ids in tqdm(validation_data):
        src_ids = src_ids.cuda()
        decoder_ids = decoder_ids.cuda()
        mask = mask.cuda()
        label_ids = label_ids.cuda()
        loss = model(input_ids=src_ids, attention_mask=mask, decoder_input_ids=decoder_ids, labels=label_ids)[0]
        total_loss += loss.item()
        loss = None
    logger.info(total_loss)
    return total_loss