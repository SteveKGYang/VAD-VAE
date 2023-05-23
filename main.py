"""Full training script"""
import logging
import random
import numpy as np
import torch
from torch import nn
from transformers import get_linear_schedule_with_warmup, AdamW
import json
from model import BARTVAEClassifier, BARTDecoderClassifier, BARTVADVAEClassifier, RobertaClassifier
from utils import ErcTextDataset, get_num_classes, get_label_VAD, convert_label_to_VAD, save_latent_params, compute_VAD_pearson_correlation, replace_for_robust_eval
import os
import math
import argparse
import yaml
from torch.nn.utils.rnn import pad_sequence
from sklearn.metrics import f1_score, confusion_matrix, accuracy_score, classification_report, \
    precision_recall_fscore_support, precision_score, recall_score
import torch.cuda.amp.grad_scaler as grad_scaler
import torch.cuda.amp.autocast_mode as autocast_mode
from vae import losses

os.environ["CUDA_VISIBLE_DEVICES"] = "1"


def train(epoch, model, optimizer, scheduler, loss_function, mode, data, batch_size, cuda, alpha, beta, scaler, kl_weights_dict, mi_loss_weight):
    '''The training function with grad scaler.'''
    random.shuffle(data)
    if mode == 'train':
        model.train()
    else:
        model.eval()
    predicts = []
    ground_truth = []
    loss_list = []
    for i in range(0, len(data), batch_size):
        if mode == 'train':
            optimizer.zero_grad()
        bs_data = data[i: min(i+batch_size, len(data))]
        input_data = pad_sequence([torch.LongTensor(item['input_ids']) for item in bs_data], batch_first=True, padding_value=1)
        masks = pad_sequence([torch.LongTensor(item['attention_mask']) for item in bs_data], batch_first=True, padding_value=0)
        '''decoder_input_data = pad_sequence([torch.LongTensor(item['input_ids'])[:-1] for item in bs_data], batch_first=True,
                                  padding_value=1)
        decoder_masks = pad_sequence([torch.LongTensor(item['attention_mask'])[:-1] for item in bs_data], batch_first=True,
                             padding_value=0)
        decoder_labels = pad_sequence([torch.LongTensor(item['input_ids'])[1:] for item in bs_data], batch_first=True,
                                  padding_value=1)'''
        decoder_input_data = pad_sequence([torch.LongTensor(item['current_ids'])[:-1] for item in bs_data],
                                          batch_first=True,
                                          padding_value=1)
        decoder_masks = pad_sequence([torch.LongTensor(item['current_masks'])[:-1] for item in bs_data],
                                     batch_first=True,
                                     padding_value=0)
        decoder_labels = pad_sequence([torch.LongTensor(item['current_ids'])[1:] for item in bs_data], batch_first=True,
                                      padding_value=1)
        labels = torch.LongTensor([item['label'] for item in bs_data])
        #o_labels = [item['label'] for item in bs_data]
        #labels = convert_label_to_VAD(o_labels, label_VAD)
        if cuda:
            input_data = input_data.cuda()
            masks = masks.cuda()
            labels = labels.cuda()
            decoder_input_data = decoder_input_data.cuda()
            decoder_masks = decoder_masks.cuda()
            decoder_labels = decoder_labels.cuda()
        with autocast_mode.autocast():
            outputs, lm_loss, latent_params = model(input_data, masks, decoder_input_data, decoder_masks, decoder_labels, mode, labels)
            #outputs, lm_loss = model(input_data, masks, decoder_input_data, decoder_masks, decoder_labels)
            ce_loss = loss_function(outputs, labels)
            kl_loss = losses.compute_kl_divergence_losses(
                model, latent_params, kl_weights_dict)['total_weighted_kl']
            '''mi_loss = losses.compute_mi_losses(
                model, latent_params, beta=mi_loss_weight)['total_mi']'''
            '''if epoch < 2:
                n_beta = 0.
            else:
                n_beta = beta'''
            loss = ce_loss + alpha*lm_loss + kl_loss
            #loss = ce_loss + alpha*lm_loss
        if mode == 'train':
            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scheduler.step()
            scaler.update()
        ground_truth += labels.cpu().numpy().tolist()
        #ground_truth += o_labels
        predicts += torch.argmax(outputs, dim=1).cpu().numpy().tolist()
        #predicts += compute_predicts(outputs.cpu(), label_VAD)
        loss_list.append(loss.item())
    avg_loss = round(np.sum(loss_list) / len(loss_list), 4)
    avg_accuracy = round(accuracy_score(ground_truth, predicts) * 100, 2)
    weighted_f1 = round(f1_score(ground_truth, predicts, average='weighted') * 100, 2)
    micro_f1 = round(f1_score(ground_truth, predicts, average='micro') * 100, 2)
    macro_f1 = round(f1_score(ground_truth, predicts, average='macro') * 100, 2)
    if mode == 'train':
        print(
            "For epoch {}, train loss:{}, weighted F1 {}, micro F1 {}, macro F1 {}".format(epoch, avg_loss, weighted_f1,
                                                                                        micro_f1,
                                                                                        macro_f1))
    if mode == 'dev':
        print(
            "For epoch {}, dev loss:{}, weighted F1 {}, micro F1 {}, macro F1 {}".format(epoch, avg_loss, weighted_f1,
                                                                                        micro_f1,
                                                                                        macro_f1))
    if mode == 'test':
        print(
            "For epoch {}, test loss:{}, weighted F1 {}, micro F1 {}, macro F1 {}".format(epoch, avg_loss, weighted_f1,
                                                                                        micro_f1,
                                                                                        macro_f1))
        print(f1_score(ground_truth, predicts, average=None))


def train_or_eval_no_scaler(epoch, model, optimizer, scheduler, loss_function, mode, data, batch_size, cuda, alpha, beta,
                    scaler, kl_weights_dict, label_VAD):
    '''The training function without grad scaler.'''
    random.shuffle(data)
    if mode == 'train':
        model.train()
    else:
        model.eval()
    predicts = []
    ground_truth = []
    loss_list = []
    gen_loss_list = []
    latent_param_dict = {}
    for item in kl_weights_dict.keys():
        latent_param_dict[item] = []
    label_records = []
    all_vad_predicts = []
    all_vad_labels = []
    all_mis = []
    sep_all_mis = {"V-A":[], "V-D": [], "A-D":[]}
    for i in range(0, len(data), batch_size):
        if mode == 'train':
            optimizer.zero_grad()
        bs_data = data[i: min(i+batch_size, len(data))]
        input_data = pad_sequence([torch.LongTensor(item['input_ids']) for item in bs_data], batch_first=True, padding_value=1)
        masks = pad_sequence([torch.LongTensor(item['attention_mask']) for item in bs_data], batch_first=True, padding_value=0)
        decoder_input_data = pad_sequence([torch.LongTensor(item['current_ids'])[:-1] for item in bs_data],
                                          batch_first=True,
                                          padding_value=1)
        decoder_masks = pad_sequence([torch.LongTensor(item['current_masks'])[:-1] for item in bs_data],
                                     batch_first=True,
                                     padding_value=0)
        decoder_labels = pad_sequence([torch.LongTensor(item['current_ids'])[1:] for item in bs_data], batch_first=True,
                                      padding_value=1)
        o_labels = [item['label'] for item in bs_data]
        labels = torch.LongTensor(o_labels)
        vad_labels = convert_label_to_VAD(o_labels, label_VAD)
        #o_labels = [item['label'] for item in bs_data]
        #labels = convert_label_to_VAD(o_labels, label_VAD)
        if cuda:
            input_data = input_data.cuda()
            masks = masks.cuda()
            labels = labels.cuda()
            decoder_input_data = decoder_input_data.cuda()
            decoder_masks = decoder_masks.cuda()
            decoder_labels = decoder_labels.cuda()
            vad_labels = vad_labels.cuda()
        outputs, lm_loss, latent_params, vad_loss, vad_predicts = model(input_data, masks, decoder_input_data, decoder_masks, decoder_labels,
                                                mode, vad_labels, labels)
        #outputs = model(input_data, masks)
        ce_loss = loss_function(outputs, labels)
        kl_loss = losses.compute_kl_divergence_losses(
            model, latent_params, kl_weights_dict)['total_weighted_kl']
        if args['mi_loss']:
            mi_result = losses.compute_mi_losses(
                model, latent_params, beta=args["mi_loss_weight"])
            mi_loss, mi_dict = mi_result['total_mi'], mi_result['idv_mi_estimates']
            loss = ce_loss + alpha * lm_loss + kl_loss + vad_loss + mi_loss
        else:
            loss = ce_loss + alpha * lm_loss + kl_loss + vad_loss
        if mode == 'train':
            loss.backward()
            optimizer.step()
            scheduler.step()
            if args['mi_loss']:
                for (latent_pair_name, s_mi_loss) in mi_dict.items():
                    mi_estimator = model.mi_estimators[latent_pair_name]
                    mi_estimator.train()
                    latent_name_1, latent_name_2 = latent_pair_name.split('-')
                    params1 = latent_params[latent_name_1]
                    params2 = latent_params[latent_name_2]
                    e_mi_loss = mi_estimator.learning_loss(
                        params1.z.detach(), params2.z.detach())
                    mi_estimator.optimizer_step(e_mi_loss)
                    mi_estimator.eval()

        ground_truth += labels.cpu().numpy().tolist()
        #ground_truth += o_labels
        predicts += torch.argmax(outputs, dim=1).cpu().numpy().tolist()
        #predicts += compute_predicts(outputs.cpu(), label_VAD)
        loss_list.append(loss.item())
        gen_loss_list.append(lm_loss.item())
        if mode == 'eval':
            for item in latent_params.keys():
                k = torch.cat([latent_params[item].mu.cpu().detach(), torch.exp(latent_params[item].logvar.cpu().detach()),
                               latent_params[item].z.cpu().detach()], dim=-1)
                latent_param_dict[item].append(k)
            label_records += labels.cpu().tolist()
        if mode == 'eval' or mode == 'test':
            all_vad_labels.append(vad_labels.detach().cpu())
            all_vad_predicts.append(vad_predicts.detach().cpu())
            if args['mi_loss']:
                all_mis.append(float(mi_loss.detach().cpu()))
                for (latent_pair_name, s_mi_loss) in mi_dict.items():
                    sep_all_mis[latent_pair_name].append(s_mi_loss)
    '''if args['DATASET'] is 'DailyDialog' and mode is not 'train':
        new_ground_truth = []
        new_predicts = []
        for gt, p in zip(ground_truth, predicts):
            if gt != 0:
                new_ground_truth.append(gt)
                new_predicts.append(p)
        ground_truth = new_ground_truth
        predicts = new_predicts'''
    avg_loss = round(np.sum(loss_list) / len(loss_list), 4)
    gen_avg_loss = round(np.sum(gen_loss_list) / len(gen_loss_list), 4)
    avg_accuracy = round(accuracy_score(ground_truth, predicts) * 100, 2)
    weighted_f1 = round(f1_score(ground_truth, predicts, average='weighted') * 100, 2)
    if args['DATASET'] == 'DailyDialog':
        micro_f1 = round(f1_score(ground_truth, predicts, average='micro', labels=list(range(1, 7))) * 100, 2)
    else:
        micro_f1 = round(f1_score(ground_truth, predicts, average='micro') * 100, 2)
    #micro_f1 = round(f1_score(ground_truth, predicts, average='micro', labels=list(range(1, 7))) * 100, 2)
    macro_f1 = round(f1_score(ground_truth, predicts, average='macro') * 100, 2)
    if mode == 'train':
        print(
            "For epoch {}, train loss:{}, gen loss: {}, weighted F1 {}, micro F1 {}, macro F1 {}".format(epoch, avg_loss, gen_avg_loss,weighted_f1,
                                                                                        micro_f1,
                                                                                        macro_f1))
    if mode == 'dev':
        print(
            "For epoch {}, dev loss:{}, gen loss: {}, weighted F1 {}, micro F1 {}, macro F1 {}".format(epoch, avg_loss, gen_avg_loss, weighted_f1,
                                                                                        micro_f1,
                                                                                        macro_f1))
    if mode == 'test' or mode == 'eval':
        print(
            "For epoch {}, test loss:{}, gen loss: {}, weighted F1 {}, micro F1 {}, macro F1 {}".format(epoch, avg_loss, gen_avg_loss, weighted_f1,
                                                                                        micro_f1,
                                                                                        macro_f1))
        pcv, pca, pcd = compute_VAD_pearson_correlation(torch.cat(all_vad_predicts, dim=0), torch.cat(all_vad_labels, dim=0))
        print("The pearson's coefficient: V:{}, A:{}, D:{}".format(pcv, pca, pcd))
        if args['mi_loss']:
            print("The average V-A MI: {}, average V-D MI: {}, average A-D MI: {}".format(
                sum(sep_all_mis['V-A']) / len(data) / args["mi_loss_weight"],
                sum(sep_all_mis['V-D']) / len(data) / args["mi_loss_weight"],
                sum(sep_all_mis['A-D']) / len(data) / args["mi_loss_weight"]))
            print("The average MI: {}".format(sum(all_mis) / len(data) / args["mi_loss_weight"] / 6))
        if args['DATASET'] == 'DailyDialog':
            print(f1_score(ground_truth, predicts, average=None, labels=list(range(1, 7))))
        else:
            print(f1_score(ground_truth, predicts, average=None))
        if mode == 'eval':
            save_latent_params(latent_param_dict, label_records, args['latent_param_save_path']+args['DATASET'])


def controlled_generate(model, data, batch_size, cuda):
    model.eval()
    for i in range(0, len(data), batch_size):
        bs_data = data[i: min(i + batch_size, len(data))]
        current_ids = [item['current_ids'] for item in bs_data]
        input_data = pad_sequence([torch.LongTensor(item['input_ids']) for item in bs_data], batch_first=True,
                                  padding_value=1)
        max_len = input_data.shape[1]
        masks = pad_sequence([torch.LongTensor(item['attention_mask']) for item in bs_data], batch_first=True,
                             padding_value=0)
        decoder_input_data = pad_sequence([torch.LongTensor(item['current_ids'])[:-1] for item in bs_data],
                                          batch_first=True,
                                          padding_value=1)
        decoder_masks = pad_sequence([torch.LongTensor(item['current_masks'])[:-1] for item in bs_data],
                                     batch_first=True,
                                     padding_value=0)
        #decoder_input_data = torch.zeros(batch_size, 1, dtype=torch.long)
        # o_labels = [item['label'] for item in bs_data]
        # labels = convert_label_to_VAD(o_labels, label_VAD)
        if cuda:
            input_data = input_data.cuda()
            masks = masks.cuda()
            decoder_input_data = decoder_input_data.cuda()
            decoder_masks = decoder_masks.cuda()

        x = model.encoder(input_data, attention_mask=masks)[0]
        x = x[:, 0, :].squeeze(1)

        # params is a dict of {name: namedtuple(z, mu, logvar)} for each
        # discriminator/latent space
        latent_params = model.compute_latent_params(x, None)

        zs = [latent_params[param].z for param in model.latent_variables]
        zs.append(latent_params["content"].z)
        zs = torch.cat(zs, dim=1)
        decoder_hidden = model.z2hidden(zs)
        decoder_outputs = model.decoder(
            input_ids=decoder_input_data,
            attention_mask=decoder_masks,
            encoder_hidden_states=decoder_hidden.unsqueeze(1))
        # print(decoder_outputs.last_hidden_state.shape)
        lm_logits = model.lm_head(decoder_outputs.last_hidden_state)
        s_predict = torch.argmax(lm_logits, dim=-1)
        print(model.tokenizer.convert_ids_to_tokens(current_ids[0]))
        print(model.tokenizer.convert_ids_to_tokens(s_predict[0, :]))
        '''for i in range(max_len):
            decoder_outputs = model.decoder(
                input_ids=decoder_input_data,
                attention_mask=decoder_masks,
                encoder_hidden_states=decoder_hidden.unsqueeze(1))
            # print(decoder_outputs.last_hidden_state.shape)
            lm_logits = model.lm_head(decoder_outputs.last_hidden_state)
            s_predict = torch.argmax(lm_logits[:, -1, :], dim=-1)
            decoder_input_data = torch.cat([decoder_input_data, s_predict.unsqueeze(-1)], dim=-1)
        print(current_ids[0])
        print(decoder_input_data[0, :])'''


def main(CUDA: bool, LR: float, SEED: int, DATASET: str, BATCH_SIZE: int, model_checkpoint: str, bart_model_checkpoint: str,
         speaker_mode: str, num_past_utterances: int, num_future_utterances: int,
         NUM_TRAIN_EPOCHS: int, WEIGHT_DECAY: float, WARMUP_RATIO: float, **kwargs):

    ROOT_DIR = './data'
    NUM_CLASS = get_num_classes(DATASET)
    lr = float(LR)
    label_VAD = get_label_VAD(DATASET)

    ds_train = ErcTextDataset(DATASET=DATASET, SPLIT='train', speaker_mode=speaker_mode,
                              num_past_utterances=num_past_utterances, num_future_utterances=num_future_utterances,
                              model_checkpoint=model_checkpoint,
                              ROOT_DIR=ROOT_DIR, SEED=SEED)
    ds_val = ErcTextDataset(DATASET=DATASET, SPLIT='val', speaker_mode=speaker_mode,
                            num_past_utterances=num_past_utterances, num_future_utterances=num_future_utterances,
                            model_checkpoint=model_checkpoint,
                            ROOT_DIR=ROOT_DIR, SEED=SEED)
    ds_test = ErcTextDataset(DATASET=DATASET, SPLIT='test', speaker_mode=speaker_mode,
                             num_past_utterances=num_past_utterances, num_future_utterances=num_future_utterances,
                             model_checkpoint=model_checkpoint,
                             ROOT_DIR=ROOT_DIR, SEED=SEED)
    tr_data = ds_train.inputs_
    dev_data = ds_val.inputs_
    test_data = ds_test.inputs_
    if args['robust_rate'] != 0.:
        tr_data = replace_for_robust_eval(tr_data, args['robust_rate'], NUM_CLASS)

    latent_variables = ['V', 'A', 'D']
    #latent_variables = []
    #model = RobertaClassifier(model_checkpoint, NUM_CLASS)
    model = BARTVADVAEClassifier(model_checkpoint, bart_model_checkpoint, NUM_CLASS, 64, args['device'], BATCH_SIZE, latent_variables, args['decoder_type'])
    #model = BARTVAEClassifier(model_checkpoint, bart_model_checkpoint, NUM_CLASS, 256, args['device'])
    #model = BARTDecoderClassifier(model_checkpoint, bart_model_checkpoint, NUM_CLASS)
    #predicter = PredcitVADandClassfromLogit(args, label_type='single', label_VAD=label_VAD)
    predicter = None
    if args['mode'] != 'train':
        model.load_state_dict(torch.load(args['model_load_path']))
        model.eval()

    kl_weights_dict = {}
    weight_val = args['kl_weight']
    for lv in latent_variables:
        kl_weights_dict[lv] = weight_val
    kl_weights_dict['content'] = weight_val

    loss_function = nn.CrossEntropyLoss()
    if CUDA:
        model.cuda()

    if args['mode'] == 'train':

        #optimizer = AdamW(model.parameters(), lr=lr, weight_decay=WEIGHT_DECAY)
        optimizer = AdamW(model.parameters(), lr=lr, weight_decay=WEIGHT_DECAY)

        '''Use linear scheduler.'''
        s_total_steps = float(10 * len(ds_train.inputs_)) / BATCH_SIZE
        scheduler = get_linear_schedule_with_warmup(optimizer, int(s_total_steps * WARMUP_RATIO),
                                                    math.ceil(s_total_steps))
        #total_steps = math.ceil(float(args['NUM_TRAIN_EPOCHS'] * len(ds_train.inputs_)) / BATCH_SIZE)

        '''Due to the limitation of computational resources, we use mixed floating point precision.'''
        scaler = grad_scaler.GradScaler()
        # loss_function = nn.MSELoss()
        # loss_function = EMDLoss(args, label_type='single', label_VAD=label_VAD)

        for n in range(NUM_TRAIN_EPOCHS):
            # steps = n * math.ceil(float(len(tr_data)) / BATCH_SIZE)

            train_or_eval_no_scaler(n, model, optimizer, scheduler, loss_function, "train", tr_data, BATCH_SIZE, CUDA,
                            args['alpha'],
                            args['beta'], scaler, kl_weights_dict, label_VAD)
            train_or_eval_no_scaler(n, model, optimizer, scheduler, loss_function, "dev", dev_data, BATCH_SIZE, CUDA,
                            args['alpha'],
                            args['beta'], scaler, kl_weights_dict, label_VAD)
            train_or_eval_no_scaler(n, model, optimizer, scheduler, loss_function, "test", test_data, BATCH_SIZE, CUDA,
                            args['alpha'],
                            args['beta'], scaler, kl_weights_dict, label_VAD)
            torch.save(model.state_dict(), args['model_save_dir'] + "/model_state_dict_" + str(n) + ".pth")
            print("-------------------------------")
    elif args['mode'] == 'generate':
        controlled_generate(model, test_data, BATCH_SIZE, CUDA)
    else:
        train_or_eval_no_scaler(None, model, None, None, loss_function, "eval", test_data, BATCH_SIZE, CUDA,
                                args['alpha'],
                                args['beta'], None, kl_weights_dict, label_VAD)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description='erc RoBERTa text huggingface training')
    parser.add_argument('--DATASET', type=str, default="IEMOCAP")
    parser.add_argument('--CUDA', action='store_true')
    parser.add_argument('--model_checkpoint', type=str, default="roberta-base")
    parser.add_argument('--bart_model_checkpoint', type=str, default="facebook/bart-base")
    parser.add_argument('--speaker_mode', type=str, default="upper")
    parser.add_argument('--num_past_utterances', type=int, default=1000)
    parser.add_argument('--num_future_utterances', type=int, default=1000)
    parser.add_argument('--BATCH_SIZE', type=int, default=4)
    parser.add_argument('--LR', type=float, default=1e-5)
    parser.add_argument('--HP_ONLY_UPTO', type=int, default=10)
    parser.add_argument('--NUM_TRAIN_EPOCHS', type=int, default=10)
    parser.add_argument('--WEIGHT_DECAY', type=float, default=0.01)
    parser.add_argument('--WARMUP_RATIO', type=float, default=0.2)
    parser.add_argument('--HP_N_TRIALS', type=int, default=5)
    parser.add_argument('--OUTPUT-DIR', type=str, default="./output")
    parser.add_argument('--SEED', type=int, default=42)
    parser.add_argument('--alpha', default=0.8,
                        type=float, help='The loss coefficient.')
    parser.add_argument('--beta', default=0.5,
                        type=float, help='The con loss coefficient.')
    parser.add_argument('--kl_weight', default=0.05,
                        type=float, help='The kl loss coefficient.')
    parser.add_argument('--mi_loss_weight', default=0.001,
                        type=float, help='The mi loss coefficient.')
    parser.add_argument('--mode', default="train",
                        type=str, help='Training or eval mode.')
    parser.add_argument('--model_save_dir', default="./model_save_dir/IEMOCAP",
                        type=str, help='Save the trained model in this dir.')
    parser.add_argument('--model_load_path',
                        type=str, help='Load the trained model from here.')
    parser.add_argument('--latent_param_save_path', default="./latent_save_dir/",
                        type=str, help='Save the latent params here.')
    parser.add_argument('--mi_loss', action='store_true', help='Whether add the mi loss.')
    parser.add_argument('--robust_rate', default=0., type=float, help='Replace rate for robustness evaluation')
    parser.add_argument('--decoder_type', type=str, default="LSTM")

    args = parser.parse_args()
    args = vars(args)
    device = torch.device("cuda" if torch.cuda.is_available() and args['CUDA'] is True else "cpu")
    args['n_gpu'] = torch.cuda.device_count()
    args['device'] = device

    '''with open('./train-erc-text.yaml', 'r') as stream:
        args_ = yaml.load(stream, Loader=yaml.FullLoader)

    for key, val in args_.items():
        args[key] = val'''

    logging.info(f"arguments given to {__file__}: {args}")
    main(**args)