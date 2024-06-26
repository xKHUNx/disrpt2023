import logging
import os
import json
import pickle
import math
import random
import time
import datetime
from tqdm import tqdm, trange

import argparse
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data.sampler import RandomSampler, Sampler, SequentialSampler
from torch.utils.data.dataloader import DataLoader

from transformers.optimization import AdamW, get_linear_schedule_with_warmup
from transformers.models.roberta import RobertaConfig, RobertaTokenizer
from transformers.models.bert import BertConfig, BertTokenizer
from transformers.models.electra import ElectraConfig, ElectraTokenizer
# from transformers.models.xlm_roberta import XLMRobertaConfig, XLMRobertaTokenizer
from transformers import XLMRobertaConfig, XLMRobertaTokenizer
from transformers import CamembertConfig, CamembertTokenizer

from utils import *
from task_dataset import *
from models import *
from seg_eval import get_scores

# set logger, print to console and write to file
logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)
BASIC_FORMAT = "%(asctime)s:%(levelname)s: %(message)s"
DATE_FORMAT = '%Y-%m-%d %H:%M:%S'
formatter = logging.Formatter(BASIC_FORMAT, DATE_FORMAT)
chlr = logging.StreamHandler()
chlr.setFormatter(formatter)
logger.addHandler(chlr)

# for output
dt = datetime.datetime.now()
TIME_CHECKPOINT_DIR = "checkpoint_{}-{}-{}_{}{}".format(dt.year, dt.month, dt.day, dt.hour, dt.minute)
PREFIX_CHECKPOINT_DIR = "checkpoint"

import warnings
warnings.filterwarnings('ignore')

os.environ['CUDA_LAUNCH_BLOCKING'] = "1"


def get_argparse():
    parser = argparse.ArgumentParser()

    # path
    parser.add_argument("--data_dir", default="data/dataset", type=str)
    parser.add_argument("--dataset", default="eng.rst.gum", type=str, help="pdtb2, pdtb3")
    parser.add_argument("--output_dir", default="data/result", type=str)
    parser.add_argument("--feature_size", default=0, type=int)
    parser.add_argument("--trained_model", default="eng.dep.scidtb", type=str)

    # for training
    parser.add_argument("--model_type", default="base", type=str, help="roberta-bilstm-crf")
    parser.add_argument("--encoder_type", default="roberta", type=str, help="roberta, ...")
    parser.add_argument("--do_train", default=False, action="store_true")
    parser.add_argument("--do_dev", default=False, action="store_true")
    parser.add_argument("--do_test", default=False, action="store_true")
    parser.add_argument("--do_freeze", default=False, action="store_true")
    parser.add_argument("--do_adv", default=False, action="store_true")
    parser.add_argument("--run_plus", default=False, action="store_true")
    parser.add_argument("--bagging", default=False, action="store_true")
    parser.add_argument("--ratio", default=0.8, type=float)

    parser.add_argument("--bag_nb", default=0, type=int)
    parser.add_argument("--train_batch_size", default=8, type=int)
    parser.add_argument("--eval_batch_size", default=24, type=int)
    parser.add_argument("--max_seq_length", default=512, type=int)
    parser.add_argument("--num_train_epochs", default=10, type=int, help="training epoch")
    parser.add_argument("--learning_rate", default=3e-5, type=float, help="learning rate")
    parser.add_argument("--dropout", default=0.1, type=float)
    parser.add_argument("--max_grad_norm", default=2.0, type=float)
    parser.add_argument("--weight_decay", default=0.1, type=float)
    parser.add_argument("--warmup_ratio", default=0.06, type=float)
    parser.add_argument("--seed", default=106524, type=int, help="random seed")
    parser.add_argument("--extra_feat_dim", default=400, type=int)

    parser.add_argument("--pos1_convert", default="sequence", type=str, help="one-hot or sequence")
    parser.add_argument("--pos2_convert", default="sequence", type=str, help="one-hot or sequence")
    return parser


def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def get_dataloader(dataset, args, mode="train"):
    print("  {} dataset length: ".format(mode), len(dataset))
    if mode.lower() == "train":
        # here, if you want to use random sampler, then we cannot map the result one-by-one in the evaluate step
        # but, since it's training stage, I think it's ok, you can input the training file again under the test stage.
        sampler = RandomSampler(dataset)
        batch_size = args.train_batch_size
    else:
        sampler = SequentialSampler(dataset)
        batch_size = args.eval_batch_size
    data_loader = DataLoader(
        dataset=dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=0,
        pin_memory=True,
        sampler=sampler
    )
    return data_loader


def get_optimizer(model, args, num_training_steps):
    specific_params = []
    no_deday = ["bias", "LayerNorm.weigh"]
    optimizer_grouped_parameters = [
        {
            "params": [p for n, p in model.named_parameters() if not any(nd in n for nd in no_deday)],
            "weight_decay": args.weight_decay
        },
        {
            "params": [p for n, p in model.named_parameters() if any(nd in n for nd in no_deday)],
            "weight_decay": 0.0
        }
    ]
    optimizer = AdamW(optimizer_grouped_parameters, lr=args.learning_rate)
    scheduler = get_linear_schedule_with_warmup(
        optimizer, num_warmup_steps=int(num_training_steps * args.warmup_ratio),
        num_training_steps=num_training_steps
    )
    return optimizer, scheduler


def train(model, args, tokenizer, train_dataloader, dev_dataloader=None, test_dataloader=None):
    # 1.prepare
    t_total = int(len(train_dataloader) * args.num_train_epochs)
    print_step = int(len(train_dataloader) // 4) + 1
    num_train_epochs = args.num_train_epochs
    optimizer, scheduler = get_optimizer(model, args, t_total)
    logger.info(" ***** Running training *****")
    logger.info("  Num examples = %d", len(train_dataloader.dataset))
    logger.info("  Num Epochs = %d", num_train_epochs)
    logger.info("  Batch size per device = %d", args.train_batch_size)
    logger.info("  Total optimization steps = %d", t_total)

    # 2.train
    global_step = 0
    tr_loss = 0.0
    logging_loss = 0.0
    best_dev = 0.0
    train_iterator = trange(1, int(num_train_epochs) + 1, desc="Epoch")
    for epoch in train_iterator:
        epoch_iterator = tqdm(train_dataloader, desc="Iteration")
        model.train()
        model.zero_grad()
        for step, batch in enumerate(epoch_iterator):
            batch = tuple(t.to(args.device) for t in batch)
            inputs = {
                "input_ids": batch[0],
                "attention_mask": batch[1],
                "labels": batch[2],
                "flag": "Train",
            }

            outputs = model(**inputs)
            loss = outputs[0]
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), args.max_grad_norm)
            optimizer.step()
            scheduler.step()
            optimizer.zero_grad()

            global_step += 1
            logging_loss = loss.item() * args.train_batch_size
            tr_loss += logging_loss

            if global_step % print_step == 0:
                print(" global_step=%d, cur loss=%.4f, global avg loss=%.4f" % (
                        global_step, logging_loss, tr_loss / global_step)
                )

        # 3. evaluate and save
        model.eval()
        if False and train_dataloader is not None:
            #score_dict = evaluate(model, args, train_dataloader, tokenizer, epoch, desc="train")
            score_dict = evaluate_new(model, args, train_dataloader, tokenizer, epoch, desc="train")
            print("\nTrain: Epoch=%d, F1=%.4f\n"%(epoch, score_dict["f_score"]))
        if dev_dataloader is not None:
            #score_dict = evaluate(model, args, dev_dataloader, tokenizer, epoch, desc="dev")
            if args.bagging:
                score_dict = evaluate_new(model, args, dev_dataloader, tokenizer, epoch, desc="dev", nb4bag=args.bag_nb)
            else:
                score_dict = evaluate_new(model, args, dev_dataloader, tokenizer, epoch, desc="dev")
            if score_dict["f_score"] > best_dev:
                best_dev = score_dict["f_score"]
            print("\nDev: Epoch=%d, F1=%.4f\n"%(epoch, score_dict["f_score"]))
        if test_dataloader is not None:
            #score_dict = evaluate(model, args, test_dataloader, tokenizer, epoch, desc="test")
            if args.bagging:
                score_dict = evaluate_new(model, args, test_dataloader, tokenizer, epoch, desc="test", nb4bag=args.bag_nb)
            else:
                score_dict = evaluate_new(model, args, test_dataloader, tokenizer, epoch, desc="test")
            print("\nTest: Epoch=%d, F1=%.4f\n"%(epoch, score_dict["f_score"]))
    output_dir = os.path.join(args.output_dir, args.dataset + "_"+ args.model_type)
    output_dir = os.path.join(output_dir, f"{PREFIX_CHECKPOINT_DIR}_{epoch}")
    os.makedirs(output_dir, exist_ok=True)
    torch.save(model.state_dict(), os.path.join(output_dir, "pytorch_model.bin"))
    print("\nBest F1 on dev: %.4f"%(best_dev))


def train_plus(model, args, tokenizer, train_dataloader, dev_dataloader=None, test_dataloader=None):
    # 1.prepare
    t_total = int(len(train_dataloader) * args.num_train_epochs)
    print_step = int(len(train_dataloader) // 4) + 1
    num_train_epochs = args.num_train_epochs
    optimizer, scheduler = get_optimizer(model, args, t_total)
    extra_feat_dim = args.extra_feat_dim

    logger.info(" ***** Running training *****")
    logger.info("  Num examples = %d", len(train_dataloader.dataset))
    logger.info("  Num Epochs = %d", num_train_epochs)
    logger.info("  Batch size per device = %d", args.train_batch_size)
    logger.info("  Total optimization steps = %d", t_total)

    # 2.train
    global_step = 0
    tr_loss = 0.0
    logging_loss = 0.0
    best_dev = 0.0
    train_iterator = trange(1, int(num_train_epochs) + 1, desc="Epoch")
    for epoch in train_iterator:
        epoch_iterator = tqdm(train_dataloader, desc="Iteration")
        model.train()
        model.zero_grad()
        for step, batch in enumerate(epoch_iterator):
            batch = tuple(t.to(args.device) for t in batch)
            inputs = {
                "input_ids": batch[0],
                "attention_mask": batch[1],
                "labels": batch[2],
                "flag": "Train",
                "pos1_ids": batch[4],
                "pos2_ids": batch[5],
                "ft_embeds": batch[6]
            }

            outputs = model(**inputs)
            loss = outputs[0]
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), args.max_grad_norm)
            optimizer.step()
            scheduler.step()
            optimizer.zero_grad()

            global_step += 1
            logging_loss = loss.item() * args.train_batch_size
            tr_loss += logging_loss

            if global_step % print_step == 0:
                print(" global_step=%d, cur loss=%.4f, global avg loss=%.4f" % (
                    global_step, logging_loss, tr_loss / global_step)
                      )

        # 3. evaluate and save
        model.eval()
        if False and train_dataloader is not None:
            score_dict = evaluate_new(model, args, train_dataloader, tokenizer, epoch, desc="train")
            print("\nTrain: Epoch=%d, F1=%.4f\n" % (epoch, score_dict["f_score"]))
        if dev_dataloader is not None:
            score_dict = evaluate_new(model, args, dev_dataloader, tokenizer, epoch, desc="dev")
            if score_dict["f_score"] > best_dev:
                best_dev = score_dict["f_score"]
            print("\nDev: Epoch=%d, F1=%.4f\n" % (epoch, score_dict["f_score"]))
        if test_dataloader is not None:
            score_dict = evaluate_new(model, args, test_dataloader, tokenizer, epoch, desc="test")
            print("\nTest: Epoch=%d, F1=%.4f\n" % (epoch, score_dict["f_score"]))
        output_dir = os.path.join(args.output_dir, TIME_CHECKPOINT_DIR)
        output_dir = os.path.join(output_dir, f"{PREFIX_CHECKPOINT_DIR}_{epoch}")
        # os.makedirs(output_dir, exist_ok=True)
        # torch.save(model.state_dict(), os.path.join(output_dir, "pytorch_model.bin"))
    print("\nBest F1 on dev: %.4f" % (best_dev))

def evaluate(model, args, dataloader, tokenizer, epoch, desc="dev", write_file=False):
    all_input_ids = None
    all_attention_mask = None
    all_label_ids = None
    all_pred_ids = None
    for batch in tqdm(dataloader, desc=desc):
        batch = tuple(t.to(args.device) for t in batch)
        inputs = {
            "input_ids": batch[0],
            "attention_mask": batch[1],
            "labels": batch[2],
            "flag": "Eval"
        }
        with torch.no_grad():
            outputs = model(**inputs)
            preds = outputs[0]

        input_ids = batch[0].detach().cpu().numpy()
        attention_mask = batch[1].detach().cpu().numpy()
        label_ids = batch[2].detach().cpu().numpy()
        pred_ids = preds.detach().cpu().numpy()
        if all_input_ids is None:
            all_input_ids = input_ids
            all_attention_mask = attention_mask
            all_label_ids = label_ids
            all_pred_ids = pred_ids
        else:
            all_input_ids = np.append(all_input_ids, input_ids, axis=0)
            all_attention_mask = np.append(all_attention_mask, attention_mask, axis=0)
            all_label_ids = np.append(all_label_ids, label_ids, axis=0)
            all_pred_ids = np.append(all_pred_ids, pred_ids, axis=0)

    ## evaluation
    if desc == "train":
        gold_file = args.train_data_file.replace(".json", ".tok")
    elif desc == "dev":
        gold_file = args.dev_data_file.replace(".json", ".tok")
    elif desc == "test":
        gold_file = args.test_data_file.replace(".json", ".tok")
    # print(all_pred_ids)
    pred_file = seg_preds_to_file(all_input_ids, all_pred_ids, all_attention_mask, args.tokenizer, args.label_list, gold_file)
    score_dict = get_scores(gold_file, pred_file)

    return score_dict


def evaluate_new(model, args, dataloader, tokenizer, epoch, desc="dev", write_file=False, nb4bag=None):
    all_input_ids = None
    all_attention_mask = None
    all_label_ids = None
    all_pred_ids = None
    for batch in tqdm(dataloader, desc=desc):
        batch = tuple(t.to(args.device) for t in batch)
        if args.run_plus:
            inputs = {
                "input_ids": batch[0],
                "attention_mask": batch[1],
                "labels": batch[2],
                "flag": "Eval",
                "pos1_ids": batch[4],
                "pos2_ids": batch[5],
                "ft_embeds": batch[6]
            }
        else:
            inputs = {
                "input_ids": batch[0],
                "attention_mask": batch[1],
                "labels": batch[2],
                "flag": "Eval"
            }
        with torch.no_grad():
            outputs = model(**inputs)
            preds = outputs[0]

        input_ids = batch[0].detach().cpu().numpy()
        attention_mask = batch[1].detach().cpu().numpy()
        label_ids = batch[2].detach().cpu().numpy()
        og_tok_idxs = batch[3].detach().cpu().numpy()
        if args.model_type == "bilstm+crf":
            max_len = args.max_seq_length
            padded_list = []
            for l in preds:
                padded_l = l + [0] * (max_len - len(l))
                padded_list.append(padded_l)

            pred_ids = np.array(padded_list)

        else:
            pred_ids = preds.detach().cpu().numpy()

        if all_input_ids is None:
            all_input_ids = input_ids
            all_attention_mask = attention_mask
            all_label_ids = label_ids
            all_pred_ids = pred_ids
            all_tok_idxs = og_tok_idxs
        else:
            all_input_ids = np.append(all_input_ids, input_ids, axis=0)
            all_attention_mask = np.append(all_attention_mask, attention_mask, axis=0)
            all_label_ids = np.append(all_label_ids, label_ids, axis=0)
            all_pred_ids = np.append(all_pred_ids, pred_ids, axis=0)
            all_tok_idxs = np.append(all_tok_idxs, og_tok_idxs, axis=0)
    ## evaluation
    if desc == "train":
        gold_file = args.train_data_file.replace(".json", ".tok")
    elif desc == "dev":
        gold_file = args.dev_data_file.replace(".json", ".tok")
    elif desc == "test":
        gold_file = args.test_data_file.replace(".json", ".tok")
    #print(gold_file)
    if nb4bag is None:
        pred_file = seg_preds_to_file_new2(all_input_ids, all_pred_ids, all_attention_mask, all_tok_idxs, args.tokenizer, args.label_list, gold_file)
    else:
        pred_file = seg_preds_to_file_new2(all_input_ids, all_pred_ids, all_attention_mask, all_tok_idxs, args.tokenizer, args.label_list, gold_file, nb4bag)

    score_dict = get_scores(gold_file, pred_file)
    return score_dict

def inference(model, args, dataloader, tokenizer, epoch, desc="dev", write_file=False, nb4bag=None):
    all_input_ids = None
    all_attention_mask = None
    # all_label_ids = None
    all_pred_ids = None

    all_preds = []

    for batch in tqdm(dataloader, desc=desc):
        print(batch)
        batch = tuple(t.to(args.device) if not isinstance(t, str) else None for t in batch if not isinstance(t, str))
        if args.run_plus:
            inputs = {
                "input_ids": batch[0],
                "attention_mask": batch[1],
                "labels": batch[2],
                "flag": "Eval",
                "pos1_ids": batch[4],
                "pos2_ids": batch[5],
                "ft_embeds": batch[6]
            }
        else:
            inputs = {
                "input_ids": batch[0],
                "attention_mask": batch[1],
                "labels": batch[2],
                "flag": "Eval"
            }
        with torch.no_grad():
            outputs = model(**inputs)
            preds = outputs[0]

            all_preds.append(preds)

        input_ids = batch[0].detach().cpu().numpy()
        attention_mask = batch[1].detach().cpu().numpy()
        label_ids = batch[2].detach().cpu().numpy()
        og_tok_idxs = batch[3].detach().cpu().numpy()
        if args.model_type == "bilstm+crf":
            max_len = args.max_seq_length
            padded_list = []
            for l in preds:
                padded_l = l + [0] * (max_len - len(l))
                padded_list.append(padded_l)

            pred_ids = np.array(padded_list)

        else:
            pred_ids = preds.detach().cpu().numpy()
        
        if all_input_ids is None:
            all_input_ids = input_ids
            all_attention_mask = attention_mask
            all_label_ids = label_ids
            all_pred_ids = pred_ids
            all_tok_idxs = og_tok_idxs
        else:
            all_input_ids = np.append(all_input_ids, input_ids, axis=0)
            all_attention_mask = np.append(all_attention_mask, attention_mask, axis=0)
            all_label_ids = np.append(all_label_ids, label_ids, axis=0)
            all_pred_ids = np.append(all_pred_ids, pred_ids, axis=0)
            all_tok_idxs = np.append(all_tok_idxs, og_tok_idxs, axis=0)

    return all_preds, input_sents
        

def main():
    args = get_argparse().parse_args()

    print(args)

    if torch.cuda.is_available():
        args.n_gpu = 1
        device = torch.device("cuda:0")
        print("Device:", device)
    else:
        device = torch.device("cpu")
        args.n_gpu = 0
    args.device = device
    logger.info("Training/evaluation parameters %s", args)
    set_seed(args.seed)

    # 1.prepare pretrained path
    lang_type = args.dataset.split(".")[0]
    args.lang_type = lang_type
    print(lang_type)
    if lang_type.lower() == "deu":
        encoder_type = "xlm-roberta"
        pretrained_path = "xlm-roberta-base"
        # to change
        #encoder_type = "bert"
        #pretrained_path = "bert-base-german-cased"

    elif lang_type.lower() == "eng":
        encoder_type = "roberta" # "electra"
        # pretrained_path = "bert-base-cased" # "google/electra-large-discriminator"
        pretrained_path = "roberta-base"

    elif lang_type.lower() == "eus":
        encoder_type = "bert"
        pretrained_path = "ixa-ehu/berteus-base-cased"

    elif lang_type.lower() == "fas":
        encoder_type = "bert"
        pretrained_path = "HooshvareLab/bert-fa-base-uncased"

    elif lang_type.lower() == "fra":
        #encoder_type = "camembert"
        #pretrained_path = "camembert-large"
        encoder_type = "xlm-roberta"
        pretrained_path = "xlm-roberta-base"

    elif lang_type.lower() == "ita":
        #encoder_type = "bert"
        #pretrained_path = "dbmdz/bert-base-italian-uncased"
        encoder_type = "xlm-roberta"
        pretrained_path = "xlm-roberta-base"

    elif lang_type.lower() == "nld":
        encoder_type = "roberta"
        pretrained_path = "pdelobelle/robbert-v2-dutch-base"

    elif lang_type.lower() == "por":
        encoder_type = "bert"
        pretrained_path = "neuralmind/bert-base-portuguese-cased"

    elif lang_type.lower() == "rus":
        encoder_type = "bert"
        pretrained_path = "DeepPavlov/rubert-base-cased"

    elif lang_type.lower() == "spa":
        encoder_type = "bert"
        pretrained_path = "dccuchile/bert-base-spanish-wwm-cased"

    elif lang_type.lower() == "tur":
        encoder_type = "bert"
        pretrained_path = "dbmdz/bert-base-turkish-cased"

    elif lang_type.lower() == "zho":
        #encoder_type = "bert"
        #pretrained_path = "hfl/chinese-macbert-large"
        encoder_type = "bert"
        pretrained_path = "bert-base-chinese"
        #encoder_type = "xlm-roberta"
        #pretrained_path = "xlm-roberta-base"

    elif lang_type.lower() == "tha":
        encoder_type = "camembert"
        pretrained_path = "airesearch/wangchanberta-base-att-spm-uncased"



    #pretrained_path = os.path.join("/hits/basement/nlp/liuwi/resources/pretrained_models", pretrained_path)

    #print(pretrained_path)

    args.encoder_type = encoder_type
    args.pretrained_path = pretrained_path

    # 2.prepare data
    data_dir = os.path.join(args.data_dir, args.dataset)
    args.data_dir = data_dir
    train_data_file = os.path.join(data_dir, "{}_train.json".format(args.dataset))
    dev_data_file = os.path.join(data_dir, "{}_dev.json".format(args.dataset))
    test_data_file = os.path.join(data_dir, "{}_test.json".format(args.dataset))

    if args.do_train or args.do_dev:
        if os.path.exists(train_data_file):
            label_dict, label_list = token_labels_from_file(train_data_file)
            tok_pos_1, tok_pos_2, tok_pos_1_dict, tok_pos_2_dict = token_pos_from_file(train_data_file)
        else:
            label_dict, label_list = token_labels_from_file(dev_data_file)
            tok_pos_1, tok_pos_2, tok_pos_1_dict, tok_pos_2_dict = token_pos_from_file(dev_data_file)

        args.train_data_file, args.dev_data_file, args.test_data_file = train_data_file, dev_data_file, test_data_file
        args.label_dict, args.label_list, args.num_labels = label_dict, label_list, len(label_list)
    else:
        args.num_labels = 2
        args.label_dict = {'BeginSeg=Yes': 0, '_': 1}
        args.label_list = ['BeginSeg=Yes', '_']

    output_dir = os.path.join(args.output_dir, args.dataset)
    output_dir = os.path.join(output_dir, "{}+{}".format(args.model_type, args.encoder_type))
    os.makedirs(output_dir, exist_ok=True)
    args.output_dir = output_dir

    '''ft_model_path = "./ft_dicts/" + args.dataset + "_ftdict.npy"
    #ft_model_path = os.path.join(data_dir, ft_file_path)
    if os.path.exists(ft_model_path):
        ft_dict = np.load(ft_model_path, allow_pickle=True).item()
    else:
        print("fasttext embedding file is not in the right dirctory or not exist at all!!!! Please fix it!!!!")'''

    # 2.define models
    if args.model_type.lower() == "base":
        if args.encoder_type.lower() == "roberta":
            config = RobertaConfig.from_pretrained(pretrained_path)
            tokenizer = RobertaTokenizer.from_pretrained(pretrained_path)
        elif args.encoder_type.lower() == "bert":
            config = BertConfig.from_pretrained(pretrained_path)
            tokenizer = BertTokenizer.from_pretrained(pretrained_path)
        elif args.encoder_type.lower() == "electra":
            config = ElectraConfig.from_pretrained(pretrained_path)
            tokenizer = ElectraTokenizer.from_pretrained(pretrained_path)
        elif args.encoder_type.lower() == "xlm-roberta":
            config = XLMRobertaConfig.from_pretrained(pretrained_path)
            tokenizer = XLMRobertaTokenizer.from_pretrained(pretrained_path)
        elif args.encoder_type.lower() == "camembert":
            config = CamembertConfig.from_pretrained(pretrained_path)
            tokenizer = CamembertTokenizer.from_pretrained(pretrained_path)

        dataset_name = "SegDataset2"
    elif args.model_type.lower() == "bilstm+crf":
        if args.encoder_type.lower() == "roberta":
            config = RobertaConfig.from_pretrained(pretrained_path)
            tokenizer = RobertaTokenizer.from_pretrained(pretrained_path)
        elif args.encoder_type.lower() == "bert":
            config = BertConfig.from_pretrained(pretrained_path)
            tokenizer = BertTokenizer.from_pretrained(pretrained_path)
        elif args.encoder_type.lower() == "electra":
            config = ElectraConfig.from_pretrained(pretrained_path)
            tokenizer = ElectraTokenizer.from_pretrained(pretrained_path)
        elif args.encoder_type.lower() == "xlm-roberta":
            config = XLMRobertaConfig.from_pretrained(pretrained_path)
            tokenizer = XLMRobertaTokenizer.from_pretrained(pretrained_path)
        elif args.encoder_type.lower() == "camembert":
            config = CamembertConfig.from_pretrained(pretrained_path)
            tokenizer = CamembertTokenizer.from_pretrained(pretrained_path)
        #dataset_name = "SegDataset"
        dataset_name = "SegDatasetPlus"
        # you can test my new dataset by using folowing code
        # dataset_name = "SegDataset2"
        # now you can aplly SegDatasetPlus

    if args.run_plus:
        dataset_params = {
            "tokenizer": tokenizer,
            "max_seq_length": args.max_seq_length,
            "label_dict": label_dict,
            "pos1_dict": tok_pos_1_dict,
            "pos1_list": tok_pos_1,
            "pos1_convert": args.pos1_convert,
            "pos2_dict": tok_pos_2_dict,
            "pos2_list": tok_pos_2,
            "pos2_convert": args.pos2_convert,
        }

        #dataset_module = __import__("task_dataset")
        #MyDataset = getattr(dataset_module, dataset_name)
        MyDataset = SegDatasetPlus
    elif args.bagging:
        dataset_params = {
            "tokenizer": tokenizer,
            "max_seq_length": args.max_seq_length,
            "label_dict": label_dict,
            "ratio": args.ratio
        }
        # dataset_module = __import__("task_dataset")
        # MyDataset = getattr(dataset_module, dataset_name)
        # MyDataset = SegDataset
        MyDataset = SegDataset4Bag
    else:
        if not args.do_test:
            dataset_params = {
                "tokenizer": tokenizer,
                "max_seq_length": args.max_seq_length,
                "label_dict": label_dict,
            }
        else:
            dataset_params = {
                "tokenizer": tokenizer,
                "max_seq_length": args.max_seq_length,
                "label_dict": [],
            }

        MyDataset = SegDataset3
    if args.model_type.lower() == "base":
        model = BaseSegClassifier(config=config, args=args)
    elif args.model_type.lower() == "bilstm+crf":
        if args.run_plus:
            args.pos1_vocab_len = len(tok_pos_1_dict)
            args.pos2_vocab_len = len(tok_pos_2_dict)
            # for the sequence mode, to control the length of the embedding
            args.pos1_dim = 50
            args.pos2_dim = 50
            model = BiLSTMCRFPlus(config=config, args=args)

        else:
            model = BiLSTMCRF(config=config, args=args)
            #model = BiLSTMCRFAdv(config=config, args=args)

    model = model.to(args.device)
    args.tokenizer = tokenizer


    if args.do_train:
        if os.path.exists(train_data_file):
            train_dataset = MyDataset(train_data_file, params=dataset_params)
            print("Total size of the training dataset for " + train_data_file + " is " + str(train_dataset.__len__()))
            train_dataloader = get_dataloader(train_dataset, args, mode="train")
        tok_pos_1_dev, tok_pos_2_dev, tok_pos_1_dict_dev, tok_pos_2_dict_dev = token_pos_from_file(dev_data_file)
        if args.run_plus:
            dev_dataset_params = {
                "tokenizer": tokenizer,
                "max_seq_length": args.max_seq_length,
                "label_dict": label_dict,
                "pos1_dict": tok_pos_1_dict_dev,
                "pos1_list": tok_pos_1_dev,
                "pos1_convert": args.pos1_convert,
                "pos2_dict": tok_pos_2_dict_dev,
                "pos2_list": tok_pos_2_dev,
                "pos2_convert": args.pos2_convert,
            }
            extra_feat_len = train_dataset.get_extra_feat_len()
            args.extra_feat_dim = extra_feat_len
            dev_dataset = MyDataset(dev_data_file, params=dev_dataset_params)
        elif args.bagging:
            dev_dataset_params = {
                "tokenizer": tokenizer,
                "max_seq_length": args.max_seq_length,
                "label_dict": label_dict,
            }
            dev_dataset = SegDataset3(dev_data_file, params=dev_dataset_params)
        else:
            dev_dataset = MyDataset(dev_data_file, params=dataset_params)
        print("Total size of the dev dataset for " + dev_data_file + " is " + str(dev_dataset.__len__()))

        print(test_data_file)



    if os.path.exists(test_data_file):
        print("++in++")
        if args.run_plus:
            tok_pos_1_test, tok_pos_2_test, tok_pos_1_dict_test, tok_pos_2_dict_test = token_pos_from_file(test_data_file)
            test_dataset_params = {
                "tokenizer": tokenizer,
                "max_seq_length": args.max_seq_length,
                "label_dict": label_dict,
                "pos1_dict": tok_pos_1_dict_test,
                "pos1_list": tok_pos_1_test,
                "pos1_convert": args.pos1_convert,
                "pos2_dict": tok_pos_2_dict_test,
                "pos2_list": tok_pos_2_test,
                "pos2_convert": args.pos2_convert,
            }
            test_dataset = MyDataset(test_data_file, params=test_dataset_params)

        elif args.bagging:
            test_dataset_params = {
                "tokenizer": tokenizer,
                "max_seq_length": args.max_seq_length,
                "label_dict": label_dict,
            }
            test_dataset = SegDataset3(test_data_file, params=test_dataset_params)

        else:
            test_dataset = MyDataset(test_data_file, params=dataset_params, mode="test")


    else:
        test_dataset = None


    # print("Total size of the test dataset for " + test_data_file + " is " + str(test_dataset.__len__()))

    # dev_dataloader = get_dataloader(dev_dataset, args, mode="dev")

    if test_dataset is not None:
        test_dataloader = get_dataloader(test_dataset, args, mode="test")
    else:
        test_dataloader = None

    #if args.run_plus:
    #    train_plus(model, args, tokenizer, train_dataloader, dev_dataloader, test_dataloader)
    #else:
    if args.do_train:
        train(model, args, tokenizer, train_dataloader, dev_dataloader, test_dataloader)


    if args.do_dev or args.do_test:
        #time_dir = "good"
        #temp_dir = os.path.join(args.output_dir, time_dir)
        #temp_file = os.path.join(temp_dir, "checkpoint_{}/pytorch_model.bin")
        #temp_file = os.path.join(args.output_dir, args.trained_model + "_" + args.model_type)
        #temp_file = os.path.join(output_dir, f"{PREFIX_CHECKPOINT_DIR}_{epoch}")
        #args.output_dir = os.path.dirname(checkpoint_file)
        #data/result/eng.dep.scidtb/bilstm+crf+roberta/eng.dep.scidtb_bilstm+crf/checkpoint_10/pytorch_model.bin
        #data/result/eng.dep.scidtb/bilstm+crf+roberta/eng.dep.scidtb_bilstm+crf+roberta/eng.dep.scidtb_bilstm+crf/check_point_10/pytorch_model.bin'

        # args.trained_model = eng.dep.scidtb args.model_type=bilstm+crf args.encoder_type.lower()=roberta
        if args.dataset == "eng.dep.covdtb":
            checkpoint_file = "data/result/eng.dep.scidtb/bilstm+crf+roberta/eng.dep.scidtb_bilstm+crf/checkpoint_10/pytorch_model.bin"
            args.trained_model = "eng.dep.scidtb"
        elif args.dataset == "eng.pdtb.tedm":
            checkpoint_file ="data/result/eng.pdtb.pdtb/bilstm+crf+roberta/eng.pdtb.pdtb_bilstm+crf/checkpoint_10/pytorch_model.bin"
            args.trained_model = "eng.pdtb.pdtb"
        elif args.dataset == "por.pdtb.tedm":
            checkpoint_file = "data/result/por.pdtb.crpc/bilstm+crf+bert/por.pdtb.crpc_bilstm+crf/checkpoint_10/pytorch_model.bin"
            args.trained_model = "por.pdtb.crpc"
        elif args.dataset == "tur.pdtb.tedm":
            checkpoint_file = "data/result/tur.pdtb.tdb/bilstm+crf+bert/tur.pdtb.tdb_bilstm+crf/checkpoint_10/pytorch_model.bin"
            args.trained_model = "tur.pdtb.tdb"
        elif args.dataset == "eng.sdrt.stac":
            checkpoint_file = "data/result/eng.sdrt.stac/bilstm+crf+roberta/eng.sdrt.stac_bilstm+crf/checkpoint_10/pytorch_model.bin"
            args.trained_model = "eng.sdrt.stac"
        elif args.dataset == "eng.sdrt.malaysia_hansard":
            checkpoint_file = "data/result/eng.sdrt.stac/bilstm+crf+roberta/eng.sdrt.stac_bilstm+crf/checkpoint_10/pytorch_model.bin"
            args.trained_model = "eng.sdrt.stac"
        if args.do_dev:
            if args.bagging:
                dev_dataset = SegDataset3(dev_data_file, params=dataset_params)
            else:
                dev_dataset = MyDataset(dev_data_file, params=dataset_params)
            dev_dataloader = get_dataloader(dev_dataset, args, mode="dev")
        if args.do_test:
            if args.bagging:
                test_dataset = SegDataset3(test_data_file, params=dataset_params)
            else:
                test_dataset = MyDataset(test_data_file, params=dataset_params)
            test_dataloader = get_dataloader(test_dataset, args, mode="test")

        for epoch in range(1):
            #checkpoint_file = temp_file.format(str(epoch))
            #print(" Epoch: {}".format(str(epoch)))
            #print(checkpoint_file)

            model.load_state_dict(torch.load(checkpoint_file))
            model.eval()

            if args.bagging:
                if args.do_dev:
                    evaluate_new(model, args, dev_dataloader, tokenizer, epoch, desc="dev", write_file=False, nb4bag=args.bag_nb)
                if args.do_test:
                    evaluate_new(model, args, test_dataloader, tokenizer, epoch, desc="test", write_file=False, nb4bag=args.bag_nb)
            else:
                if args.do_dev:
                    #evaluate(model, args, dev_dataloader, tokenizer, epoch, desc="dev", write_file=True)
                    score_dict = evaluate_new(model, args, dev_dataloader, tokenizer, epoch, desc="dev", write_file=False)
                    print(args.dataset)
                    print("\nDev: F1=%.4f\n" % (score_dict["f_score"]))
                    # print(" Dev: acc=%.4f, p=%.4f, r=%.4f, f1=%.4f\n" % (acc, p, r, f1))
                if args.do_test:
                    #evaluate(model, args, test_dataloader, tokenizer, epoch, desc="test", write_file=True)
                    all_preds = inference(model, args, test_dataloader, tokenizer, epoch, desc="test", write_file=False)
                    print(args.dataset)

                    inputs = []
                    with open(test_data_file, 'r') as f:
                        for line in f.readlines():
                            line_content = json.loads(line)
                            for sent in line_content["doc_sents"]:
                                inputs.append(sent)

                    for input, preds in zip(inputs, all_preds):
                        print(input)
                        print(preds)
                    # print(" Test: acc=%.4f, p=%.4f, r=%.4f, f1=%.4f\n" % (acc, p, r, f1))
            print()

if __name__ == "__main__":
    main()
