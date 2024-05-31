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


def dict_to_namespace(d):
    namespace = argparse.Namespace()
    for key, value in d.items():
        setattr(namespace, key, value)
    return namespace

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


args = {
    "data_dir": "data/dataset",
    "dataset": "eng.sdrt.malaysia_hansard",
    "output_dir": "data/result",
    "trained_model": "eng.sdrt.stac",
    "checkpoint_file": "data/result/eng.sdrt.stac/bilstm+crf+roberta/eng.sdrt.stac_bilstm+crf/checkpoint_10/pytorch_model.bin",
    "model_type": "bilstm+crf",
    "encoder_type": "roberta",
    "do_train": False, 
    "do_dev": False, 
    "do_test": False, 
    "do_freeze": False, 
    "do_adv": False, 
    "run_plus": False,
    "bagging": False,
    "ratio": 0.8,
    "bag_nb": 0,
    "train_batch_size": 8,    
    "eval_batch_size": 24,
    "max_seq_length": 512,
    "num_train_epochs": 10, 
    "learning_rate": 3e-5, 
    "dropout": 0.1, 
    "max_grad_norm": 2.0, 
    "weight_decay": 0.1, 
    "warmup_ratio": 0.06, 
    "seed":  106524,
    "extra_feat_dim": 400,
    "pos1_convert": "sequence",
    "pos2_convert": "sequence",
    "n_gpu": 1,
    "device": "cuda:0",
}



def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def get_dataloader(dataset, args, mode="test"):
    print("  {} dataset length: ".format(mode), len(dataset))
    if mode.lower() == "train":
        sampler = RandomSampler(dataset)
        batch_size = args["train_batch_size"]
    else:
        sampler = SequentialSampler(dataset)
        batch_size = args["eval_batch_size"]
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
            "weight_decay": args["weight_decay"]
        },
        {
            "params": [p for n, p in model.named_parameters() if any(nd in n for nd in no_deday)],
            "weight_decay": 0.0
        }
    ]
    optimizer = AdamW(optimizer_grouped_parameters, lr=args["learning_rate"])
    scheduler = get_linear_schedule_with_warmup(
        optimizer, num_warmup_steps=int(num_training_steps * args["warmup_ratio"]),
        num_training_steps=num_training_steps
    )
    return optimizer, scheduler

def inference(model, args, dataloader, tokenizer, epoch, desc="dev", write_file=False, nb4bag=None):
    all_input_ids = None
    all_attention_mask = None
    # all_label_ids = None
    all_pred_ids = None
    all_preds = []
    all_speakers = []
    all_inputs = []
    for batch in tqdm(dataloader, desc=desc):
        batch = (
            batch[0].to(args["device"]),
            batch[1].to(args["device"]),
            batch[2].to(args["device"]),
            batch[3].to(args["device"]),
            batch[4],
            batch[5],
        )
        all_speakers.append(batch[4])
        if args["run_plus"]:
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
        all_inputs.append(batch)
        input_ids = batch[0].detach().cpu().numpy()
        attention_mask = batch[1].detach().cpu().numpy()
        label_ids = batch[2].detach().cpu().numpy()
        og_tok_idxs = batch[3].detach().cpu().numpy()
        if args["model_type"] == "bilstm+crf":
            max_len = args["max_seq_length"]
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
    og_tokens, pred_labels, original_tokens,  = seg_preds_to_file_new4(all_input_ids, all_pred_ids, all_attention_mask, all_tok_idxs, args["tokenizer"], args["label_list"], test_data_file)
    return all_preds, og_tokens, pred_labels, original_tokens, all_speakers, all_inputs
        







if torch.cuda.is_available():
    args["n_gpu"] = 1
    device = torch.device("cuda:0")
    print("Device:", device)
else:
    device = torch.device("cpu")
    args["n_gpu"] = 0


args["device"] = device
logger.info("Training/evaluation parameters %s", args)
set_seed(args["seed"])
# 1.prepare pretrained path
lang_type = args["dataset"].split(".")[0]
args["lang_type"] = lang_type
print(lang_type)

if lang_type.lower() == "deu":
    encoder_type = "xlm-roberta"
    pretrained_path = "xlm-roberta-base"
elif lang_type.lower() == "eng":
    encoder_type = "roberta" # "electra"
    pretrained_path = "roberta-base"
elif lang_type.lower() == "eus":
    encoder_type = "bert"
    pretrained_path = "ixa-ehu/berteus-base-cased"
elif lang_type.lower() == "fas":
    encoder_type = "bert"
    pretrained_path = "HooshvareLab/bert-fa-base-uncased"
elif lang_type.lower() == "fra":
    encoder_type = "xlm-roberta"
    pretrained_path = "xlm-roberta-base"
elif lang_type.lower() == "ita":
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
    encoder_type = "bert"
    pretrained_path = "bert-base-chinese"
elif lang_type.lower() == "tha":
    encoder_type = "camembert"
    pretrained_path = "airesearch/wangchanberta-base-att-spm-uncased"

args["encoder_type"] = encoder_type
args["pretrained_path"] = pretrained_path


# 2.prepare data
data_dir = os.path.join(args["data_dir"], args["dataset"])
args["data_dir"] = data_dir
train_data_file = os.path.join(data_dir, "{}_train.json".format(args["dataset"]))
dev_data_file = os.path.join(data_dir, "{}_dev.json".format(args["dataset"]))
# test_data_file = os.path.join(data_dir, "{}_test.json".format(args["dataset"]))
# test_data_file = r"/mnt/c/Users/user/Documents/Monash/Research/discourse_parsing/disrpt2023/eng.sdrt.malaysia_hansard_test.json"
test_data_file = r"/mnt/c/Users/user/Documents/Monash/Research/hansard_exploration/src/hansard_preprocessing/test0606.json"


args["num_labels"] = 2
args["label_list"] = ['BeginSeg=Yes', '_']
args["label_dict"] = {'BeginSeg=Yes': 0, '_': 1}

output_dir = os.path.join(args["output_dir"], args["dataset"])
output_dir = os.path.join(output_dir, "{}+{}".format(args["model_type"], args["encoder_type"]))
os.makedirs(output_dir, exist_ok=True)
args["output_dir"] = output_dir


# 2.define models
if args["model_type"].lower() == "base":
    if args["encoder_type"].lower() == "roberta":
        config = RobertaConfig.from_pretrained(pretrained_path)
        tokenizer = RobertaTokenizer.from_pretrained(pretrained_path)
    elif args["encoder_type"].lower() == "bert":
        config = BertConfig.from_pretrained(pretrained_path)
        tokenizer = BertTokenizer.from_pretrained(pretrained_path)
    elif args["encoder_type"].lower() == "electra":
        config = ElectraConfig.from_pretrained(pretrained_path)
        tokenizer = ElectraTokenizer.from_pretrained(pretrained_path)
    elif args["encoder_type"].lower() == "xlm-roberta":
        config = XLMRobertaConfig.from_pretrained(pretrained_path)
        tokenizer = XLMRobertaTokenizer.from_pretrained(pretrained_path)
    elif args["encoder_type"].lower() == "camembert":
        config = CamembertConfig.from_pretrained(pretrained_path)
        tokenizer = CamembertTokenizer.from_pretrained(pretrained_path)
    dataset_name = "SegDataset2"
elif args["model_type"].lower() == "bilstm+crf":
    if args["encoder_type"].lower() == "roberta":
        config = RobertaConfig.from_pretrained(pretrained_path)
        tokenizer = RobertaTokenizer.from_pretrained(pretrained_path)
    elif args["encoder_type"].lower() == "bert":
        config = BertConfig.from_pretrained(pretrained_path)
        tokenizer = BertTokenizer.from_pretrained(pretrained_path)
    elif args["encoder_type"].lower() == "electra":
        config = ElectraConfig.from_pretrained(pretrained_path)
        tokenizer = ElectraTokenizer.from_pretrained(pretrained_path)
    elif args["encoder_type"].lower() == "xlm-roberta":
        config = XLMRobertaConfig.from_pretrained(pretrained_path)
        tokenizer = XLMRobertaTokenizer.from_pretrained(pretrained_path)
    elif args["encoder_type"].lower() == "camembert":
        config = CamembertConfig.from_pretrained(pretrained_path)
        tokenizer = CamembertTokenizer.from_pretrained(pretrained_path)
    #dataset_name = "SegDataset"
    dataset_name = "SegDatasetPlus"

if args["model_type"].lower() == "base":
    if args["encoder_type"].lower() == "roberta":
        config = RobertaConfig.from_pretrained(pretrained_path)
        tokenizer = RobertaTokenizer.from_pretrained(pretrained_path)
    elif args["encoder_type"].lower() == "bert":
        config = BertConfig.from_pretrained(pretrained_path)
        tokenizer = BertTokenizer.from_pretrained(pretrained_path)
    elif args["encoder_type"].lower() == "electra":
        config = ElectraConfig.from_pretrained(pretrained_path)
        tokenizer = ElectraTokenizer.from_pretrained(pretrained_path)
    elif args["encoder_type"].lower() == "xlm-roberta":
        config = XLMRobertaConfig.from_pretrained(pretrained_path)
        tokenizer = XLMRobertaTokenizer.from_pretrained(pretrained_path)
    elif args["encoder_type"].lower() == "camembert":
        config = CamembertConfig.from_pretrained(pretrained_path)
        tokenizer = CamembertTokenizer.from_pretrained(pretrained_path)
    dataset_name = "SegDataset2"
elif args["model_type"].lower() == "bilstm+crf":
    if args["encoder_type"].lower() == "roberta":
        config = RobertaConfig.from_pretrained(pretrained_path)
        tokenizer = RobertaTokenizer.from_pretrained(pretrained_path)
    elif args["encoder_type"].lower() == "bert":
        config = BertConfig.from_pretrained(pretrained_path)
        tokenizer = BertTokenizer.from_pretrained(pretrained_path)
    elif args["encoder_type"].lower() == "electra":
        config = ElectraConfig.from_pretrained(pretrained_path)
        tokenizer = ElectraTokenizer.from_pretrained(pretrained_path)
    elif args["encoder_type"].lower() == "xlm-roberta":
        config = XLMRobertaConfig.from_pretrained(pretrained_path)
        tokenizer = XLMRobertaTokenizer.from_pretrained(pretrained_path)
    elif args["encoder_type"].lower() == "camembert":
        config = CamembertConfig.from_pretrained(pretrained_path)
        tokenizer = CamembertTokenizer.from_pretrained(pretrained_path)
    dataset_name = "SegDatasetPlus"

if args["run_plus"]:
    dataset_params = {
        "tokenizer": tokenizer,
        "max_seq_length": args["max_seq_length"],
        "label_dict": label_dict,
        "pos1_dict": tok_pos_1_dict,
        "pos1_list": tok_pos_1,
        "pos1_convert": args["pos1_convert"],
        "pos2_dict": tok_pos_2_dict,
        "pos2_list": tok_pos_2,
        "pos2_convert": args["pos2_convert"],
    }
    MyDataset = SegDatasetPlus
elif args["bagging"]:
    dataset_params = {
        "tokenizer": tokenizer,
        "max_seq_length": args["max_seq_length"],
        "label_dict": label_dict,
        "ratio": args["ratio"]
    }
    MyDataset = SegDataset4Bag
else:
    dataset_params = {
        "tokenizer": tokenizer,
        "max_seq_length": args["max_seq_length"],
        "label_dict": [],
    }
    MyDataset = CustomDataset

# Convert dictionary to argparse object
actual_args = dict_to_namespace(args)

if args["model_type"].lower() == "base":
    model = BaseSegClassifier(config=config, args=actual_args)
elif args["model_type"].lower() == "bilstm+crf":
    if args["run_plus"]:
        args["pos1_vocab_len"] = len(tok_pos_1_dict)
        args["pos2_vocab_len"] = len(tok_pos_2_dict)
        # for the sequence mode, to control the length of the embedding
        args["pos1_dim"] = 50
        args["pos2_dim"] = 50
        model = BiLSTMCRFPlus(config=config, args=actual_args)
    else:
        model = BiLSTMCRF(config=config, args=actual_args)
        #model = BiLSTMCRFAdv(config=config, args=actual_args)

model = model.to(args["device"])
args["tokenizer"] = tokenizer
actual_args = dict_to_namespace(args)



if os.path.exists(test_data_file):
    print("++in++")
    test_dataset = MyDataset(test_data_file, params=dataset_params)
else:
    test_dataset = None


if test_dataset is not None:
    test_dataloader = get_dataloader(test_dataset, args)
else:
    test_dataloader = None


if args["bagging"]:
    test_dataset = SegDataset3(test_data_file, params=dataset_params)
else:
    test_dataset = MyDataset(test_data_file, params=dataset_params)
test_dataloader = get_dataloader(test_dataset, args)

for epoch in range(1):
    model.load_state_dict(torch.load(args["checkpoint_file"]))
    model.eval()
    if args["bagging"]:
        evaluate_new(model, args, test_dataloader, tokenizer, epoch, desc="test", write_file=False, nb4bag=args["bag_nb"])
    else:
        all_preds, og_tokens, pred_labels, original_tokens, all_speakers, all_inputs = inference(model, args, test_dataloader, tokenizer, epoch, desc="test", write_file=False)
    print()

unflat_pred_labels = unflatten_list(pred_labels, original_tokens)
flatten_speakers = flatten_outer_layer(all_speakers)
raw_result = {"speakers": flatten_speakers, "tokens": original_tokens, "pred_segments": unflat_pred_labels}

with open("seg_results.json", "w", encoding="utf-8") as f:
    json.dump(raw_result, f, indent=4,
              ensure_ascii=False)

# Display output
for i, speaker in enumerate(raw_result["speakers"]):
    print(f"({i}) - {speaker}", end="")
    for token, pred in zip(raw_result["tokens"][i], raw_result["pred_segments"][i]):
        if pred == "BeginSeg=Yes":
            print(f"\n - {token}", end=" ")
        else:
            print(token, end=" ")
    print()
    print()


# Save as EDUs
edus = []
j = 0
for i, speaker in enumerate(raw_result["speakers"]):
    content = []
    for j, (token, pred) in enumerate(zip(raw_result["tokens"][i], raw_result["pred_segments"][i])):
        if pred == "BeginSeg=Yes":
            if len(content) > 0:
                edu = {
                    "edu_index": j,
                    "sent_index": i,
                    "speaker": speaker,
                    "content": " ".join(content)
                }
                # Add to list
                edus.append(edu)
                # Increase EDU index count
                j = j + 1
            # Restart new list for new EDU segment
            content = [token]
        else:
            # Continue EDU segment
            content.append(token)

with open("edus_results.json", "w", encoding="utf-8") as f:
    json.dump({"edus": edus}, f, indent=4,
              ensure_ascii=False)
