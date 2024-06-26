import os
import json
from collections import defaultdict

import torch
import torch.nn.functional as F
import numpy as np
# import fasttext
# import fasttext.util

from seg_eval import get_scores

def token_labels_from_file(file_name):
    labels = set()
    with open(file_name, "r", encoding="utf-8") as f:
        lines = f.readlines()
        for line in lines:
            line = line.strip()
            if line:
                sample = json.loads(line)
                doc_sent_token_labels = sample["doc_sent_token_labels"]
                for sent_token_labels in doc_sent_token_labels:
                    for l in sent_token_labels:
                        # labels.add(l.lower())
                        labels.add(l)
    labels = list(labels)
    labels = sorted(labels)
    print(" Total label number: %d\n"%(len(labels)))
    label_dict = {l: idx for idx, l in enumerate(labels)}
    # label_id_dict = {idx: l for idx, l in enumerate(labels)}
    return label_dict, labels

def token_pos_from_file(file_name):
    tok_pos_1 = set()
    tok_pos_2 = set()
    with open(file_name, "r", encoding="utf-8") as f:
        lines = f.readlines()
        for line in lines:
            line = line.strip()
            if line:
                sample = json.loads(line)
                doc_sent_token_labels = sample["doc_sent_token_features"]
                for sent_token_labels in doc_sent_token_labels:
                    for feat in sent_token_labels:
                        tok_pos_1.add(feat[1])
                        tok_pos_2.add(feat[2])
    tok_pos_1 = list(tok_pos_1)
    tok_pos_2 = list(tok_pos_2)
    tok_pos_1 = sorted(tok_pos_1)
    tok_pos_2 = sorted(tok_pos_2)
    tok_pos_1_dict = {t: idx + 1 for idx, t in enumerate(tok_pos_1)}
    tok_pos_2_dict = {t: idx + 1 for idx, t in enumerate(tok_pos_2)}
    tok_pos_1_dict["SEPCIAL_TOKEN"] = 0
    tok_pos_2_dict["SEPCIAL_TOKEN"] = 0
    return tok_pos_1, tok_pos_2, tok_pos_1_dict, tok_pos_2_dict


def unify_rel_labels(label, corpus_name):
    """
    Here we convert original label into an unified label
    Args:
        label: original label
        corpus_name:
    """
    if corpus_name == "eng.dep.covdtb":
        return label.lower()
    elif corpus_name == "eng.sdrt.stac":
        return label.lower()
    else:
        return label

def rel_label_to_original(label, corpus_name):
    """
    We remap the rel label to original one. Doing so, we can recall rel_eval
    Args:
        label:
        corpus_name:
    """
    if corpus_name == "eng.dep.covdtb":
        return label.upper()
    elif corpus_name == "eng.sdrt.stac":
        if label == "q_elab":
            return "Q_Elab"
        else:
            return label.capitalize()
    else:
        return label

def rel_map_for_zeroshot(label, dname):
    """
    Some zeroshot corpora contain totally different label set to
    corpora with the similar annotated theory. For example, eng.dep.covdtb
    has very different labels to eng.dep.scidtb and zho.dep.scidtb. So here
    we design a mapping function for such zero-shot corpora.

    Note, mapping only works for corpora without training set
    """
    if dname == "eng.dep.covdtb":
        mapping_dict = {
            'ATTRIBUTION': 'ATTRIBUTION', 'BG-COMPARE': 'BACKGROUND', 'BG-GENERAL': 'BACKGROUND',
            'BG-GOAL': 'BACKGROUND', 'CAUSE': 'CAUSE-RESULT', 'COMPARISON': 'COMPARISON',
            'CONDITION': 'CONDITION', 'CONTRAST': 'CONTRAST', 'ELAB-ADDITION': 'ELABORATION',
            'ELAB-ASPECT': 'ELABORATION', 'ELAB-DEFINITION': 'ELABORATION', 'ELAB-ENUMEMBER': 'ELABORATION',
            'ELAB-EXAMPLE': 'ELABORATION', 'ELAB-PROCESS_STEP': 'ELABORATION', 'ENABLEMENT': 'ENABLEMENT',
            'EVALUATION': 'EVALUATION', 'EXP-EVIDENCE': 'CAUSE-RESULT', 'EXP-REASON': 'CAUSE-RESULT',
            'JOINT': 'JOINT', 'MANNER-MEANS': 'MANNER-MEANS', 'PROGRESSION': 'PROGRESSION',
            'RESULT': 'CAUSE-RESULT', 'ROOT': 'ROOT', 'SUMMARY': 'SUMMARY', 'TEMPORAL': 'TEMPORAL'
        }
        return mapping_dict[label]
    elif dname in ["por.pdtb.tedm", "eng.pdtb.tedm", "tur.pdtb.tedm"]:
        mapping_dict = {
            "QAP.Hypophora": "Hypophora", "QAP": "Hypophora", "Expansion.Level": "Expansion.Level-of-detail",
            "Comparison": "Comparison.Concession", "Temporal": "Temporal.Synchronous"
        }
        if label in mapping_dict:
            return mapping_dict[label]
        else:
            return label

def rel_labels_from_file(file_name):
    label_frequency = defaultdict(int)
    with open(file_name, "r", encoding="utf-8") as f:
        lines = f.readlines()
        for line in lines:
            line = line.strip()
            if line:
                sample = json.loads(line)
                dname = sample["dname"]
                doc_unit_labels = sample["doc_unit_labels"]
                for label_pair in doc_unit_labels:
                    l = label_pair[0]
                    label_frequency[unify_rel_labels(l, dname)] += 1
    labels = []
    for key in label_frequency:
        if label_frequency[key] >= 0:
            labels.append(key)
    labels = sorted(labels, key=lambda x: x.upper())
    label_dict = {l: idx for idx, l in enumerate(labels)}
    # print(labels)
    # print(label_dict)
    # print(" Total label number: %d\n"%(len(labels)))

    return label_dict, labels

def seg_preds_to_file(all_pred_ids, all_label_ids, all_attention_mask, tokenizer, label_id_dict, gold_file):
    """
    convert prediction ids to labels, and save the results into a file with the same format as gold_file
    Args:
        all_pred_ids: predicted tokens' id list 
        all_label_ids: predicted labels' id list 
        all_attention_mask: attention mask of the pre-trained LM
        label_id_dict: the dictionary map the labels' id to the original string label
        gold_file: the original .tok file
    """
    all_doc_data = []
    new_doc_data = []
    with open(gold_file, "r", encoding="utf-8") as f:
        lines = f.readlines()
        tmp_doc_id = None
        tmp_doc = []
        for line in lines:
            all_doc_data.append(line)
    og_tokens = []
    pred_labels = []

    for i in range(len(all_attention_mask)):
        tmp_toks = tokenizer.convert_ids_to_tokens(all_pred_ids[i])
        for j in range(len(all_attention_mask[i])):
            if all_attention_mask[i][j]:
                # the mapping problem happens here! I fixed it.
                if tmp_toks[j] != "[CLS]" and tmp_toks[j] != "<s>":
                    if tmp_toks[j] == "[SEP]" or tmp_toks[j] == "</s>":
                        og_tokens.append(".")
                        pred_labels.append("_")
                    else:
                        og_tokens.append(tmp_toks[j])
                        pred_labels.append(label_id_dict[int(all_label_ids[i][j])])
    pointer = 0
    for line in all_doc_data:
        if line != '\n':
            if "newdoc_id" in line.lower():
                new_doc_data.append(line)
            else:
                items = line.split("\t")
                if "-" in items[0]:  # ignore such as 16-17
                    continue
                items[-1] = pred_labels[pointer]
                # here, I force items[-2] to be the original token, so you can see from the output file
                # that every token is mapped well. If you check that everything is ok, it can be deleted.
                items[-2] = og_tokens[pointer]
                new_doc_data.append("\t".join(items))
                pointer += 1
        else:
            new_doc_data.append('\n')

    pred_file = gold_file.replace(".tok", "_pred.tok")
    with open(pred_file,"w") as f:
        for line in new_doc_data:
            if line[-1:] != "\n":
                f.write(line + "\n")
            else:
                f.write(line)

    return pred_file

def seg_preds_to_file_new(all_input_ids, all_label_ids, all_attention_mask, all_tok_idxs, tokenizer, label_id_dict, gold_tok_file, nb4bag=None):
    """
    new version of writing a result tok file
    convert prediction ids to labels, and save the results into a file with the same format as gold_file
    Args:
        all_input_ids: predicted tokens' id list 
        all_label_ids: predicted labels' id list 
        all_attention_mask: attention mask of the pre-trained LM
        label_id_dict: the dictionary map the labels' id to the original string label
        gold_file: the original .tok file
    """
    all_doc_data = []
    new_doc_data = []
    with open(gold_tok_file, "r", encoding="utf-8") as f:
        lines = f.readlines()
        for line in lines:
            all_doc_data.append(line)
    og_tokens = []
    pred_labels = []

    for i in range(len(all_attention_mask)):

        tmp_idx = [idx for idx in all_tok_idxs[i] if idx > 0]

        og_toks_ids = all_input_ids[i][tmp_idx]
        og_labels = all_label_ids[i][tmp_idx]
        tmp_toks = tokenizer.convert_ids_to_tokens(og_toks_ids)
        for j in range(len(tmp_toks)):
            og_tokens.append(tmp_toks[j])
            pred_labels.append(label_id_dict[int(og_labels[j])])

    pointer = 0
    #print(len(og_tokens))
    #print(len(pred_labels))
    for line in all_doc_data:
        if line != '\n':
            #if "newdoc_id" in line.lower():
            if "newdoc" in line.lower() or "newdoc_id" in line.lower():
                new_doc_data.append(line)
            else:
                items = line.split("\t")
                if "-" in items[0]:  # ignore such as 16-17
                    continue
                #print(pointer)
                items[-1] = pred_labels[pointer]
                items[-2] = og_tokens[pointer]
                #print(items)
                new_doc_data.append("\t".join(items))
                pointer += 1
        else:
            new_doc_data.append('\n')
    if nb4bag is None:
        pred_file = gold_tok_file.replace(".tok", "_pred.tok")
    else:
        pred_file = gold_tok_file.replace(".tok", "_pred_bag"+ str(nb4bag) +".tok")
    with open(pred_file, "w") as f:
        for line in new_doc_data:
            if line[-1:] != "\n":
                f.write(line + "\n")
            else:
                f.write(line)
    return pred_file

def seg_preds_to_file_new2(all_input_ids, all_label_ids, all_attention_mask, all_tok_idxs, tokenizer, label_id_dict, gold_tok_file, nb4bag=None):
    """
    new version of writing a result tok file
    convert prediction ids to labels, and save the results into a file with the same format as gold_file
    Args:
        all_input_ids: predicted tokens' id list
        all_label_ids: predicted labels' id list
        all_attention_mask: attention mask of the pre-trained LM
        label_id_dict: the dictionary map the labels' id to the original string label
        gold_file: the original .tok file
    """
    all_doc_data = []
    new_doc_data = []
    with open(gold_tok_file, "r", encoding="utf-8") as f:
        lines = f.readlines()
        for line in lines:
            all_doc_data.append(line)
    og_tokens = []
    pred_labels = []

    for i in range(len(all_tok_idxs)):
        og_toks_ids = all_input_ids[i]
        og_labels = all_label_ids[i]
        tmp_toks = tokenizer.convert_ids_to_tokens(og_toks_ids)
        for j in range(len(all_tok_idxs[i])):
            if all_tok_idxs[i][j] == 1:
                og_tokens.append(tmp_toks[j])
                pred_labels.append(label_id_dict[int(og_labels[j])])

    pointer = 0
    for line in all_doc_data:
        if line != '\n':
            #if "newdoc_id" in line.lower():
            if "newdoc id" in line.lower() or "newdoc_id" in line.lower():
                new_doc_data.append(line)
            else:
                items = line.split("\t")
                if "-" in items[0]:  # ignore such as 16-17
                    continue
                #print("=====================================error-------------")
                #print(pointer)
                #print(items)
                #print("=====================================error-------------")
                items[-1] = pred_labels[pointer]
                items[-2] = og_tokens[pointer]
                #if pointer > 1745 and pointer < 1755:
                #    print(pointer)
                #    print(items)
                new_doc_data.append("\t".join(items))
                pointer += 1

        else:
            new_doc_data.append('\n')
    if nb4bag is None:
        pred_file = gold_tok_file.replace(".tok", "_pred.tok")
    else:
        pred_file = gold_tok_file.replace(".tok", "_pred_bag"+ str(nb4bag) +".tok")
    with open(pred_file, "w") as f:
        for line in new_doc_data:
            if line[-1:] != "\n":
                f.write(line + "\n")
            else:
                f.write(line)
    return pred_file


def seg_preds_to_file_new3(all_input_ids, all_label_ids, all_attention_mask, all_tok_idxs, tokenizer, label_id_dict, input_file, nb4bag=None):
    """
    new version of writing a result tok file
    convert prediction ids to labels, and save the results into a file with the same format as gold_file
    Args:
        all_input_ids: predicted tokens' id list
        all_label_ids: predicted labels' id list
        all_attention_mask: attention mask of the pre-trained LM
        label_id_dict: the dictionary map the labels' id to the original string label
        gold_file: the original .tok file
    """

    og_tokens = []
    pred_labels = []

    for i in range(len(all_tok_idxs)):
        og_toks_ids = all_input_ids[i]
        og_labels = all_label_ids[i]
        tmp_toks = tokenizer.convert_ids_to_tokens(og_toks_ids)
        for j in range(len(all_tok_idxs[i])):
            if all_tok_idxs[i][j] == 1:
                og_tokens.append(tmp_toks[j])
                pred_labels.append(label_id_dict[int(og_labels[j])])

    
    original_tokens = []
    with open(input_file, "r", encoding="utf-8") as f:
        for line in f.readlines():
            line_content = json.loads(line)
            original_tokens += line_content["doc_sents"]
    # original_tokens = flatten_outer_layer(original_n tokens)

    return og_tokens, pred_labels, original_tokens


def seg_preds_to_file_new4(all_input_ids, all_label_ids, all_attention_mask, all_tok_idxs, tokenizer, label_id_dict, input_file, nb4bag=None):
    """
    new version of writing a result tok file
    convert prediction ids to labels, and save the results into a file with the same format as gold_file
    Args:
        all_input_ids: predicted tokens' id list
        all_label_ids: predicted labels' id list
        all_attention_mask: attention mask of the pre-trained LM
        label_id_dict: the dictionary map the labels' id to the original string label
        gold_file: the original .tok file
    """
    print(len(all_input_ids), len(all_label_ids), len(all_attention_mask), len(all_tok_idxs))

    og_tokens = []
    pred_labels = []

    for i in range(len(all_tok_idxs)):
        og_toks_ids = all_input_ids[i]
        og_labels = all_label_ids[i]
        tmp_toks = tokenizer.convert_ids_to_tokens(og_toks_ids)
        for j in range(len(all_tok_idxs[i])):
            if all_tok_idxs[i][j] == 1:
                og_tokens.append(tmp_toks[j])
                pred_labels.append(label_id_dict[int(og_labels[j])])

    
    original_tokens = []
    doc_token_list = []

    with open(input_file, "r", encoding="utf-8") as f:
        data = json.load(f)

    # Iterate each agenda
    for agenda in data["agendas"]:
        # Load all tokens
        original_tokens += [sent["translated_content_token"] for sent in agenda["translated_sentences"]]
        # original_tokens += line_content["doc_sents"]
    # original_tokens = flatten_outer_layer(original_tokens)

    return og_tokens, pred_labels, original_tokens



def merge4bag(data_path, gold_tok_file):
    file_names = os.listdir(data_path)
    bagging_files = [data_path + "/" + file_name for file_name in file_names if "test_pred_bag" in file_name]
    result_list = []
    # read all pred files
    for bagging_file in bagging_files:
        with open(bagging_file, "r", encoding="utf-8") as f:
            print(bagging_file)
            lines = f.readlines()
            temp = []
            for line in lines:
                if line != '\n':
                    if "newdoc_id" in line.lower() or "newdoc" in line.lower():
                        continue
                    else:
                        items = line.split("\t")
                        if "-" in items[0]:  # ignore such as 16-17
                            continue
                        temp.append(items[-1])
        print(len(temp))
        print("===============================================================")
        result_list.append(temp)

    merge_res = []
    # select the most voted label
    for i in range(len(result_list[0])):
        temp = []
        for j in range(len(result_list)):
            temp.append(result_list[j][i])
        merge_res.append(temp)
    final_res = []
    for k in range(len(merge_res)):
        most_voted = max(merge_res[k], key=merge_res[k].count)
        final_res.append(most_voted)
    print("--------------------------------------")
    print(len(final_res))
    print("--------------------------------------")
    all_doc_data = []
    new_doc_data = []
    with open(gold_tok_file, "r", encoding="utf-8") as f:
        lines = f.readlines()
        for line in lines:
            all_doc_data.append(line)

    pointer = 0
    for line in all_doc_data:
        if line != '\n':
            # if "newdoc_id" in line.lower():
            if "newdoc id" in line.lower() or "newdoc_id" in line.lower():
                new_doc_data.append(line)
            else:
                items = line.split("\t")
                if "-" in items[0]:  # ignore such as 16-17
                    continue
                # print("=====================================error-------------")
                # print(pointer)
                # print(items)
                # print("=====================================error-------------")
                items[-1] = final_res[pointer]
                # print(items)
                new_doc_data.append("\t".join(items))
                pointer += 1

        else:
            new_doc_data.append('\n')

    pred_file = gold_tok_file.replace(".tok", "_pred_bg.tok")
    with open(pred_file, "w") as f:
        for line in new_doc_data:
            if line[-1:] != "\n":
                f.write(line + "\n")
            else:
                f.write(line)
    return pred_file

def merge_result(gold_tok_file):
    file_names = os.listdir(gold_tok_file)
    bagging_files = [gold_tok_file + "/" + file_name for file_name in file_names if "test_pred_bag" in file_name]
    result_list = []
    for bagging_file in bagging_files:
        with open(bagging_file, "r", encoding="utf-8") as f:
            lines = f.readlines()
            temp = []
            for line in lines:
                if line != '\n':
                    if "newdoc_id" in line.lower():
                        continue
                    else:
                        items = line.split("\t")
                        if "-" in items[0]:  # ignore such as 16-17
                            continue
                        temp.append(items)
        result_list.append(temp)

    all_preds = [[] for i in range(len(result_list[0]))]
    for sub_res in result_list:
        for i in range(len(sub_res)):
            all_preds[i].append(sub_res[i][-1])
    final_res = []
    for pred in all_preds:
        most_pred = max(set(pred), key=pred.count)
        final_res.append(most_pred)

    new_doc_data = []
    pointer = 0
    for line in result_list[0]:
        line = " ".join(line)
        if line != '\n':
            # if "newdoc_id" in line.lower():
            if "newdoc" in line.lower() or "newdoc_id" in line.lower():
                new_doc_data.append(line)
            else:
                items = line.split("\t")
                if "-" in items[0]:  # ignore such as 16-17
                    continue
                items[-1] = final_res[pointer]
                #items[-2] = og_tokens[pointer]
                new_doc_data.append("\t".join(items))
                pointer += 1
        else:
            new_doc_data.append('\n')


    pred_file = bagging_files[0].replace("bag1.tok", "bag_final.tok")
    gold_tok = "/hits/basement/nlp/yif/disrpt2023-main/data/dataset/deu.rst.pcc/deu.rst.pcc_test.tok"
    with open(pred_file, "w") as f:
        for line in new_doc_data:
            if line[-1:] != "\n":
                f.write(line + "\n")
            else:
                f.write(line)
    print(gold_tok)
    print(pred_file)
    score_dict = get_scores(gold_tok, pred_file)
    return score_dict

def generate_ft_dict(train_file_path, dev_file_path, test_file_path, output_path, ft_model_path, ft_lang):
    all_files = [train_file_path, dev_file_path, test_file_path]
    #all_files = [dev_file_path, test_file_path]
    # fasttext.util.download_model(ft_lang, if_exists='ignore') 
    #ft = fasttext.load_model(ft_model_path)
    ft = fasttext.load_model(ft_model_path)
    all_texts = []
    token_list = []
    ft_dict = {}
    for path in all_files:
        with open(path, 'r') as f:
            for line in f.readlines():
                line_content = json.loads(line)
                all_texts.append(line_content)
        for doc in all_texts:
            doc_token_list = doc["doc_sents"]
            for i in range(len(doc_token_list)):
                for j in range(len(doc_token_list[i])):
                    token_list.append(doc_token_list[i][j])  
    for i in range(len(token_list)):
        ft_dict[token_list[i]] = ft.get_word_vector(token_list[i])
    np.save(output_path, ft_dict)
    print(" Finish filtering the unrelated embedding from {}.".format(ft_model_path))
    return ft_dict


def rel_preds_to_file(pred_ids, label_list, gold_file):
    dname = gold_file.split("/")[-1].split("_")[0].strip()
    pred_labels = [label_list[idx] for idx in pred_ids]
    pred_labels = [rel_label_to_original(label, dname) for label in pred_labels]
    if dname in ["eng.dep.covdtb", "por.pdtb.tedm", "eng.pdtb.tedm", "tur.pdtb.tedm"]:
        pred_labels = [rel_map_for_zeroshot(label, dname) for label in pred_labels]
    valid_lines = []
    with open(gold_file, "r", encoding="utf-8") as f:
        lines = f.readlines()
        title_line = lines[0]
        lines = lines[1:]
        for line in lines:
            line = line.strip()
            if line:
                valid_lines.append(line)

    assert len(pred_labels) == len(valid_lines), (len(pred_labels), len(valid_lines))

    pred_contents = []
    for pred, line in zip(pred_labels, valid_lines):
        items = line.split("\t")
        new_items = items[:-1]
        new_items.append(pred)
        pred_contents.append("\t".join(new_items))

    pred_file = gold_file.replace(".rels", "_pred.rels")
    with open(pred_file, "w", encoding="utf-8") as f:
        f.write("%s\n"%(title_line.strip()))
        for text in pred_contents:
            f.write("%s\n"%(text))

    return pred_file

def fix_param(model):
    for name, param in model.named_parameters():
        param.requires_grad = False
    return model

def merge_datasets(discourse_type="rst"):
    """
    merge a group of corpus for training
    Args:
        dataset_list: corpus list
        mode:
    """
    out_dir = os.path.join("data/dataset", "super.{}".format(discourse_type))
    os.makedirs(out_dir, exist_ok=True)
    out_file = os.path.join(out_dir, "super.{}_train.json".format(discourse_type))

    if os.path.exists(out_file):
        return out_file

    if discourse_type == "rst":
        # we remove the zho.rst.gcdt because this corpus has no overlaping labels with other corpus
        dataset_list = [
            "deu.rst.pcc", "eng.rst.gum", "eng.rst.rstdt",
            "eus.rst.ert", "fas.rst.prstc", "nld.rst.nldt",
            "por.rst.cstn", "rus.rst.rrt", "spa.rst.rststb",
            "spa.rst.sctb","zho.rst.sctb"
        ]
    elif discourse_type == "pdtb":
        # we remove the zho.pdtb.cdtb because this corpus has no common labels with other corpus
        dataset_list = [
            "ita.pdtb.luna", "tur.pdtb.tdb", "tha.pdtb.tdtb", 
            "eng.pdtb.pdtb", "por.pdtb.crpc"
        ]
    elif discourse_type == "dep":
        dataset_list = ["eng.dep.scidtb", "zho.dep.scidtb"]
    elif discourse_type == "sdrt":
        dataset_list = ["eng.sdrt.stac", "fra.sdrt.annodis"]
    all_merged_texts = []
    for dataset in dataset_list:
        data_dir = os.path.join("data/dataset", dataset)
        data_file = os.path.join(data_dir, "{}_train.json".format(dataset))
        with open(data_file, "r", encoding="utf-8") as f:
            all_texts = f.readlines()
            for text in all_texts:
                text = text.strip()
                if text:
                    sample = json.loads(text)
                    doc_units = sample["doc_units"]
                    doc_unit_labels = sample["doc_unit_labels"]
                    corpus_name = dataset
                    new_doc_unit_labels = []
                    for label in doc_unit_labels:
                        new_doc_unit_labels.append(label)

                    new_sample = {}
                    new_sample["dname"] = corpus_name
                    new_sample["doc_units"] = doc_units
                    new_sample["doc_unit_labels"] = new_doc_unit_labels
                    all_merged_texts.append(new_sample)

    with open(out_file, "w", encoding="utf-8") as f:
        for text in all_merged_texts:
            f.write("%s\n"%(json.dumps(text, ensure_ascii=False)))

    return out_file

def flatten_outer_layer(nested_list_of_lists_of_lists):
    flattened = []
    for sublist_list in nested_list_of_lists_of_lists:
        flattened.extend(sublist_list)
    return flattened

def unflatten_list(flat_list, nested_list):
    """
    Reconstruct a nested list based on the structure of a nested list
    using elements from a flat list.
    
    Args:
        flat_list (list): The flat list containing elements.
        nested_list (list): The template nested list.
        
    Returns:
        list: A nested list with the same structure as nested_list.

    Example:
        >>> nested_list = [[1, 2], [3, 4, 5], [6]]
        >>> flat_list = ["a", "b", "c", "d", "e", "f"]
        >>> unflattened_list = unflatten_list(flat_list, nested_list)
        >>> print(unflattened_list)
        [['a', 'b'], ['c', 'd', 'e'], ['f']]
    """
    # Create an iterator from the flat list to keep track of the position
    it = iter(flat_list)
    
    def build(template):
        """
        Recursively rebuild the nested structure.
        
        Args:
            template (list): The current nested template part.
        
        Returns:
            list: The rebuilt nested list part.
        """
        if isinstance(template, list):
            return [build(subtemplate) for subtemplate in template]
        else:
            # Replace the template element with the next element from the flat list
            return next(it)
    
    return build(nested_list)