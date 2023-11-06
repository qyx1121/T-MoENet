import os
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import argparse

from torch.utils.data import DataLoader

from transformers import DebertaV2Tokenizer
from datasets.videoqa_dataset import VideoQA_Dataset, videoqa_collate_fn
from model import build_model, get_tokenizer
from args import get_args_parser
from util.misc import get_mask

from tqdm import tqdm
from collections import OrderedDict

from datasets import build_videoqa_dataset


def main(args):

    data_name = args.dataset_path.split("/")[-2]
    new_state_dict = OrderedDict()
    ckpt = torch.load(args.model_path)
    for k, v in ckpt['model'].items():
        new_state_dict[k.replace("module.","")] = v

    cfgs = ckpt['args']
    cfgs.max_feats = 10
    cfgs.sample_nums = 10
    if cfgs.add_video_feat:
        cfgs.max_feats += 1

    tokenizer = get_tokenizer(cfgs)
    type_maps = {"MSVD-QA": {0: "what", 1: "how", 2: "color", 3: "where", 4: "who", 5: "when"}, 
                "MSRVTT-QA": {0: "what", 1: "how", 2: "color", 3: "where", 4: "who", 5: "when"},
                'ActivityNet-QA':{
                0: "motion",
                1: "spatial",
                2: "temporal",
                3: "yesno",
                4: "color",
                5: "object",
                6: "location",
                7: "number",
                8: "other",
            },
            'TGIF-QA':{0: "what", 1: "how", 2: "color", 3: "where"},
            'iVQA': None
                }

    model = build_model(cfgs, None)

    total_num = sum(p.numel() for p in model.parameters())

    model.cuda()
    model.eval()
    model.load_state_dict(new_state_dict, strict=False)
    
    type_map = type_maps[data_name]
    dataset = VideoQA_Dataset(
        csv_path=args.dataset_path,
        features_path=args.feat_path,
        vocab_path=args.vocab_path,
        tokenizer=tokenizer,
        type_map=type_map,
        max_feats = cfgs.sample_nums,
    )

    loader = DataLoader(dataset, batch_size = 32, collate_fn=videoqa_collate_fn, shuffle=True)
    cfgs.n_ans = len(dataset.a2id)

    model.n_ans = cfgs.n_ans
    if 'deberta' in args.model_name:
        model.answer_embeddings = nn.Embedding(
                args.n_ans, model.deberta.embeddings.embedding_size
            ).cuda()
    elif 'roberta' in args.model_name:
        model.answer_embeddings = nn.Embedding(
                args.n_ans, model.roberta.embeddings.embedding_size
            ).cuda()
    else:
        model.answer_embeddings = nn.Embedding(
                args.n_ans, model.bert.embeddings.embedding_size
            ).cuda()

    model.answer_bias = nn.Parameter(torch.zeros(args.n_ans, device="cuda"))

# Init answer embedding module
    aid2tokid = torch.zeros(cfgs.n_ans, cfgs.max_atokens).long()
    for a, aid in dataset.a2id.items():
        tok = torch.tensor(
            tokenizer(
                a,
                add_special_tokens=False,
                max_length=cfgs.max_atokens,
                truncation=True,
                padding="max_length",
            )["input_ids"],
            dtype=torch.long,
        )
        aid2tokid[aid] = tok
    model.set_answer_embeddings(aid2tokid.cuda(), freeze_last=False)

    res = {}
    for idx, item in enumerate(tqdm(loader)):
        
        video = item["video"].cuda()

        video_len = item["video_len"]

        video_mask = get_mask(video_len, video.size(1)).cuda()
        text = item["text"]
        encoded = tokenizer(
            text,
            add_special_tokens=True,
            max_length=cfgs.max_tokens,
            padding="longest",
            truncation=True,
            return_tensors="pt",
        )

        inputs = encoded["input_ids"].cuda()
        attention_mask = encoded["attention_mask"].cuda()
        
        # forward
        if cfgs.add_video_feat:
            video_mask = torch.cat([torch.ones((video.size(0),1)).cuda(),video_mask], dim=1)

        output = model(
            video=video,
            video_mask=video_mask,
            input_ids=inputs,
            attention_mask=attention_mask,
        )
        
        thresholds=[1]
        logits = output["logits"]
        delay = cfgs.max_feats if cfgs.use_video else 0
        logits = logits[:, delay : encoded["input_ids"].size(1) + delay][
            encoded["input_ids"] == tokenizer.mask_token_id
        ]  # get the prediction on the mask token
        logits = logits.softmax(-1)
        topk_aids = torch.topk(logits, max(thresholds), -1).indices

        answer_id, qids = item["answer_id"].cuda(), item["qid"]
        types = item["type"]
        if "sub" in item:
            subs = item["sub"]
        else:
            subs = [0] * len(types)
    
        if data_name == "iVQA":
            answer_id = (answer_id / 2).clamp(max=1)
            answer_id_expanded = answer_id.cuda()
        else:
            answer_id_expanded = answer_id.view(-1, 1).expand_as(topk_aids).cuda()

        agreeings = {}
        for x in thresholds:
            if data_name not in ["iVQA", "vqa"]:
                agreeings[x] = topk_aids[:, :x] == answer_id_expanded[:, :x]
            else:
                predicted = F.one_hot(
                    topk_aids[:, :x], num_classes=answer_id_expanded.shape[-1]
                ).sum(1)
                agreeings[x] = (predicted * answer_id_expanded).max(1)[0]
        for i, (qid, gt, pred, type, sub) in enumerate(
            zip(qids, answer_id, topk_aids, types, subs)
        ):
            res[qid] = { 
                "pred": dataset.id2a[pred.tolist()[0]],
                "gt": gt.tolist() if data_name =="iVQA" else gt.item(),
                "type": int(type),
                "sub": sub,
            }
            for x in thresholds:
                res[qid][f"acc{x}"] = agreeings[x][i].sum().detach().cpu().item()

        dico = {"acc": agreeings[1].sum() / len(qids)}
        acc_value = dico["acc"].item()

    results = res
    assert len(results) == len(loader.dataset)
    out = {}
    for x in thresholds:
        out[f"acc{x}"] = sum(results[qid][f"acc{x}"] for qid in results) / len(results)
    if type_map is not None and len(type_map) > 1:
        acc_type = {
            type_map[i]: sum(
                results[qid][f"acc1"] for qid in results if results[qid]["type"] == i
            )
            / len([x for x in results.values() if x["type"] == i])
            for i in type_map
        }
    n_sub = len([x for x in results.values() if x["sub"]])
    if n_sub:
        acc_sub = (
            sum(results[qid][f"acc1"] for qid in results if results[qid]["sub"]) / n_sub
        )
    print(data_name)
    for x in thresholds:
        print(f"test acc{x}: {out[f'acc{x}']: .2%}")
    if type_map is not None and len(type_map) > 1:
        for x in acc_type:
            print(f"acc {x}: {acc_type[x]: .2%}")
        out.update(acc_type)
    if n_sub:
        print(f"acc sub: {acc_sub: .2%}; proportion {n_sub / len(results): .2%}")
        out["acc_sub"] = acc_sub


if __name__ == "__main__":
    parser = argparse.ArgumentParser()

    parser.add_argument(    
            "--dataset_path",
            type=str,
            help="the path to the test/val set"
        )
    
    parser.add_argument(    
            "--feat_path",
            type=str,
            help="the path of the pre-processed video feature"
        )
    
    parser.add_argument(    
            "--vocab_path",
            type=str,
            help="the path to the answer dictionary for the dataset"
        )
    
    parser.add_argument(    
            "--model_path",
            type=str,
            help="the path of the pre-trained model"
        )
    
    parser.add_argument(    
            "--save_result",
            action="store_true",
            help="whether to save the result file after inference"
        )
    
    parser.add_argument(    
            "--save_dir",
            type=str,
            help="the directory where the inference result files are saved"
        )
    
    args = parser.parse_args()
    main(args)
    









