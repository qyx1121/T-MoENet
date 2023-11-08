import os
import os.path as osp
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader

from tqdm import tqdm
import argparse
from datasets.mc_dataset import MC_Dataset, mc_collate_fn
from collections import OrderedDict
from model import build_model, get_tokenizer
from util.misc import get_mask
import json
from collections import defaultdict

def main(args):

    data_name = args.dataset_path.split("/")[-2]
    new_state_dict = OrderedDict()
    ckpt = torch.load(args.model_path)
    cfg = ckpt['args']
    cfg.n_ans = 2
    cfg.max_tokens = 256
    if cfg.add_video_feat:
        cfg.max_feats += 1
    cfg.sample_nums = 10
    for k, v in ckpt['model'].items():
        new_state_dict[k.replace("module.","")] = v

    model = build_model(cfg)
    model.cuda()
    model.eval()
    model.load_state_dict(new_state_dict, strict=False)
    tokenizer = get_tokenizer(cfg)
    type_map={1: "all"}
    dataset = MC_Dataset(
        csv_path=args.dataset_path, 
        features_path=args.feat_path,
        tokenizer=tokenizer,
        type_map=type_map,
        use_context=cfg.use_context,
        subtitles_path = "",
        suffix=cfg.suffix,
        max_feats=cfg.sample_nums
    )
    
    loader = DataLoader(dataset, batch_size = 12, collate_fn=mc_collate_fn, shuffle=False)

    tok_yes = torch.tensor(
                    tokenizer(
                        "Yes",
                        add_special_tokens=False,
                        max_length=1,
                        truncation=True,
                        padding="max_length",
                    )["input_ids"],
                    dtype=torch.long,
                )
    tok_no = torch.tensor(
        tokenizer(
            "No",
            add_special_tokens=False,
            max_length=1,
            truncation=True,
            padding="max_length",
        )["input_ids"],
        dtype=torch.long,
    )     

    a2tok = torch.stack([tok_yes, tok_no])
    model.set_answer_embeddings(
        a2tok.to(model.device), freeze_last=cfg.freeze_last
    )

    res = {}

    for i_batch, batch_dict in enumerate(tqdm(loader)):
        video = batch_dict["video"].cuda()
        video_len = batch_dict["video_len"]
        video_mask = get_mask(video_len, video.size(1)).cuda()
        
        text = batch_dict["text"]
        logits_list = []
        if cfg.add_video_feat:
                video_mask = torch.cat([torch.ones((video.size(0),1)).cuda(),video_mask], dim=1)
        for aid in range(len(text)):
            encoded = tokenizer(
                text[aid],
                add_special_tokens=True,
                max_length=cfg.max_tokens,
                padding="longest",
                truncation=True,
                return_tensors="pt",
            )
            # forward
            output = model(
                video=video,
                video_mask=video_mask,
                input_ids=encoded["input_ids"].cuda(),
                attention_mask=encoded["attention_mask"].cuda(),
            )

            logits = output["logits"]
            # get logits for the mask token
            delay = cfg.max_feats if cfg.use_video else 0
            logits = logits[:, delay : encoded["input_ids"].size(1) + delay][
                encoded["input_ids"] == tokenizer.mask_token_id
            ]
            logits_list.append(logits.softmax(-1)[:, 0])
        
        logits = torch.stack(logits_list, 1)
        if logits.shape[1] == 1:
            preds = logits.round().long().squeeze(1)
        else:
            preds = logits.max(1).indices
        qids = batch_dict["qid"]
        types = batch_dict["type"]
        if batch_dict["answer_id"][0].item() != -1:
            answer_id = batch_dict["answer_id"].cuda()
            agreeings = preds == answer_id

            for i, (qid, gt, pred, type) in enumerate(
                zip(qids, answer_id, preds, types)
            ):
                res[qid] = (
                    {
                        "pred": pred.cpu().detach().item(),
                        "gt": gt.cpu().detach().item(),
                        "type": int(type),
                    }
                    if type_map is not None and len(type_map) > 1
                    else {
                        "pred": pred.cpu().detach().item(),
                        "gt": gt.cpu().detach().item(),
                    }
                )
                res[qid][f"acc"] = agreeings[i].cpu().detach().item()

            dico = {"acc": agreeings.sum() / len(qids)}
        else:
            for i, (qid, pred, type) in enumerate(zip(qids, preds, types)):
                res[str(qid)] = int(pred.cpu().detach().item())

    assert len(res) == len(loader.dataset)
    if isinstance(next(iter(res.values())), dict):
        acc = sum(int(res[qid][f"acc"]) for qid in res) / len(res)
        if type_map is not None and len(type_map) > 1:
            acc_type = {
                type_map[i]: sum(
                    res[qid][f"acc"] for qid in res if res[qid]["type"] == i
                )
                / len([x for x in res.values() if x["type"] == i])
                for i in type_map
            }
            if type_map is not None and len(type_map) > 1:
                for x in acc_type:
                    print(f"acc {x}: {acc_type[x]: .2%}")

        print(data_name)
        print(f"acc: {acc: .2%}")
    
    
    if args.save_result:
        
        if data_name == "starqa":
            submission = defaultdict(list)
            for k, v in res.items():
                qtype = k.split("_")[0]
                submission[qtype].append({"question_id":k, "answer":v['pred']})
            
            json.dump(submission, open(osp.join(args.save_dir, "submission.json"), "w"))

        for k,v in res.items():
            res[str(k)] = v
        json.dump(res, open(osp.join(args.save_dir, "{}.json".format(data_name)), "w"), indent=2)


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

        