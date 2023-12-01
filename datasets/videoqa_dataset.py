import torch as th
from torch.utils.data import Dataset
from torch.utils.data.dataloader import default_collate
import pandas as pd
import collections
import json
import pickle
from tqdm import tqdm
import math


class VideoQA_Dataset(Dataset):
    def __init__(
        self,
        csv_path,
        features_path,
        max_feats=10,
        features_dim=768,
        vocab_path=None,
        train=False,
        prefix="",
        suffix=".",
        tokenizer=None,
        type_map=None,
    ):
        self.data = pd.read_csv(csv_path)  
        
        self.features = th.load(features_path)
        self.max_feats = max_feats
        self.features_dim = features_dim
        self.a2id = json.load(open(vocab_path, "r"))
        self.id2a = {v:k for k,v in self.a2id.items()}
        self.id2a[-1] = ' '
        self.train = train
        self.prefix = prefix
        self.suffix = suffix
        self.mask = tokenizer.mask_token
        self.type_map = type_map
        if train:  # for training remove answers that are not in vocab
            print(len(self.data))
            ok = []
            for i, row in tqdm(self.data.iterrows()):
                if "answer" in self.data:
                    answer = row["answer"]
                else:
                    answer = collections.Counter(
                        [
                            row["answer1"],
                            row["answer2"],
                            row["answer3"],
                            row["answer4"],
                            row["answer5"],
                        ]
                    )
                    answer = answer.most_common(1)[0][0]
                if answer in self.a2id:
                    ok.append(i)
            self.data = self.data[self.data.index.isin(ok)]
            print(len(self.data))

    def __len__(self):
        return len(self.data)

    def _get_text(self, question, mask):
        text = (
            f"{self.prefix} Question: {question} Answer: {mask}{self.suffix}"
        )
        text = text.strip()
        return text

    def _get_video(self, video_id):
        if video_id not in self.features:
            print(video_id)
            video = th.zeros(1, self.features_dim)
        else:
            video = self.features[video_id].float()
        if len(video) > self.max_feats:
            sampled = []
            for j in range(self.max_feats):
                sampled.append(video[(j * len(video)) // self.max_feats])
            video = th.stack(sampled)
            video_len = self.max_feats
        elif len(video) < self.max_feats:
            video_len = len(video)
            video = th.cat(
                [video, th.zeros(self.max_feats - video_len, self.features_dim)], 0
            )
        else:
            video_len = self.max_feats

        return video, video_len

    def __getitem__(self, idx):
        # get question
        question = self.data["question"].values[idx].capitalize().strip()
        if question[-1] != "?":
            question = str(question) + "?"
        type = 0
        if "type" in self.data:
            type = self.data["type"].values[idx]

        # get answer
        if "answer" in self.data:
            answer = self.data["answer"].values[idx]
            answer_id = self.a2id.get(answer, -1)
        else:  # iVQA
            answer = collections.Counter(
                [
                    self.data["answer1"].values[idx],
                    self.data["answer2"].values[idx],
                    self.data["answer3"].values[idx],
                    self.data["answer4"].values[idx],
                    self.data["answer5"].values[idx],
                ]
            )
            answer_id = th.zeros(len(self.a2id))
            for x in answer:
                if x in self.a2id:
                    answer_id[self.a2id[x]] = answer[x]
            final = []
            for x in answer:
                if answer[x] >= 2:
                    final.extend([x] * 2)
                else:
                    final.append(x)
            answer = final

        video_id = self.data["video_id"].values[idx]

        # get pattern
        text = self._get_text(question, self.mask)

        # get video
        start = None
        end = None
        if "start" in self.data.columns:
            start = self.data["start"].values[idx]
            end = self.data["end"].values[idx]
        video, video_len = self._get_video(video_id)
        return {
            "video_id": video_id,
            "video": video,
            "video_len": video_len,
            "text": text,
            "qid": idx,
            "answer_id": answer_id,
            "type": type,
            "answer": answer,
        }


def videoqa_collate_fn(batch):
    bs = len(batch)
    video = th.stack([batch[i]["video"] for i in range(bs)])
    video_len = th.tensor([batch[i]["video_len"] for i in range(bs)], dtype=th.long)
    text = (
        [batch[i]["text"] for i in range(bs)]
        if isinstance(batch[0]["text"], str)
        else [
            [batch[i]["text"][j] for i in range(bs)]
            for j in range(len(batch[0]["text"]))
        ]
    )
    qid = [batch[i]["qid"] for i in range(bs)]
    answer_id = default_collate([batch[i]["answer_id"] for i in range(bs)])
    type = [batch[i]["type"] for i in range(bs)]
    answer = [batch[i]["answer"] for i in range(bs)]
    video_id =[batch[i]["video_id"] for i in range(bs)]
    out = {
        "video_id":video_id,
        "video": video,
        "video_len": video_len,
        "text": text,
        "qid": qid,
        "answer_id": answer_id,
        "type": type,
        "answer": answer,
    }
    return out


def build_videoqa_dataset(dataset_name, split, args, tokenizer):
    if dataset_name == "msvd":
        if split == "train":
            csv_path = args.msvd_train_csv_path
        elif split == "val":
            csv_path = args.msvd_val_csv_path
        elif split == "test":
            csv_path = args.msvd_test_csv_path
        else:
            raise NotImplementedError
        features_path = args.msvd_features_path
        vocab_path = args.msvd_vocab_path
        type_map = {0: "what", 1: "how", 2: "color", 3: "where", 4: "who", 5: "when"}
    
    elif dataset_name == "msrvtt":
        if split == "train":
            csv_path = args.msrvtt_train_csv_path
        elif split == "val":
            csv_path = args.msrvtt_val_csv_path
        elif split == "test":
            csv_path = args.msrvtt_test_csv_path
        else:
            raise NotImplementedError
        features_path = args.msrvtt_features_path
        vocab_path = args.msrvtt_vocab_path
        type_map = {0: "what", 1: "how", 2: "color", 3: "where", 4: "who", 5: "when"}
    
    elif dataset_name == "ivqa":
        if split == "train":
            csv_path = args.ivqa_train_csv_path
        elif split == "val":
            csv_path = args.ivqa_val_csv_path
        elif split == "test":
            csv_path = args.ivqa_test_csv_path
        else:
            raise NotImplementedError
        features_path = args.ivqa_features_path
        vocab_path = args.ivqa_vocab_path
        type_map = None
    
    elif dataset_name == "tgif":
        if split == "train":
            csv_path = args.tgif_frameqa_train_csv_path
        elif split == "val":
            csv_path = args.tgif_frameqa_test_csv_path  # no val set in TGIF
        elif split == "test":
            csv_path = args.tgif_frameqa_test_csv_path
        else:
            raise NotImplementedError
        features_path = args.tgif_features_path
        vocab_path = args.tgif_vocab_path
        type_map = {0: "what", 1: "how", 2: "color", 3: "where"}

    if dataset_name in ["msvd", "msrvtt", "ivqa", "tgif"]:
        return VideoQA_Dataset(
            csv_path=csv_path,
            features_path=features_path,
            max_feats=args.sample_nums,
            features_dim=args.features_dim,
            vocab_path=vocab_path,
            train=split == "train",
            prefix=args.prefix,
            suffix=args.suffix,
            tokenizer=tokenizer,
            type_map=type_map,
        )
    else:
        raise NotImplementedError
