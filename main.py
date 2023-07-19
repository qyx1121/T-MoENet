import os
import math
import sys

import torch
import torch.nn
import torch.optim
import numpy as np
import random
import time
import json
import datetime
import argparse
from torch.utils.data import DataLoader, DistributedSampler
from torch.nn.parallel import DistributedDataParallel
from collections import namedtuple

from datasets import build_videotext_dataset, videotext_collate_fn
from model import build_model, get_tokenizer
from util.misc import get_mask, mask_tokens, adjust_learning_rate
from util import dist
from util.metrics import MetricLogger
from args import get_args_parser

from collections import OrderedDict

def train_one_epoch(
    model,
    tokenizer,
    data_loader,
    optimizer,
    epoch,
    args,
    max_norm,
    device_id
):
    model.train()
    metric_logger = MetricLogger(delimiter="  ")
    header = "Epoch: [{}]".format(epoch)
    num_training_steps = int(len(data_loader) * args.epochs)

    for i_batch, batch_dict in enumerate(
        metric_logger.log_every(data_loader, args.print_freq, header)
    ):
        video = batch_dict["video"].to(device_id)
        video_len = batch_dict["video_len"]
        video_mask = get_mask(video_len, video.size(1)).to(device_id)
        if args.add_video_feat:
            video_mask = torch.cat([torch.ones((video.size(0),1)).to(device_id),video_mask], dim=1)
        text = batch_dict["text"]
        encoded = tokenizer(
            text,
            add_special_tokens=True,
            max_length=args.max_tokens,
            padding="longest",
            truncation=True,
            return_tensors="pt",
        )
        inputs, labels = mask_tokens(
            encoded["input_ids"], tokenizer, mlm_probability=args.mlm_prob
        )
        inputs, labels = inputs.to(device_id), labels.to(device_id)

        # forward
        output = model(
            video=video,
            video_mask=video_mask,
            input_ids=inputs,
            attention_mask=encoded["attention_mask"].to(device_id),
            labels=labels,
            train_mode = True,
        )
        mlm_loss = output["loss"]

        # moe_loss
        loss_moe = output["loss_moe"]

        # reduce losses over all GPUs for logging purposes

        if loss_moe is not 0:
            loss = mlm_loss + loss_moe
            loss_dict = {"mlm_loss": mlm_loss, "moe_loss":loss_moe}
        else:
            loss = mlm_loss
            loss_dict = {"mlm_loss": mlm_loss}

        loss_dict_reduced = dist.reduce_dict(loss_dict)
        loss_reduced = sum(loss_dict_reduced.values())
        loss_value = loss_reduced.item()

        if not math.isfinite(loss_value):
            print("Loss is {}, stopping training".format(loss_value))
            print(loss_dict_reduced)
            sys.exit(1)

        optimizer.zero_grad()
        loss.backward()
        if max_norm > 0:
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm)
        optimizer.step()

        adjust_learning_rate(
            optimizer,
            curr_step=epoch * len(data_loader) + i_batch,
            num_training_steps=num_training_steps,
            args=args,
        )

        metric_logger.update(loss=loss_value, **loss_dict_reduced)
    # gather the stats from all processes
    metric_logger.synchronize_between_processes()
    print("Averaged stats:", metric_logger)
    return {k: meter.global_avg for k, meter in metric_logger.meters.items()}


@torch.no_grad()
def evaluate(
    model,
    tokenizer,
    data_loader,
    device_id,
    args,
):
    model.eval()
    metric_logger = MetricLogger(delimiter="  ")
    header = "Val:"

    for i_batch, batch_dict in enumerate(
        metric_logger.log_every(data_loader, args.print_freq, header)
    ):
        video = batch_dict["video"].to(device_id)
        video_len = batch_dict["video_len"]
        video_mask = get_mask(video_len, video.size(1)).to(device_id)
        if args.add_video_feat:
            video_mask = torch.cat([torch.ones((video.size(0),1)).to(device_id),video_mask], dim=1)
        
        text = batch_dict["text"]
        encoded = tokenizer(
            text,
            add_special_tokens=True,
            max_length=args.max_tokens,
            padding="longest",
            truncation=True,
            return_tensors="pt",
        )
        inputs, labels = mask_tokens(
            encoded["input_ids"], tokenizer, mlm_probability=args.mlm_prob
        )
        inputs, labels = inputs.to(device_id), labels.to(device_id)

        # forward
        output = model(
            video=video,
            video_mask=video_mask,
            input_ids=inputs,
            attention_mask=encoded["attention_mask"].to(device_id),
            labels=labels,
        )
        loss = output["loss"]
        # reduce losses over all GPUs for logging purposes
        loss_dict = {"mlm_loss": loss}
        loss_dict_reduced = dist.reduce_dict(loss_dict)
        loss_reduced = sum(loss_dict_reduced.values())
        loss_value = loss_reduced.item()

        metric_logger.update(
            loss=loss_value,
            **loss_dict_reduced,
        )

    return {k: meter.global_avg for k, meter in metric_logger.meters.items()}


def main(args):
    # Init distributed mode
    dist.init_distributed_mode(args)

    if dist.is_main_process():
        if args.save_dir and not (os.path.isdir(args.save_dir)):
            os.mkdir(os.path.join(args.save_dir))
        print(args)

    # Fix seeds
    device_id = dist.get_rank() % torch.cuda.device_count()

    seed = args.seed + dist.get_rank()
    torch.manual_seed(seed)
    np.random.seed(seed)
    random.seed(seed)
    
    # Build model
    model = build_model(args)
    model = model.to(device_id)
    if args.distributed:
        model = DistributedDataParallel(model,device_ids=[device_id])
    
    tokenizer = get_tokenizer(args)

    n_parameters = sum(p.numel() for p in model.parameters() if p.requires_grad)
    if dist.is_main_process():
        print("number of params:", n_parameters)

    # Set up optimizer
    params_for_optimization = list(p for p in model.parameters() if p.requires_grad)
    optimizer = torch.optim.Adam(
        params_for_optimization,
        lr=args.lr,
        betas=(args.beta1, args.beta2),
        weight_decay=args.weight_decay,
    )

    # Set up dataloaders
    if not args.eval:
        if "webvid" in args.combine_datasets:
            dataset_train = build_videotext_dataset("train", args)
            sampler_train = (
                DistributedSampler(dataset_train)
                if args.distributed
                else torch.utils.data.RandomSampler(dataset_train)
            )
            batch_sampler_train = torch.utils.data.BatchSampler(
                sampler_train, args.batch_size, drop_last=True
            )
            dataloader_train = DataLoader(
                dataset_train,
                batch_sampler=batch_sampler_train,
                collate_fn=videotext_collate_fn,
                num_workers=args.num_workers,
            )
        else:
            raise NotImplementedError

    nt = namedtuple(
        typename="data",
        field_names=["dataset_name", "dataloader"],
    )

    tuples = []
    
    if "webvid" in args.combine_datasets_val:
        webvid_dataset_val = build_videotext_dataset("val", args)
        webvid_sampler_val = (
            DistributedSampler(webvid_dataset_val, shuffle=False)
            if args.distributed
            else torch.utils.data.SequentialSampler(webvid_dataset_val)
        )
        webvid_dataloader_val = DataLoader(
            webvid_dataset_val,
            batch_size=args.batch_size_val,
            sampler=webvid_sampler_val,
            collate_fn=videotext_collate_fn,
            num_workers=args.num_workers,
            drop_last=True
        )
        tuples.append(nt(dataset_name="webvid", dataloader=webvid_dataloader_val))
    else:
        raise NotImplementedError
    

    # Load pretrained checkpoint
    if args.load:
        if dist.is_main_process():
            print("loading from", args.load)
        checkpoint = torch.load(args.load, map_location="cpu")
        model.load_state_dict(checkpoint["model"], strict=False)
        if args.resume and not args.eval:
            optimizer.load_state_dict(checkpoint["optimizer"])
            args.start_epoch = checkpoint["epoch"] + 1

    if args.eval:
        val_stats = {}
        for i, item in enumerate(tuples):
            curr_val_stats = evaluate(
                model=model,
                tokenizer=tokenizer,
                data_loader=item.dataloader,
                device_id=device_id,
                args=args,
            )
            val_stats.update(
                {item.dataset_name + "_" + k: v for k, v in curr_val_stats.items()}
            )

        log_stats = {
            **{f"val_{k}": v for k, v in val_stats.items()},
            "epoch": args.start_epoch,
            "n_parameters": n_parameters,
        }

        if args.save_dir and dist.is_main_process():
            json.dump(
                log_stats, open(os.path.join(args.save_dir, "log_stats.json"), "w")
            )
        return

    # Run training and evaluates after every --eval_skip epochs
    if dist.is_main_process():
        print("Start training")
    start_time = time.time()
    for epoch in range(args.start_epoch, args.epochs):
        if dist.is_main_process():
            print(f"Starting epoch {epoch}")
        if args.distributed:
            sampler_train.set_epoch(epoch)

        train_stats = train_one_epoch(
            model=model,
            tokenizer=tokenizer,
            data_loader=dataloader_train,
            optimizer=optimizer,
            epoch=epoch,
            args=args,
            max_norm=args.clip_max_norm,
            device_id=device_id
        )
        if args.save_dir and (epoch + 1) % args.eval_skip == 0:
            checkpoint_path = os.path.join(args.save_dir, f"checkpoint{epoch:04}.pth")
            ori_params = model.state_dict()
            save_params = OrderedDict()
            for k, v in model.named_parameters():
                if v.requires_grad:
                    save_params[k] = ori_params[k]
            dist.save_on_master(
                {
                    "model": save_params,
                    "optimizer": optimizer.state_dict(),
                    "epoch": epoch,
                    "args": args,
                },
                checkpoint_path,
            )

        if (epoch + 1) % args.eval_skip == 0:
            val_stats = {}
            for i, item in enumerate(tuples):
                print(f"Evaluating {item.dataset_name}")

                curr_val_stats = evaluate(
                    model=model,
                    tokenizer=tokenizer,
                    data_loader=item.dataloader,
                    device_id=device_id,
                    args=args,
                )
                val_stats.update(
                    {item.dataset_name + "_" + k: v for k, v in curr_val_stats.items()}
                )
        else:
            val_stats = {}


        log_stats = {
            **{f"train_{k}": v for k, v in train_stats.items()},
            **{f"val_{k}": v for k, v in val_stats.items()},
            "epoch": epoch,
            "n_parameters": n_parameters,
        }

        if args.save_dir and dist.is_main_process():
            with open(os.path.join(args.save_dir, "log.txt"), "a") as f:
                f.write(json.dumps(log_stats) + "\n")

    total_time = time.time() - start_time
    total_time_str = str(datetime.timedelta(seconds=int(total_time)))
    print("Training time {}".format(total_time_str))


if __name__ == "__main__":
    parser = argparse.ArgumentParser(parents=[get_args_parser()])
    args = parser.parse_args()
    if args.save_dir:
        args.save_dir = os.path.join(args.presave_dir, args.save_dir)
    main(args)
