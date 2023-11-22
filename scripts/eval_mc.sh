CUDA_VISIBLE_DEVICES=7 python inference_mc.py --dataset_path /mnt/hdd3/qinyixin/FrozenBilm/NEXT-QA/val.csv \
--feat_path /mnt/hdd3/qinyixin/FrozenBilm/NEXT-QA/clipvitl14.pth \
--model_path /mnt/hdd1/qinyixin/FrozenBilm/ckpts/deberta_moe_ffn_first_2/checkpoint0004.pth --batch_size=12 \