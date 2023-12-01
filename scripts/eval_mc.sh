CUDA_VISIBLE_DEVICES=7 python inference_mc.py --dataset_path /mnt/hdd3/qinyixin/T-MoENet/NEXT-QA/val.csv \
--feat_path /mnt/hdd3/qinyixin/T-MoENet/NEXT-QA/clipvitl14.pth \
--model_path /mnt/hdd1/qinyixin/T-MoENet/ckpts/deberta_moe_ffn_first_2/checkpoint0004.pth --batch_size=12 \