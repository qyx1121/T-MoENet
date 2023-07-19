CUDA_VISIBLE_DEVICES=4,5,6,7 python -m torch.distributed.launch --master_port=1121 --nproc_per_node 4 main.py \
--combine_datasets webvid --combine_datasets_val webvid --save_dir=/mnt/hdd1/qinyixin/FrozenBilm/ckpts/moe_deberta_2_expert/ \
--lr=3e-5 --ds_factor_ff=8 --ds_factor_attn=8 --add_video_feat \
--batch_size=32 --batch_size_val=16 --epochs=5 --webvid_features_path /mnt/hdd3/qinyixin/feats \
--eval_skip 1 --webvid_train_csv_path data/exp_2/train_2m.csv