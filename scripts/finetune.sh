CUDA_VISIBLE_DEVICES=2,3 python -m torch.distributed.launch --master_port=1121 --nproc_per_node 2 mc.py \
--combine_datasets starqa --combine_datasets_val starqa --save_dir=/mnt/hdd1/qinyixin/FrozenBilm/ckpts/ft_starqa/ \
--lr=5e-5 --schedule=linear_with_warmup --load=/mnt/hdd1/qinyixin/FrozenBilm/ckpts/deberta_moe_ffn_first_2/checkpoint0004.pth \
--ds_factor_ff=8 --ds_factor_attn=8 --add_video_feat \
--batch_size=16 --batch_size_val=32 --max_tokens 256 --epochs=10 --eval_skip 1