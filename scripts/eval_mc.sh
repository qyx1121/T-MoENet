CUDA_VISIBLE_DEVICES=2,3,4,5 python -m torch.distributed.launch --nproc_per_node 4 ori_mc.py --eval \
--combine_datasets nextqa --combine_datasets_val nextqa \
--ds_factor_ff=8 --ds_factor_attn=8 --suffix="." --add_video_feat \
--batch_size_val=16 --max_tokens=256 --load=/mnt/hdd1/qinyixin/FrozenBilm/ckpts/moe_deberta_2_expert/checkpoint0004.pth