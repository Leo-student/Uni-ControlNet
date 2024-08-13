# CUDA_VISIBLE_DEVICES=0 python src/train/train.py  --resume-path ./ckpt/init_local.ckpt 
# CUDA_VISIBLE_DEVICES=0 python src/train/train.py --config-path ./configs/global_v15.yaml --resume-path ./ckpt/init_global.ckpt --logdir ./log_global/
CUDA_VISIBLE_DEVICES=0,1 python src/train/train.py --config-path ./configs/local_v15.yaml --resume-path ./ckpt/init_local.ckpt --logdir ./log_local/