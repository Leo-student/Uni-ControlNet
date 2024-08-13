将两个模型结合到一起
CUDA_VISIBLE_DEVICES=0 CUDA_LAUNCH_BLOCKING=1   python utils/prepare_weights.py integrate log_local/lightning_logs/version_2/checkpoints/epoch\=166-step\=500.ckpt ckpt/init_global.ckpt  configs/uni_v15.yaml ckpt/flare.ckpt
CUDA_VISIBLE_DEVICES=0 CUDA_LAUNCH_BLOCKING=1   python utils/prepare_weights.py integrate log_local/lightning_logs/version_8/checkpoints/epoch\=33-step\=100.ckpt ckpt/init_global.ckpt  configs/uni_v15.yaml ckpt/flare2.ckpt
CUDA_VISIBLE_DEVICES=0 CUDA_LAUNCH_BLOCKING=1   python utils/prepare_weights.py integrate log_local/lightning_logs/version_8/checkpoints/epoch\=33-step\=100.ckpt log_global/lightning_logs/version_5/checkpoints/epoch_333-step_1000.ckpt  configs/uni_v15.yaml ckpt/flare3.ckpt
得到的ckpt为 flare.ckpt

test 
python src/test/test.py 生成 新的图像与结果 

train

是有监督学习，需要有gt！放在data/images文件夹下

如何确定condition 是有效的 ？
训练时为了简化不保存实体的npy数组，进行的操作：

#train中
__init__.py中
class ContentDetector: 

返回值为 _io.BytesIO 

在dataset中 
class UniDataset(Dataset): 
将指针转到 _io.BytesIO 的开头 
if global_file is not None:
                content_emb = apply_content(content_image)
            else:
                content_emb = np.zeros((768))
            content_emb.seek(0)
condition = np.load(content_emb, allow_pickle=True)
global_conditions.append(condition)

#test 是将global condition 









