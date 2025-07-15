import torch

# 1. 用 map_location='cpu' 把模型载到 CPU（无 GPU 也能用）
ckpt = torch.load("/u/hwang41/hwang41/3ai/espnet/egs_band/bandcodec/codec1/exp/codec_pretrain_encoder_5bands_oneencdec_raw_fs24000/696epoch_eval.pth", map_location="cpu")

# 2. 看一下顶层结构
print(type(ckpt))
if isinstance(ckpt, dict):
    print("Keys in checkpoint:")
    for k in ckpt.keys():
        print(" ", k)
else:
    print("Checkpoint is not a dict, it's:", ckpt)

# 3. 如果它是个 dict，查看 state_dict 里的每个张量名和形状
if "model" in ckpt:
    sd = ckpt["model"]
elif "state_dict" in ckpt:
    sd = ckpt["state_dict"]
else:
    # 如果就是直接存的 state_dict
    sd = ckpt

print("\nLayers and tensor shapes:")
for name, tensor in sd.items():
    print(f"  {name:<50} {tuple(tensor.shape)}")