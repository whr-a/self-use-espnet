#!/bin/bash
#SBATCH --job-name=41band_oneenc        # 作业名字
#SBATCH --account=bbjs-delta-gpu
#SBATCH --partition=gpuA100x8         # GPU 分区
#SBATCH --nodes=1                      # 1 个节点
#SBATCH --ntasks=1                     # 1 个任务
#SBATCH --gres=gpu:1                   # 请求 1 张 GPU
#SBATCH --cpus-per-task=64              # 每任务 32 核 CPU
#SBATCH --mem=1000G
#SBATCH --time=2-00:00:00              # 最长运行 2 天
##SBATCH --dependency=afterok:10984451
#SBATCH --output=/u/hwang41/reserve/reserve4_gpu_%j.out   # 输出文件

echo "[$(date '+%F %T')] 分配到节点：$SLURM_NODELIST"
cd /u/hwang41/hwang41/3ai/espnet/egs_band/bandcodec/codec1
source ~/.bashrc
conda activate espnet
bash run_41band_oneenc.sh
