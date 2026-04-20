"""
Train a diffusion model on images.
"""

import argparse
import json, torch, os
from improved_diffusion import gaussian_diffusion as gd
from improved_diffusion.respace import SpacedDiffusion, space_timesteps
import numpy as np
from improved_diffusion import dist_util, logger
from improved_diffusion.transformer_model2 import TransformerNetModel2
from improved_diffusion.image_datasets import load_data
from improved_diffusion.text_datasets import load_data_text
from improved_diffusion.resample import create_named_schedule_sampler
from improved_diffusion.script_util import (
    model_and_diffusion_defaults,
    add_dict_to_argparser,
)
from transformers import AutoTokenizer
from improved_diffusion.train_util import TrainLoop
from transformers import set_seed
from functools import partial
from improved_diffusion.test_util import get_weights, compute_logp
from improved_diffusion.rounding import load_models, load_tokenizer
import torch.distributed as dist
import wandb
from mytokenizers import regexTokenizer
from mydatasets import get_dataloader,ChEBIdataset
import warnings
import torch.multiprocessing as mp
warnings.filterwarnings("ignore")

wandb.login(key='7691a1d03baeffaf7f44a78cb0de4b37dd55a78c')
def main_worker(rank,world_size):
    args = create_argparser().parse_args()  # 解析命令行参数
    set_seed(args.seed)     # 设置随机种子，以便实验可重复
    # 如果是 rank 0 进程（即主进程），初始化 wandb 并打印配置信息
    if rank == 0:
        wandb.init(
            project = "DiffusionLMRegexAug_selfies_258_gen",    # 项目名称
            config = args.__dict__,  # 配置使用的命令行参数
        )
        print(wandb.config)

    # 初始化分布式训练，设置不同进程的通信
    dist_util.setup_dist(rank,world_size) 
    print("creating model and diffusion...")
    # 初始化分子SMILES字符串的正则化分词器（最大序列长度为256）
    smtokenizer = regexTokenizer(max_len=258)

    print('load data', '*' * 50)
    # 初始化数据集
    train_dataset = ChEBIdataset(
        dir='/nfs/home/yuanhang/AI_drug/DATA/',  # 数据集所在目录
        smi_tokenizer=smtokenizer,  # 使用的分子分词器
        split='train',  # 使用训练集
        replace_desc=False,  # 不替换描述
        corrupt_prob=0.,  # 不引入数据腐蚀
        mask_desc=False,  # 不掩盖描述,
        # pre = pre
    )
    gene_num = train_dataset.gene_num
    print(f"🔹 训练数据集基因数量: {gene_num}")
    print('In total', len(train_dataset), 'for training....')

    model = TransformerNetModel2(       # 初始化模型，设置各项超参数
        in_channels=args.model_in_channels,  # 3, DEBUG**   # 输入通道数
        # deep_channels = 10,
        model_channels=args.model_model_channels,       # 模型通道数
        dropout=args.model_dropout,     # Dropout比例
        use_checkpoint=False,       # 是否使用模型检查点
        config_name='bert-base-uncased',     # 使用的预训练BERT模型
        training_mode='e2e',     # 训练模式
        vocab_size=len(smtokenizer),     # 词汇表大小
        experiment_mode='lm',   # 实验模式（语言模型）
        logits_mode=1,      # 输出logits的模式
        hidden_size = args.model_hidden_size,       # 隐藏层大小
        num_attention_heads=args.model_num_attention_heads,     # 注意力头数量
        num_hidden_layers = args.model_num_hidden_layers,       # 隐藏层数量
        gene_num=gene_num
    )
    # 如果是 rank 0 进程，打印模型参数数量和分词器的词汇表长度
    if rank==0:
        # 计算模型的总参数数量
        pytorch_total_params = sum(p.numel() for p in model.parameters())
        print('Model total prameter number is :', pytorch_total_params)
        print('Smiles tokenizer vocab length:',len(smtokenizer))
    # 初始化扩散模型（SpacedDiffusion）
    diffusion = SpacedDiffusion(
        use_timesteps=[i for i in range(2000)],     # 使用2000个时间步
        betas=gd.get_named_beta_schedule('sqrt', 2000),     # 使用平方根衰减的beta调度
        # 使用开始时的图像作为均值
        model_mean_type=(
             gd.ModelMeanType.START_X
        ),
        # 使用固定的大方差
        model_var_type=(
            (
                gd.ModelVarType.FIXED_LARGE
            )
        ),
        # 使用端到端的均方误差作为损失函数
        loss_type=gd.LossType.E2E_MSE,
        rescale_timesteps=True,
        model_arch='transformer',   # 模型架构选择为Transformer
        training_mode='e2e',    # 训练模式为端到端
    )
    # 创建调度采样器，使用均匀采样
    schedule_sampler = create_named_schedule_sampler('uniform', diffusion)

    # 获取数据加载器，用于批量加载数据
    dataloader = get_dataloader(train_dataset,args.batch_size,rank,world_size)
    # 验证数据集，这里设置为 None，表示没有验证数据
    data_valid = None
    TrainLoop(      # 初始化训练循环，传入各种参数和超参数
        model=model,
        diffusion=diffusion,
        data=dataloader,
        batch_size=args.batch_size,     # 批量大小
        microbatch=args.microbatch,     # 微批量大小
        lr=args.lr,     # 学习率
        ema_rate=args.ema_rate,     # EMA (指数滑动平均)的衰减率
        log_interval=args.log_interval,     # 日志记录间隔
        save_interval=args.save_interval,       # 模型保存间隔
        resume_checkpoint=args.resume_checkpoint,   # 是否恢复检查点
        use_fp16=args.use_fp16,     # 是否使用16位浮点数
        fp16_scale_growth=args.fp16_scale_growth,   # FP16模式的规模增长
        schedule_sampler=schedule_sampler,   # 采样调度器
        weight_decay=args.weight_decay,     # 权重衰减
        lr_anneal_steps=args.lr_anneal_steps,       # 学习率衰减的步数
        checkpoint_path=args.checkpoint_path,       # 模型检查点路径
        gradient_clipping=args.gradient_clipping,    # 梯度裁剪
        eval_data=data_valid,        # 验证数据
        eval_interval=args.eval_interval        # 验证间隔
    ).run_loop()
    dist.destroy_process_group()    # 结束训练后销毁分布式训练进程组


def create_argparser():
    defaults = dict()   # 初始化默认参数字典
    text_defaults = dict(       # 配置文本模型的默认参数
        attention_resolutions='16,8',    # 注意力分辨率
        batch_size=8,      # 批量大小
        cache_mode='no',    # 是否启用缓存模式
        checkpoint_path='/nfs/home/yuanhang/AI_drug/tgm-dlm-main/checkpoints_selfies_30cos_258_13755_10W',   # 检查点路径
        # checkpoint_path='/DoctoralStudents/yuanhang/AI_drug/my_model/checkpoints_selfies_gen_258_30W',  # 检查点路径
        class_cond=False,       # 是否使用类别条件
        commonGen_train='diffusion_lm/common-gen/commongen_data',    # CommonGen数据路径
        config='ll',    # 配置类型
        config_name='bert-base-uncased',    # 配置名称
        data_dir='',    # 数据路径
        dataset_config_name='wikitext-2-raw-v1',    # 数据集配置名称
        dataset_name='wikitext',    # 数据集名称
        diffusion_steps=2000,   # 扩散步数
        dropout=0.1,    # Dropout比例
        e2e_train='',   # 端到端训练模式
        ema_rate='0.9999',  #指数移动平均，用于提高模型的稳定性和生成质量，一般设为0.9-0.999
        emb_scale_factor=1.0,   # 嵌入缩放因子
        eval_interval=2000,     # 验证间隔
        experiment='random',    # 实验名称
        experiment_mode='lm',   # 实验模式：语言模型
        fp16_scale_growth=0.001, # FP16模式规模增长
        gradient_clipping=2.4,  # 梯度裁剪阈值
        image_size=8,   # 图像尺寸（可能是图像生成模型时的配置）
        in_channel=16,  # 输入通道数
        learn_sigma=False,  # 是否学习sigma
        log_interval=2000,  # 日志记录间隔
        logits_mode=1,      # logits模式
        lr = 0.00005,       # 学习率
        # lr=0.0001, # Lower the learing rate in 2024/5/5 for better convergence 
        lr_anneal_steps=200000,     # 学习率衰减步数
        microbatch=-1,      # 微批量大小
        modality='e2e-tgt',         # 模式类型
        model_arch='transformer',   # 模型架构
        model_name_or_path='predictability/diff_models/compress_e=5_b=60_m=gpt2_wikitext-103-raw-v1_None',  # 预训练模型路径
        noise_level=0.0,    # 噪声水平
        noise_schedule='sqrt',  # 噪声调度方式
        num_channels=128,   # 通道数
        num_heads=4,        # 注意力头数
        num_heads_upsample=-1,
        num_res_blocks=2, # 残差块数
        out_channel=16,     # 输出通道数
        padding_mode='pad',     # 填充模式
        predict_xstart=True,    # 是否预测xstart
        preprocessing_num_workers=1,    # 数据预处理的线程数
        rescale_learned_sigmas=True,     # 是否重新缩放学习到的sigmas
        rescale_timesteps=True,     # 是否重新缩放时间步
        resume_checkpoint='',       # 恢复的检查点
        roc_train='diffusion_lm/ROCstory',
        save_interval=20000,    # 模型保存间隔
        schedule_sampler='uniform',  # 采样调度器
        seed=19991009,  # 随机种子
        sigma_small=False,  # 是否使用小sigma
        timestep_respacing='',      # 时间步重新定标
        training_mode='e2e',         # 训练模式
        use_bert_tokenizer='no',    # 是否使用BERT分词器
        use_checkpoint=False,
        use_fp16=False,         # 是否使用FP16
        use_kl=False,   #是否使用KL散度
        use_scale_shift_norm=True,
        weight_decay=0.0,   # 权重衰减
        model_in_channels = 32, # 模型输入通道数
        model_model_channels = 128,      # 模型通道数
        model_dropout = 0.1,    # 模型的Dropout比例
        model_hidden_size = 1024,   # 模型隐藏层大小
        model_num_attention_heads = 16,     # 模型的注意力头数
        model_num_hidden_layers = 12        # 模型隐藏层数
    )
    # 更新默认参数
    defaults.update(model_and_diffusion_defaults())
    defaults.update(text_defaults)
    parser = argparse.ArgumentParser()  # 创建命令行解析器
    add_dict_to_argparser(parser, defaults) # 将参数字典添加到解析器
    return parser


if __name__ == "__main__":
    import os
    os.environ['CUDA_DEVICES_ORDER']='PCI_BUS_ID'   # 设置CUDA设备顺序
    os.environ['CUDA_VISIBLE_DEVICES']='0'  # 设置使用的CUDA设备
    os.environ["WANDB_MODE"] = "dryrun"  # 只记录本地，但不上传
    world_size=1        #如果你只有 1 张显卡，world_size=1 是完全可以的，分布式训练框架仍然会被启用，但只有一个进程来处理所有任务
    mp.spawn(main_worker,args=(world_size,),nprocs=world_size,join=True)
