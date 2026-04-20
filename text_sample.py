"""
Generate a large batch of image samples from a model and save them as a large
numpy array. This can be used to produce samples for FID evaluation.
"""
import argparse
import os, json
from rdkit import Chem
import torch as th
import torch.distributed as dist
from transformers import set_seed
from improved_diffusion.rounding import rounding_func, load_models, load_tokenizer
from improved_diffusion import gaussian_diffusion as gd
from improved_diffusion.respace import SpacedDiffusion, space_timesteps
from improved_diffusion import dist_util, logger
from improved_diffusion.transformer_model2 import TransformerNetModel2
from improved_diffusion.test_util import get_weights, denoised_fn_round

from improved_diffusion import dist_util, logger
from functools import partial
from improved_diffusion.script_util import (
    NUM_CLASSES,
    model_and_diffusion_defaults,
    add_dict_to_argparser,
    args_to_dict,
)
from mydatasets import get_dataloader,ChEBIdataset
os.environ["CUDA_VISIBLE_DEVICES"] = "0"

def main():
    set_seed(121)   # 设置随机种子以确保实验结果的可重复性
    args = create_argparser().parse_args()  # 从命令行参数中解析用户配置

    # dist_util.setup_dist()
    logger.configure()  # 配置日志记录器（例如，记录生成信息）
    args.sigma_small = True      # 设置扩散模型的sigma参数，这可能与生成数据的动态范围相关

    # args.diffusion_steps = 200 #500  # DEBUG
    # 修正实验名称，如果指定为 `random1`，替换为 `random`
    if args.experiment == 'random1': args.experiment = 'random'
    logger.log("creating model and diffusion...")   # 日志记录：显示模型和扩散配置的创建信息
    from mytokenizers import regexTokenizer
    tokenizer = regexTokenizer()
    # 创建模型对象，参数配置如词汇表大小、模型隐藏层大小等
    model = TransformerNetModel2(
        in_channels=32,  # 3, DEBUG** # 输入通道数（这里表示嵌入维度）
        # deep_channels = 10,
        model_channels=128,  # 模型通道数（Transformer的隐藏层大小）
        dropout=0.1,     # dropout比例，防止过拟合
        use_checkpoint=False,   # 是否使用梯度检查点，降低内存开销
        config_name='bert-base-uncased',    # 使用的Transformer基础配置
        training_mode='e2e',    # 训练模式，end-to-end
        vocab_size=len(tokenizer),  # 分词器的词汇表大小
        experiment_mode='lm',    # 实验模式，语言模型
        logits_mode=1,  # logits模式，可能与生成目标有关
        hidden_size = 1024, # 隐藏层大小
        num_attention_heads=16, # 多头注意力的头数
        num_hidden_layers = 12, # Transformer中的层数
    )
    # 创建扩散模型对象，配置扩散过程中的时间步数和损失类型
    diffusion = SpacedDiffusion(
        use_timesteps=[i for i in range(0,2000,10)],     # 时间步间隔
        betas=gd.get_named_beta_schedule('sqrt', 2000),  # beta调度方式（平方根）
        model_mean_type=(
             gd.ModelMeanType.START_X
        ),  # 模型输出的均值类型
        model_var_type=(
            (
                gd.ModelVarType.FIXED_LARGE
            )
        ),  # 固定大的方差类型
        loss_type=gd.LossType.E2E_MSE,   # 损失类型为端到端均方误差
        rescale_timesteps=True, # 是否重缩放时间步
        model_arch='transformer',   # 模型结构为Transformer
        training_mode='e2e',    # 训练模式，end-to-end
    )
    # 打印模型路径以确认加载正确的参数文件
    # print(args.model_path)
    # 加载模型的预训练权重
    model.load_state_dict(
        dist_util.load_state_dict(args.model_path, map_location="cpu")
    )
    # 计算模型的参数总量并记录到日志
    pytorch_total_params = sum(p.numel() for p in model.parameters())
    logger.log(f'the parameter count is {pytorch_total_params}')

    # diffusion.rescale_timesteps = False  # DEBUG --> REMOVE   # 打印扩散模型是否重缩放时间步，调试用标志
    print(diffusion.rescale_timesteps, 'a marker for whether we are in the debug mode')
    # 将模型加载到指定的设备上（通常是GPU），并设置为评估模式
    model.to(dist_util.dev())
    model.eval() # DEBUG
    # 日志记录：开始采样过程
    logger.log("sampling...")
    print(args.num_samples) # 打印采样的目标数量
    # model3 = get_weights(model2, args)
    # 日志记录：加载训练数据集的指定分割集
    print('--'*30)
    print('loading {} set'.format(args.split))
    print('--'*30)

    train_dataset = ChEBIdataset(
        # dir='/DoctoralStudents/yuanhang/AI_drug/DATA/new_ligands/',
        dir='/nfs/home/yuanhang/AI_drug/DATA/',
        smi_tokenizer=tokenizer,
        split=args.split,    # 数据集分割（训练/验证/测试）
        replace_desc=False  # 是否替换描述符
        # pre = pre
    )
    print('DATASETINFO-----------------------------')
    print(len(train_dataset),(train_dataset[0]['desc_state'].shape))
    # 提取数据集中的描述状态、掩码和分子SMILES字符串
    desc = [(train_dataset[i]['desc_state'],train_dataset[i]['desc_mask'],train_dataset[i]['smiles']) for i in range(args.num_samples)]
    answer = [i[2] for i in desc]   # 保存对应的原始SMILES字符串
    # 克隆模型的词嵌入权重，用于后续处理（不需要梯度更新）
    model3 = th.nn.Parameter(model.word_embedding.weight.clone().cpu())
    model3.requires_grad = False

    # 初始化生成的样本存储列表
    allsample = []
    num_done = 0    # 已生成样本计数
    # 逐批生成样本，直到达到指定的样本总量
    while num_done < args.num_samples:
        # 计算当前批次的样本范围
        idend = min(num_done+args.batch_size,args.num_samples)
        print('acquiring  {} : {}'.format(num_done,idend))
        # 合并当前批次的描述状态和掩码
        desc_state = th.concat([i[0] for i in desc[num_done:idend]],dim=0)
        desc_mask = th.concat([i[1] for i in desc[num_done:idend]],dim=0)
        # 初始化模型参数字典
        model_kwargs = {}
        print('use_ddim:{}',args.use_ddim)  # 日志记录：采样方式（是否使用DDIM采样）
        # 根据采样方法选择扩散采样函数
        # print(dir(diffusion))
        sample_fn = (
            diffusion.p_sample_loop if not args.use_ddim else diffusion.ddim_sample_loop
        )
        # 定义采样的目标形状,暂时修改为128，之前256
        sample_shape = (idend-num_done, 258, model.in_channels)
        print(sample_shape)
        # 调用采样函数生成样本
        sample = sample_fn(
            model,
            sample_shape,
            clip_denoised=args.clip_denoised,   # 是否裁剪去噪结果
            denoised_fn = None, # 去噪函数（未指定）
            model_kwargs=model_kwargs,  # 模型参数字典
            top_p =args.top_p,  # 核采样参数
            progress = True,    # 显示进度条
            desc = (desc_state,desc_mask)   # 描述符
        )
        # 将当前批次的样本添加到总列表中
        allsample.append(sample)
        num_done = idend    # 更新已完成的样本计数
    # 将所有批次的样本合并成一个完整的Tensor
    sample = th.concat(allsample,dim=0)
    # 日志记录：解码生成的样本
    print('decoding for e2e', )
    print(sample.shape)
    # 将样本转换为GPU Tensor
    # x_t = th.tensor(sample).cuda()

    # 转换为 PyTorch tensor，并移动到 GPU
    device = th.device("cuda" if th.cuda.is_available() else "cpu")
    x_t = th.as_tensor(sample, dtype=th.float32).to(device)

    # 通过模型提取 logits 并取 Top-1 的索引
    reshaped_x_t = x_t
    logits = model.get_logits(reshaped_x_t)  # bsz, seqlen, vocab
    cands = th.topk(logits, k=1, dim=-1)
    sample = cands.indices
    sample = sample.squeeze(-1)
    # 使用分词器解码生成的序列
    print(sample)
    from mytokenizers import regexTokenizer
    tokenizer = regexTokenizer()
    c = tokenizer.decode(sample)
    # 将生成的分子SMILES与真实答案保存到文件
    with open(args.outputdir,'w') as f:
        for i,x in enumerate(c):
            if i==0:
                print(x)
            f.write(x.replace('[PAD]','')+'   ||   '+answer[i]+'\n')

    # 检查生成的分子是否有效，并记录无效分子
    with open(args.outputdir) as f:
        allsmiles = [k.strip().split('||')[0].strip().replace('[EOS]','').replace('[SOS]','') for k in f.readlines()]
    f = open('/nfs/home/yuanhang/AI_drug/DATA/new_ligands/test_results/generation_selfies_90cos_258_AD_case_8W.txt','w')
    for cnt,s in enumerate(allsmiles):
        mol = Chem.MolFromSmiles(s)
        if mol is None:
            f.write(str(cnt)+'\t'+s+'\n')
    f.close()

def create_argparser():
    defaults = dict(
        clip_denoised=False,    # 是否裁剪去噪结果，默认为False
        num_samples=12000,#10000,   # 样本数量，默认为50
        batch_size=256,
        use_ddim=False,      # 是否使用DDIM采样，默认为False
        mbr_sample=1,        # MBR采样数量，默认为1
        model_path="",      # 模型的路径，默认为空字符串
        model_arch='conv-unet',     # 模型架构，默认为'conv-unet'
        verbose='yes',      # 是否打印详细信息，默认为'yes'
        out_dir="diffusion_lm/improved_diffusion/out_gen"    # 输出目录，默认为此路径
    )
    # 定义文本任务相关的默认参数配置
    text_defaults = dict(modality='text',    # 数据的模态，默认为'text'
                         dataset_name='wikitext',   # 数据集的名称，默认为'wikitext'
                         dataset_config_name='wikitext-2-raw-v1',   # 数据集的配置名称
                         model_name_or_path='predictability/diff_models/compress_e=5_b=60_m=gpt2_wikitext-103-raw-v1_None', # 预训练模型路径
                         experiment='gpt2_pre_compress', # 实验名称，默认为'gpt2_pre_compress',
                         model_arch='trans-unet',   # 模型架构，默认为'trans-unet'
                         preprocessing_num_workers=1,   # 数据预处理的工作线程数量
                         emb_scale_factor=1.0,   # 嵌入缩放因子，默认为1.0
                         clamp='clamp',  # 是否对输出进行裁剪，默认为'clamp'
                         split = 'validation',    # 数据集划分，默认为'test'
                         model_path='/nfs/home/yuanhang/AI_drug/tgm-dlm-main/checkpoints_selfies_30cos_258_20W/PLAIN_ema_0.9999_080000.pt',
                         use_ddim=False,     # 是否使用DDIM采样，默认为False
                         batch_size =256,
                         num_samples=12000,  # 样本数量，
                         top_p =1.0,    # top_p值，默认为1.0
                         out_dir='generation_outputs',  # 生成结果的输出目录
                         outputdir='/nfs/home/yuanhang/AI_drug/DATA/new_ligands/test_results/textguidtry_selfies_30cos_258_AD_case_8w.txt'    # 输出文件路径
                         )
    # 更新defaults字典，加入模型和扩散的默认参数配置
    defaults.update(model_and_diffusion_defaults())
    defaults.update(text_defaults)  # 进一步更新默认参数配置，加入text_defaults
    # defaults.update(model_and_diffusion_defaults())
    parser = argparse.ArgumentParser()  # 创建ArgumentParser对象
    add_dict_to_argparser(parser, defaults) # 将defaults字典中的参数加入解析器中
    return parser


if __name__ == "__main__":
    import os
    os.environ['CUDA_DEVICES_ORDER'] = 'PCI_BUS_ID'  # 设置CUDA设备顺序
    os.environ['CUDA_VISIBLE_DEVICES'] = '0'  # 设置使用的CUDA设备
    main()
