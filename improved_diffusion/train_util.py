import copy
import functools
import os

import blobfile as bf
import torch as th
import torch.distributed as dist
from torch.nn.parallel.distributed import DistributedDataParallel as DDP
from torch.optim import AdamW

from . import dist_util, logger
from .fp16_util import (
    make_master_params,
    master_params_to_model_params,
    model_grads_to_master_grads,
    unflatten_master_params,
    zero_grad,
)
from .nn import update_ema
from .resample import LossAwareSampler, UniformSampler
import wandb
# For ImageNet experiments, this was a good default value.
# We found that the lg_loss_scale quickly climbed to
# 20-21 within the first ~1K steps of training.
INITIAL_LOG_LOSS_SCALE = 20.0


class TrainLoop:
    def __init__(
        self,
        *,
        model,
        diffusion,
        data,
        batch_size,
        microbatch,   # 微批次大小，用于分布式训练时的批次划分
        lr,
        ema_rate,  # 指数移动平均的衰减率
        log_interval,  # 日志输出间隔
        save_interval,  # 保存模型间隔
        resume_checkpoint,  # 恢复训练的检查点路径
        use_fp16=False,    # 是否使用半精度浮点数进行训练（FP16）
        fp16_scale_growth=1e-3, # FP16的比例增长
        schedule_sampler=None,  # 扩散模型的时间步采样器
        weight_decay=0.0,   # 权重衰减（L2正则化）
        lr_anneal_steps=0,  # 学习率衰减的步数
        checkpoint_path='',
        gradient_clipping=-1.,  # 梯度裁剪的阈值
        eval_data=None,
        eval_interval=-1,   # 评估的间隔步数
    ):
        print("IN AUG trainutil")
        rank = dist.get_rank()  # 获取当前训练节点的rank（用于分布式训练）
        world_size = dist.get_world_size()  # 获取训练的总节点数
        print("initialing Trainer for",rank,'/',world_size)
        self.rank = rank    # 当前节点的rank
        self.world_size = world_size    # 总节点数
        self.diffusion = diffusion      #respace中的SpacedDiffusion
        self.data = data
        self.eval_data = eval_data
        self.batch_size = batch_size    # 每批次的训练样本数
        self.microbatch = microbatch if microbatch > 0 else batch_size  # 微批次大小，若为负则使用全批次
        self.lr = lr*world_size  # 根据总节点数调整学习率（分布式训练时每个节点处理的样本数不同）
        print("ori lr:",lr,"new lr:",self.lr)   # 输出原始和调整后的学习率
        self.ema_rate = (
            [ema_rate]  # 如果ema_rate是单一值，则直接使用
            if isinstance(ema_rate, float)
            else [float(x) for x in ema_rate.split(",")]    # 如果是逗号分隔的字符串，则拆分成多个ema_rate值
        )
        self.log_interval = log_interval  # 日志输出的间隔
        self.eval_interval = eval_interval  # 评估模型的间隔
        self.save_interval = save_interval  # 模型保存的间隔
        self.resume_checkpoint = resume_checkpoint  # 恢复训练的检查点路径
        self.use_fp16 = use_fp16  # 是否使用FP16进行训练
        self.fp16_scale_growth = fp16_scale_growth  # FP16的比例增长（用于训练稳定性）
        self.schedule_sampler = schedule_sampler or UniformSampler(diffusion)  # 采样器（默认为UniformSampler）
        self.weight_decay = weight_decay  # 权重衰减（L2正则化）
        self.lr_anneal_steps = lr_anneal_steps  # 学习率衰减的步骤
        self.gradient_clipping = gradient_clipping  # 梯度裁剪的阈值

        self.step = 0  # 当前训练步数
        self.resume_step = 0  # 恢复训练的步数
        self.global_batch = self.batch_size * dist.get_world_size()  # 全局批次大小（分布式训练时计算总批次）

        self.lg_loss_scale = INITIAL_LOG_LOSS_SCALE  # 初始化损失尺度（用于FP16训练）
        self.sync_cuda = th.cuda.is_available()  # 检查是否可以使用CUDA
        print('checkpoint_path:{}'.format(checkpoint_path))  # 输出检查点路径
        self.checkpoint_path = checkpoint_path  # 保存检查点的路径
        
        self.model = model.to(rank) # 将模型发送到对应rank的设备（GPU）
       
        # self._load_and_sync_parameters()
        # if self.use_fp16:
        #     self._setup_fp16()

        
        

        if th.cuda.is_available(): # DEBUG **
            self.use_ddp = True # 启用分布式数据并行
            self.ddp_model = DDP(   # 使用DistributedDataParallel包装模型以支持多GPU训练
                self.model,
                device_ids=[self.rank], # 当前节点使用的GPU设备
                find_unused_parameters=False,    # 是否查找未使用的参数（加速训练）
            )
        else:
            assert False    # 如果CUDA不可用，则报错
            # if dist.get_world_size() > 1:
            #     logger.warn(
            #         "Distributed training requires CUDA. "
            #         "Gradients will not be synchronized properly!"
            #     )
            # self.use_ddp = False
            # self.ddp_model = self.model
        self.model_params = list(self.ddp_model.parameters())   # 获取模型的所有参数
        self.master_params = self.model_params  # 将模型参数复制给主参数
        self.opt = AdamW(self.master_params, lr=self.lr, weight_decay=self.weight_decay)
        if self.resume_step:
            # self._load_optimizer_state()
            # # Model was resumed, either due to a restart or a checkpoint
            # # being specified at the command line.
            # self.ema_params = [
            #     self._load_ema_parameters(rate) for rate in self.ema_rate
            # ]
            pass
        else:
            self.ema_params = [
                copy.deepcopy(self.master_params) for _ in range(len(self.ema_rate))  # 为每个EMA衰减率创建备份的模型参数
            ]

    def _load_and_sync_parameters(self):  #负责加载和同步模型的参数（暂时不用）
        # 查找恢复训练的检查点路径。如果指定了resume_checkpoint，则优先使用指定路径；否则尝试自动寻找。
        resume_checkpoint = find_resume_checkpoint() or self.resume_checkpoint

        if resume_checkpoint:
            # 如果找到了恢复训练的检查点，解析出恢复的步数
            self.resume_step = parse_resume_step_from_filename(resume_checkpoint)
            # 如果当前节点的rank是0，则表示是主节点，进行模型加载
            if dist.get_rank() == 0:
                # 输出加载模型的调试信息
                # logger.log(f"loading model from checkpoint: {resume_checkpoint}...")
                print(f"loading model from checkpoint: {resume_checkpoint}...")
                # 从指定的检查点路径加载模型参数，并将其加载到当前的模型实例中
                self.model.load_state_dict(
                    dist_util.load_state_dict(
                        resume_checkpoint, map_location=dist_util.dev()
                    )
                )
        # 同步所有分布式训练节点的模型参数，以确保所有节点的模型保持一致
        dist_util.sync_params(self.model.parameters())

    def _load_ema_parameters(self, rate):   #用于加载模型的指数滑动平均（EMA）参数，并在分布式训练中同步这些参数。（暂时不用）
        # 通过深拷贝的方式将 master_params（模型的主参数）复制一份作为 ema_params
        ema_params = copy.deepcopy(self.master_params)
        # 查找恢复训练的主检查点路径。如果没有指定检查点路径，则使用默认的resume_checkpoint。
        main_checkpoint = find_resume_checkpoint() or self.resume_checkpoint
        # 根据主检查点和恢复的训练步数找到相应的EMA检查点路径
        ema_checkpoint = find_ema_checkpoint(main_checkpoint, self.resume_step, rate)
        if ema_checkpoint:
            # 如果找到了EMA检查点路径，加载EMA参数
            # 如果当前节点的rank是0，则表示是主节点，进行EMA参数加载
            if dist.get_rank() == 0:
                # 输出加载EMA参数的调试信息
                logger.log(f"loading EMA from checkpoint: {ema_checkpoint}...")
                # 从EMA检查点文件中加载状态字典（即模型的参数）
                state_dict = dist_util.load_state_dict(
                    ema_checkpoint, map_location=dist_util.dev()
                )
                # 将加载的状态字典转换成主参数格式，存储到ema_params
                ema_params = self._state_dict_to_master_params(state_dict)
        # 同步所有分布式训练节点的EMA参数，以确保所有节点的EMA参数保持一致
        dist_util.sync_params(ema_params)
        return ema_params

    def _load_optimizer_state(self):  #用于从检查点加载优化器的状态（暂时不用）
        main_checkpoint = find_resume_checkpoint() or self.resume_checkpoint
        opt_checkpoint = bf.join(
            bf.dirname(main_checkpoint), f"opt{self.resume_step:06}.pt"
        )
        if bf.exists(opt_checkpoint):
            logger.log(f"loading optimizer state from checkpoint: {opt_checkpoint}")
            state_dict = dist_util.load_state_dict(
                opt_checkpoint, map_location=dist_util.dev()
            )
            self.opt.load_state_dict(state_dict)

    def _setup_fp16(self):#（暂时不用）
        self.master_params = make_master_params(self.model_params)
        self.model.convert_to_fp16()

    def run_loop(self):
        print('START LOOP FLAG')
        # 训练循环的条件：
        # 1. 如果没有学习率退火（`self.lr_anneal_steps` 为0），则一直执行训练；
        # 2. 否则，训练步数达到指定的学习率退火步数（`self.lr_anneal_steps`）时结束训练。
        while (
            not self.lr_anneal_steps    # 如果学习率退火步数为0，继续训练
            or self.step + self.resume_step < self.lr_anneal_steps//self.world_size # 退火步骤还未达到
        ):
            DATA = self.data
            # print("train_util的DATA:", DATA)
            batch = next(self.data) # 获取下一批次的数据
            cond = None   # 条件变量，可能用于控制训练中的某些逻辑（但在此处未使用）
            # if self.step<3:
            #     print("RANK:",self.rank,"STEP:",self.step,"BATCH:",batch)
            self.run_step(batch, cond)
            if self.step % self.log_interval == 0:    # 每隔 `log_interval` 步，输出日志
                # dist.barrier()
                pass
                # print('loggggg')
                #logger.dumpkvs()
            # 如果指定了评估数据集，并且当前步数可以进行评估
            if self.eval_data is not None and self.step % self.eval_interval == 0:
                # batch_eval, cond_eval = next(self.eval_data)
                # self.forward_only(batch, cond)
                print('eval on validation set')
                pass# logger.dumpkvs()
            if self.step % self.save_interval == 0 and self.step!=0:    # 每隔 `save_interval` 步保存一次模型检查点
                self.save()
                # Run for a finite amount of time in integration tests.
                # 如果设置了有限的训练步骤用于集成测试（`DIFFUSION_TRAINING_TEST` 环境变量），
                # 在训练进行到一定步骤后结束训练。
                if os.environ.get("DIFFUSION_TRAINING_TEST", "") and self.step > 0:
                    return
            self.step += 1  # 增加训练步数
        # Save the last checkpoint if it wasn't already saved.
        if (self.step - 1) % self.save_interval != 0:
            self.save()

    def run_step(self, batch, cond):
        self.forward_backward(batch, cond)  # 执行前向传播和反向传播
        if self.use_fp16:    # 如果使用混合精度训练（FP16），执行FP16优化
            self.optimize_fp16()
        else:                # 否则执行常规优化
            self.optimize_normal()
        self.log_step()     # 记录当前训练步骤的日志

    def forward_only(self, batch, cond):  #(暂时不用)执行一次前向传播操作，并计算损失，但不进行梯度更新（即不执行反向传播）。这个方法通常用于模型评估或验证阶段，其中我们不需要更新模型参数，只需要计算损失来监控模型性能。
        with th.no_grad():
            zero_grad(self.model_params)
            for i in range(0, batch.shape[0], self.microbatch):
                micro = batch[i: i + self.microbatch].to(dist_util.dev())
                micro_cond = {
                    k: v[i: i + self.microbatch].to(dist_util.dev())
                    for k, v in cond.items()
                }
                last_batch = (i + self.microbatch) >= batch.shape[0]
                t, weights = self.schedule_sampler.sample(micro.shape[0], dist_util.dev())
                # print(micro_cond.keys())
                compute_losses = functools.partial(
                    self.diffusion.training_losses,
                    self.ddp_model,
                    micro,
                    t,
                    model_kwargs=micro_cond,
                )

                if last_batch or not self.use_ddp:
                    losses = compute_losses()
                else:
                    with self.ddp_model.no_sync():
                        losses = compute_losses()

                log_loss_dict(
                    self.diffusion, t, {f"eval_{k}": v * weights for k, v in losses.items()}
                )


    def forward_backward(self, batch, cond):
        # 清除优化器中的累积梯度
        # zero_grad(self.model_params)
        self.opt.zero_grad()    # 使用 PyTorch 的优化器自带的梯度清零方法
        for i in range(0, batch[0].shape[0], self.microbatch):  # 遍历整个批次的数据，并按微批次大小（self.microbatch）进行切分。
            # 数据部分 1（通常是输入数据）。# 数据部分 2（可能是目标标签）。 # 数据部分 3（如掩码或额外信息）。# 数据部分 4（可能是辅助输入）。  加了基因数据
            micro = (batch[0].to(self.rank), batch[1].to(self.rank), batch[2].to(self.rank), batch[3].to(self.rank), batch[4].to(self.rank))
            # print("mocro:", micro)
            # micro = (batch[0].to(self.rank), batch[1].to(self.rank), batch[2].to(self.rank), batch[3].to(self.rank), batch[4].to(self.rank))
            last_batch = True   # 标记是否为最后一个微批次。这里直接设为 True，简化处理逻辑。
            # 使用调度采样器生成时间步 t 和对应的权重 weights。t 表示当前微批次数据的采样时间步，weights 是与 t 对应的权重。
            t, weights = self.schedule_sampler.sample(micro[0].shape[0], self.rank)

            # 创建一个部分函数 compute_losses，用于延迟计算当前微批次的损失值。
            compute_losses = functools.partial(
                self.diffusion.training_losses,     # 扩散模型的损失函数。
                self.ddp_model,               # 当前的分布式数据并行模型。
                micro,                              # 当前微批次数据。
                t,                                  # 当前时间步。
                model_kwargs=None,                  # 没有额外的模型参数。
            )
            # 如果是最后一个微批次，或者没有使用分布式数据并行，直接计算损失。
            if last_batch or not self.use_ddp:
                losses = compute_losses()
            else:
                # 对于非最后一个批次，在分布式数据并行模式下禁止梯度同步，# 节省计算资源，直到最后一个批次再进行同步。
                with self.ddp_model.no_sync():
                    losses = compute_losses()
            # 如果使用基于损失感知的采样器（LossAwareSampler），则将当前时间步 t 和对应的损失值传递给采样器，更新采样权重。
            if isinstance(self.schedule_sampler, LossAwareSampler):
                self.schedule_sampler.update_with_local_losses(
                    t, losses["loss"].detach()
                )
            # 计算当前微批次的加权损失。 losses["loss"] 包含损失值，weights 是与时间步 t 对应的采样权重。
            loss = (losses["loss"] * weights).mean()
            # print('----DEBUG-----',self.step,self.log_interval)
            if self.step % self.log_interval == 0 and self.rank==0:
            # if self.step % 2000 == 0 and self.rank == 0:
                print("rank0: ",self.step,loss.item())      # 打印训练步骤和损失值。
                wandb.log({'loss':loss.item()})     # 记录损失值到 WandB。
            # log_loss_dict(
            #     self.diffusion, t, {k: v * weights for k, v in losses.items()}
            # )
            if self.use_fp16:       # 如果使用半精度训练（fp16），需要进行损失值缩放，以避免精度问题。
                # loss_scale = 2 ** self.lg_loss_scale
                # (loss * loss_scale).backward()
                pass
            else:
                loss.backward()

    def optimize_fp16(self):    #此方法用于优化半精度浮点数（FP16）训练模型的参数，并处理可能的梯度溢出问题。
        # 检查模型参数的梯度中是否存在非有限值（如 NaN 或 Inf）
        if any(not th.isfinite(p.grad).all() for p in self.model_params):
            # 如果发现梯度异常，减少损失缩放因子以降低数值溢出的风险
            self.lg_loss_scale -= 1
            # 记录日志，提示发现 NaN，并显示新的损失缩放因子
            logger.log(f"Found NaN, decreased lg_loss_scale to {self.lg_loss_scale}")
            return  # 提前退出，避免使用异常的梯度进行更新
        # 将模型参数的梯度（FP16）转换为主参数（FP32）的梯度
        model_grads_to_master_grads(self.model_params, self.master_params)
        # 对主参数的梯度进行缩放，调整因子为当前损失缩放的倒数
        self.master_params[0].grad.mul_(1.0 / (2 ** self.lg_loss_scale))
        self._log_grad_norm()    # 记录梯度范数，用于监控训练过程中的数值稳定性
        self._anneal_lr()   # 调用学习率退火方法，动态调整当前的学习率
        self.opt.step()      # 执行优化器的 step 操作，使用缩放后的梯度更新主参数（FP32）
        for rate, params in zip(self.ema_rate, self.ema_params):    # 遍历所有的 EMA 速率和对应的参数集合
            update_ema(params, self.master_params, rate=rate)    # 使用当前主参数更新 EMA 参数，速率由 rate 决定
        master_params_to_model_params(self.model_params, self.master_params)   # 将主参数（FP32）同步回模型参数（FP16），以确保两者保持一致
        self.lg_loss_scale += self.fp16_scale_growth    # 增加损失缩放因子，以支持更大的梯度范围（动态损失缩放的一部分）

    def grad_clip(self):
        # print('doing gradient clipping')
        max_grad_norm=self.gradient_clipping #通常设置为3.0  # 定义最大梯度范数，用于裁剪梯度的阈值
        if hasattr(self.opt, "clip_grad_norm"):
            # 如果优化器具有内置的梯度裁剪方法（例如 Sharded Optimizer）
            # 调用优化器的裁剪方法并应用最大梯度范数
            self.opt.clip_grad_norm(max_grad_norm)
        #如果优化器不支持 clip_grad_norm，可以检查模型是否支持裁剪方法
        # else:
        #     assert False
        # elif hasattr(self.model, "clip_grad_norm_"):
        #     # Some models (like FullyShardedDDP) have a specific way to do gradient clipping
        #     self.model.clip_grad_norm_(args.max_grad_norm)
        else:
            # 如果既没有优化器的内置方法，也没有模型的方法，使用 PyTorch 提供的默认梯度裁剪工具
            th.nn.utils.clip_grad_norm_(
                self.model.parameters(), #amp.master_params(self.opt) if self.use_apex else # 裁剪模型参数的梯度
                max_grad_norm,  # 使用预定义的最大梯度范数
            )

    def optimize_normal(self):
        # 如果定义了梯度裁剪阈值（大于0），调用梯度裁剪函数来裁剪梯度
        if self.gradient_clipping > 0:
            self.grad_clip()
        # self._log_grad_norm()
        # 调整学习率（如果有设定学习率衰减）
        self._anneal_lr()
        # 执行优化器的步进操作，更新模型参数
        self.opt.step()
        # 遍历每个 EMA（指数移动平均）率，更新对应的参数
        for rate, params in zip(self.ema_rate, self.ema_params):
            update_ema(params, self.master_params, rate=rate)   # 使用当前的主参数来更新 EMA 参数，应用不同的 EMA 更新率

    def _log_grad_norm(self):
        sqsum = 0.0  # 初始化变量 sqsum 为 0，用于累积所有模型参数的梯度平方和
        for p in self.master_params:    # 遍历所有的主参数 (master parameters)
            sqsum += (p.grad ** 2).sum().item()  # 对每个参数的梯度进行平方操作并求和，然后加到 sqsum 中，p.grad 是参数 p 的梯度，p.grad ** 2 是每个元素的平方，.sum() 是所有元素的总和
        # logger.logkv_mean("grad_norm", np.sqrt(sqsum))

    def _anneal_lr(self):
        if not self.lr_anneal_steps:    # 如果没有进行学习率退火的步骤，直接返回，不做任何操作
            return
        frac_done = (self.step + self.resume_step) / self.lr_anneal_steps    # 计算已经完成的比例：当前步数 + 恢复步数，除以学习率退火的总步数
        lr = self.lr * (1 - frac_done)  # 根据完成的比例调整学习率，初始学习率 lr 乘以 (1 - 完成的比例)，使学习率逐步减小
        for param_group in self.opt.param_groups:   # 遍历优化器中的所有参数组，将每个参数组的学习率设置为计算得到的值
            param_group["lr"] = lr

    def log_step(self):
        logger.logkv("step", self.step + self.resume_step)  # 记录当前的训练步骤，将当前步骤号（加上恢复的步骤数）记录到日志中
        logger.logkv("samples", (self.step + self.resume_step + 1) * self.global_batch) # 记录当前已处理的样本数量，(当前步骤 + 恢复步骤 + 1) 乘以全局批量大小
        if self.use_fp16:   # 如果使用了半精度浮点数（FP16），记录当前的 loss scale（损失缩放）值
            logger.logkv("lg_loss_scale", self.lg_loss_scale)

    def save(self):
        def save_checkpoint(rate, params):  # 内部函数，负责保存模型或 EMA 参数的检查点
            state_dict = self._master_params_to_state_dict(params)  # 将当前模型参数转换为可存储的状态字典
            if dist.get_rank() == 0:     # 检查当前是否是主进程（仅主进程负责保存）
                # logger.log(f"saving model {rate}...")
                print(f"saving model {rate}...")
                if not rate:     # 如果没有 EMA 速率（即原始模型），生成普通模型的文件名
                    filename = f"PLAIN_model{((self.step+self.resume_step)*self.world_size):06d}.pt"
                else:   # 如果有 EMA 速率，生成对应 EMA 参数的文件名
                    filename = f"PLAIN_ema_{rate}_{((self.step+self.resume_step)*self.world_size):06d}.pt"
                # print('writing to', bf.join(get_blob_logdir(), filename))
                # print('writing to', bf.join(self.checkpoint_path, filename))
                # with bf.BlobFile(bf.join(get_blob_logdir(), filename), "wb") as f:
                #     th.save(state_dict, f)
                with bf.BlobFile(bf.join(self.checkpoint_path, filename), "wb") as f: # DEBUG **# 使用指定的文件路径保存状态字典
                    th.save(state_dict, f)

        save_checkpoint(0, self.master_params)  # 保存普通模型的参数
        for rate, params in zip(self.ema_rate, self.ema_params):    # 遍历每个 EMA 速率，保存对应的 EMA 参数
            save_checkpoint(rate, params)

        # if dist.get_rank() == 0: # DEBUG **
        #     with bf.BlobFile(
        #         bf.join(get_blob_logdir(), f"opt{(self.step+self.resume_step):06d}.pt"),
        #         "wb",
        #     ) as f:
        #         th.save(self.opt.state_dict(), f)
        # 使用分布式屏障，确保所有进程在保存完成后同步
        dist.barrier()

    def _master_params_to_state_dict(self, master_params):
        """
        将主模型参数（master parameters）转换为模型的状态字典（state dict）。
        主要用于保存模型的当前参数。

        :param master_params: 主模型参数列表，可能是浮点16（fp16）或浮点32（fp32）格式。
        :return: 转换后的模型状态字典。
        """
        if self.use_fp16:   # 如果使用了 FP16，则需要将主模型参数解压缩（unflatten）到与模型参数匹配的形状。
            master_params = unflatten_master_params(
                list(self.model.parameters()), master_params # DEBUG **  # list（）模型当前参数列表。# master_params主模型参数列表。
            )
        state_dict = self.model.state_dict()    # 获取模型的状态字典（包括所有参数名及其值）。
        for i, (name, _value) in enumerate(self.model.named_parameters()):  # 遍历模型参数，同时将主模型参数更新到状态字典中。
            assert name in state_dict   # 确保参数名在状态字典中。
            state_dict[name] = master_params[i] # 将主模型参数赋值给对应的状态字典项。
        return state_dict

    def _state_dict_to_master_params(self, state_dict):
        """
        将模型的状态字典（state_dict）转换为主模型参数（master parameters）。

        主模型参数是训练过程中使用的高精度版本（如 FP32），
        尤其在使用混合精度训练时，用于提升训练的稳定性。

        :param state_dict: 模型的状态字典，包含参数名和对应的张量值。
        :return: 主模型参数列表。
        """
        # 从状态字典中提取所有参数值，按照模型中定义的参数顺序。
        params = [state_dict[name] for name, _ in self.model.named_parameters()]    # `state_dict[name]` 提取参数值，与 `model.named_parameters()` 的顺序一致。
        if self.use_fp16:   # 将 FP16 参数转换为 FP32 主参数。
            return make_master_params(params)
        else:
            return params


def parse_resume_step_from_filename(filename):
    """此函数可以有效解析文件名中包含的步数，为模型训练的断点恢复提供支持。
    从文件名中解析出模型的恢复步数（step）。
    文件名格式通常为 `path/to/modelNNNNNN.pt`，其中 `NNNNNN` 是保存检查点时的步数。
    如果无法从文件名中解析出步数，则返回 0。
    :param filename: 字符串，包含模型检查点的文件路径。
    :return: 整数，文件名中提取的恢复步数。
    """
    split = filename.split("model") # 以 "model" 为分隔符，将文件名拆分成多个部分
    if len(split) < 2:  # 如果分割后部分少于 2 个，说明文件名中没有包含步数，返回 0
        return 0
    split1 = split[-1].split(".")[0]    # 获取最后一部分，去掉文件扩展名，可能的格式是 "NNNNNN.pt"
    try:
        return int(split1)  # 尝试将这一部分转换为整数，作为恢复的步数
    except ValueError:
        return 0    # 如果转换失败（比如文件名中没有有效数字），返回 0


def get_blob_logdir():  #（好像暂时没用到）
    """
    获取模型训练过程中用于存储检查点（checkpoint）的日志目录。
    首先尝试从环境变量 `DIFFUSION_BLOB_LOGDIR` 中获取日志目录。
    如果未设置该环境变量，则默认使用 `logger.get_dir()` 返回的日志目录。
    :return: 字符串，表示日志目录路径。
    """
    return os.environ.get("DIFFUSION_BLOB_LOGDIR", logger.get_dir())


def find_resume_checkpoint():#（暂时没用到）
    # On your infrastructure, you may want to override this to automatically
    # discover the latest checkpoint on your blob storage, etc.
    return None


def find_ema_checkpoint(main_checkpoint, step, rate):#（暂时好像没用）
    """
    查找 EMA（Exponential Moving Average）模型检查点文件。
    :param main_checkpoint: 主检查点文件的路径。如果为 None，则不查找。
    :param step: 整数，训练步骤数。
    :param rate: EMA 的衰减率（如 0.999）。
    :return: 如果找到相应的 EMA 检查点，则返回文件路径；否则返回 None。
    """
    if main_checkpoint is None:
        return None
    filename = f"ema_{rate}_{(step):06d}.pt"
    path = bf.join(bf.dirname(main_checkpoint), filename)
    if bf.exists(path):
        return path
    return None


def log_loss_dict(diffusion, ts, losses):#（train阶段代码中函数体未执行任何操作。）
    """
    记录损失字典中的统计信息。
    :param diffusion: 一个扩散模型对象，包含训练的相关信息。
    :param ts: 一个张量，包含当前批次的时间步索引。
    :param losses: 一个字典，键为损失名称，值为对应的损失张量。
    """
    return  # 函数体目前为空，直接返回
    # 遍历损失字典中的每个损失名称和对应的值
    for key, values in losses.items():
        # 对损失的平均值进行记录
        logger.logkv_mean(key, values.mean().item())
        # Log the quantiles (four quartiles, in particular).# 遍历每个时间步索引和对应的损失值
        for sub_t, sub_loss in zip(ts.cpu().numpy(), values.detach().cpu().numpy()):
            # 计算当前时间步属于哪个四分位区间
            quartile = int(4 * sub_t / diffusion.num_timesteps)
            # 按四分位区间记录损失值
            logger.logkv_mean(f"{key}_q{quartile}", sub_loss)

