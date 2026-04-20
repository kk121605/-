from transformers import AutoConfig
from transformers.models.bert.modeling_bert import BertEncoder
import torch
import numpy as np
import torch as th
import torch.nn as nn
from .GeneVAE import GeneVAE
import torch.nn.functional as F
import pandas as pd
from torch.utils.data import DataLoader, Dataset

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
from .nn import (
    SiLU,
    conv_nd,
    linear,
    avg_pool_nd,
    zero_module,
    timestep_embedding,
    checkpoint,
)

print('checkpoint 0810 in model.py')
class TransformerNetModel2(nn.Module):
    def __init__(
        self,
        in_channels,    # 输入通道数
        model_channels, # 模型内部的通道数
        dropout=0.1,    # dropout率
        num_classes=None,   # 类别数，默认为None
        use_checkpoint=False,   # 是否使用梯度检查点
        config=None,    # 模型配置
        config_name='/DoctoralStudents/yuanhang/AI_drug/my_model/Diffusion/bert-base-uncased',
        training_mode='emb', # e2e
        vocab_size=None, #821   # 词汇表大小
        experiment_mode='lm', #lm   # 实验模式（lm表示语言建模任务）
        init_pretrained=False,  # 是否初始化预训练模型
        logits_mode=1,      # logits计算模式（1代表标准）
        num_heads=8,    # 注意力头的数量
        hidden_size=768,    # 隐藏层的大小
        num_attention_heads = 12,   # 注意力头数
        num_hidden_layers=12,   # 隐藏层的层数
        mask = False,    # 是否使用mask
        gene_num=None  # **新增参数**
    ):
        super().__init__()

        if gene_num is None:
            raise ValueError("❌ 需要提供 `gene_num`，否则 VAE 无法正确初始化！")

        config = AutoConfig.from_pretrained('/nfs/home/yuanhang/AI_drug/bert-base-uncased')
        config.is_decoder=True      # 配置为解码器模式
        config.add_cross_attention=True     # 配置支持交叉注意力
        config.hidden_dropout_prob = 0.1    # 设置隐藏层的dropout概率
        config.hidden_size = hidden_size    # 设置隐藏层大小
        config.num_attention_heads = num_attention_heads    # 设置注意力头的数量
        config.num_hidden_layers = num_hidden_layers    # 设置隐藏层的层数
            # config.hidden_size = 512

        # 加载VAE模型
        # VAE潜在空间处理
        self.vae_latent_proj = nn.Linear(1042, config.hidden_size)  # 将 mu 和 logvar 投影到 hidden_size
        self.vae_model = GeneVAE(
            input_size=gene_num,  # gene_num
            hidden_sizes=[512, 256, 128],  # gene_hidden_sizes
            latent_size=64,  # gene_latent_size
            output_size=gene_num,  # gene_num
            activation_fn=nn.ReLU(),
            dropout=0.2  #
        )

        vae_state_dict = torch.load('/nfs/home/yuanhang/AI_drug/DATA/results/saved_gene_vae_train_13755_gen3.pkl')  # 加载权重
        # vae_state_dict = torch.load('/DoctoralStudents/yuanhang/AI_drug/GxVAEs-main/results/saved_gene_vae_train.pkl')  # 加载权重
        self.vae_model.load_state_dict(vae_state_dict)  # 加载到模型实例中
        self.vae_model.eval()  # 设置为评估模式

        # 一些模型超参数
        self.mask = mask    # 是否使用mask
        self.num_heads = num_heads  # 注意力头数
        self.in_channels = in_channels # 16   # 输入通道数
        self.model_channels = model_channels # 128  # 模型内部通道数
        self.dropout =dropout     # dropout率
        self.num_classes = None # None  # 类别数（默认没有）
        self.use_checkpoint = False # False # 是否使用检查点
        self.num_heads_upsample = 4 # 上采样时使用的注意力头数
        self.logits_mode = 1    # logits计算模式
        # self.deep_channels = deep_channels
        self.word_embedding = nn.Embedding(vocab_size, self.in_channels)    # 词嵌入层，将输入的词汇索引映射到嵌入空间

        # 线性变换用于词汇模型的输出
        self.lm_head = nn.Linear(self.in_channels, vocab_size)   # 词汇表大小的输出层
        self.lm_head.weight = self.word_embedding.weight    # 共享词嵌入和输出层权重

        self.conditional_gen = False

        # 线性变换层用于描述向量的处理（如可能是分子描述符等信息）
        self.desc_down_proj = nn.Sequential(
            linear(768,config.hidden_size), # 从768到hidden_size的线性变换
            SiLU(), # SiLU激活函数
            linear(config.hidden_size, config.hidden_size), # 隐藏层到隐藏层的线性变换
        )
        # 时间嵌入，用于编码时间步（模型的输入部分）
        time_embed_dim = model_channels * 4 # 512   # 计算时间嵌入的维度
        self.time_embed = nn.Sequential(
            linear(model_channels, time_embed_dim), # 从模型通道到时间嵌入维度的线性变换
            SiLU(), # SiLU激活函数
            linear(time_embed_dim, config.hidden_size),    # 时间嵌入到隐藏层维度的线性变换
        )

        # 输入上采样投影层
        self.input_up_proj = nn.Sequential(
            nn.Linear(in_channels, config.hidden_size), # 输入通道到隐藏层的线性变换
            nn.Tanh(),  # Tanh激活函数
            nn.Linear(config.hidden_size, config.hidden_size))  # 隐藏层到隐藏层的线性变换
        # BERT编码器，使用配置文件构建
        self.input_transformers = BertEncoder(config)
        # 注册位置ID（位置编码）
        self.register_buffer("position_ids", torch.arange(config.max_position_embeddings).expand((1, -1)))
        self.position_embeddings = nn.Embedding(config.max_position_embeddings, config.hidden_size)
        # self.token_type_embeddings = nn.Embedding(config.type_vocab_size, config.hidden_size)
        # LayerNorm用于规范化层输出
        self.LayerNorm = nn.LayerNorm(config.hidden_size, eps=config.layer_norm_eps)
        self.dropout = nn.Dropout(config.hidden_dropout_prob)   # dropout层
        # 输出下投影层，将隐藏层映射回输入空间
        # self.output_down_proj = nn.Sequential(nn.Linear(config.hidden_size,
        #                                                 config.hidden_size),# 隐藏层到隐藏层的线性变换
        #                                       nn.Tanh(), nn.Linear(config.hidden_size, in_channels))# 隐藏层到输入通道的线性变换
        self.output_down_proj = nn.Sequential(nn.Linear(config.hidden_size*2,
                                                        config.hidden_size),
                                              nn.Tanh(), nn.Linear(config.hidden_size, in_channels))
    # 获取词嵌入表示
    def get_embeds(self, input_ids):
        return self.word_embedding(input_ids)        #通过模型的词嵌入层 self.word_embedding（一个 nn.Embedding 层）将输入的 input_ids 转换为对应的嵌入表示。

    def get_embeds_with_deep(self, input_ids):
        atom , deep = input_ids                    #与 get_embeds 类似，input_ids 是一个包含词汇索引的张量，但这里假设 input_ids 是一个元组或列表，其中包含两个部分：atom 和 deep。
        # th.tensor([0]).to('cuda')
        # print(atom,deep)
        # print(deep[0])
        atom = self.word_embedding(atom)          #atom是主输入的单词索引，self.word_embedding(atom)：对 atom 部分进行嵌入，即将原子部分转换为嵌入表示
        # th.tensor([0]).to('cuda')
        deep = self.deep_embedding(deep)          #deep是额外的深度输入索引，可能是用于描述其他信息（如化学结构、分子信息等），self.deep_embedding(deep)：对 deep 部分进行嵌入，这可能是与原子部分不同的嵌入层。
        # th.tensor([0]).to('cuda')
        return torch.concat([atom,deep],dim=-1)

    def get_logits_deep(self,hidden_repr):       #hidden_repr：这是输入的隐藏表示，通常是模型的最后一层隐藏状态。
        return self.deep_head(hidden_repr)       #self.deep_head(hidden_repr)：通过 self.deep_head（一个 nn.Linear 层）计算并返回 logits。这个 deep_head 可能是一个特定的分类层，用于预测某些任务（例如，深度任务的分类）。


#get_logits：更复杂的方法，根据 logits_mode 的不同进行不同类型的计算。####
    def get_logits(self, hidden_repr):
        if self.logits_mode == 1:                #模式 1：直接从隐藏表示计算输出 logits，通常用于标准的语言模型任务或分类任务。
            return self.lm_head(hidden_repr)
        elif self.logits_mode == 2:              #模式 2：计算余弦距离（或者可以认为是度量学习），这种方式可能适用于需要衡量文本相似度、生成任务或其它需要距离度量的任务。
            text_emb = hidden_repr
            #emb_norm和arr_norm：计算嵌入向量和 lm_head.weight 权重矩阵的 L2 范数，用于计算余弦相似度。emb_norm 是 lm_head 权重矩阵的平方和，arr_norm 是输入 hidden_repr 的平方和。
            emb_norm = (self.lm_head.weight ** 2).sum(-1).view(-1, 1)  # vocab
            text_emb_t = th.transpose(text_emb.view(-1, text_emb.size(-1)), 0, 1)  # d, bsz*seqlen
            arr_norm = (text_emb ** 2).sum(-1).view(-1, 1)  # bsz*seqlen, 1

            #dist计算每个单词的余弦距离。余弦距离是通过归一化的点积计算的，这里使用 emb_norm、arr_norm 和 lm_head.weight 的点积来计算每个词的距离。
            dist = emb_norm + arr_norm.transpose(0, 1) - 2.0 * th.mm(self.lm_head.weight,
                                                                     text_emb_t)  # (vocab, d) x (d, bsz*seqlen)
            #scores：通过平方根对距离进行修正，然后通过 .view 操作调整输出的形状，使其符合模型的需求。最终返回的是每个单词和每个序列位置的得分。
            scores = th.sqrt(th.clamp(dist, 0.0, np.inf)).view(emb_norm.size(0), hidden_repr.size(0),
                                                               hidden_repr.size(1)) # vocab, bsz*seqlen
            scores = -scores.permute(1, 2, 0).contiguous()  # 转置并返回最终的得分

            return scores
        else:
            raise NotImplementedError   # 如果 logits_mode 不在已知范围内，抛出异常
    #添加余弦相似度计算函数###
    def cosine_similarity_loss(self, hidden_repr, latent_rep):
        """
        计算余弦相似度损失，首先对输入表示进行归一化处理。
        """
        # 对输入表示进行归一化
        hidden_repr_normalized = F.normalize(hidden_repr, p=2, dim=-1)  # L2 归一化
        latent_rep_normalized = F.normalize(latent_rep, p=2, dim=-1)  # L2 归一化

        # 计算余弦相似度
        cos_sim = F.cosine_similarity(hidden_repr_normalized, latent_rep_normalized, dim=-1)  # dim=-1 是默认的最后一个维度

        # 将相似度转化为损失，1 - cos_sim 表示相似度越高，损失越低
        loss = 1 - cos_sim
        return loss.mean()  # 平均损失

    def forward(self, x, timesteps, desc_state, desc_mask , gene_expression, y=None):
        """
        Apply the model to an input batch.

        x:输入的张量，形状通常为 [N x C x ...]，代表一个批次的数据，N 是批次大小，C 是通道数，后面的维度可以是时间步或序列长度等。
        timesteps: 一个形状为 [N] 的一维张量，表示每个输入样本的时间步。这个用于生成时间嵌入。
        desc_state: 描述状态张量，通常用于表示与输入 x 相关的额外信息。形状可能是 [N x D]，D 代表某种特征维度。
        desc_mask: 描述状态的掩码张量，形状为 [N x L]，L 是序列的长度，用来指示哪些位置需要被关注，通常是注意力掩码。
        y: 如果模型是条件的（例如分类任务），则这是目标标签张量，形状为 [N]。如果模型不是条件的，则 y 应该是 None。
        src_ids 和 src_mask: 源输入的ID和掩码，可能用于序列任务。
        """
        # print(f'real model inputs: {timesteps}')
        assert (y is not None) == (
            self.num_classes is not None
        ), "must specify y if and only if the model is class-conditional"
        # 根据传入的 timesteps 生成时间嵌入 emb，这个嵌入通常用于处理时间序列或时序数据，并在模型中加入时间信息。[]
        emb = self.time_embed(timestep_embedding(timesteps, self.model_channels))

        ################################################################
        #如果 self.mask 为 True，会根据 timesteps 的值对 desc_state 和 desc_mask 进行条件修改。具体来说，当时间步小于200时，desc_state 被设置为0，desc_mask 被设置为1。
        if self.mask:
            desc_state = torch.where(timesteps.reshape(-1,1,1)<200,0.,desc_state)
            assert(len(desc_mask.shape)==2)
            desc_mask = torch.where(timesteps.reshape(-1,1)<200,1.,desc_mask)

        #################################################################
        # print("x的尺寸：", x.shape)
        # print(x)
        # print("gen的尺寸：", gene_expression.shape)
        # print(gene_expression)
        #emb_x：输入 x 会通过 input_up_proj（一个包含 Linear 层的 nn.Sequential 模块）进行投影。
        emb_x = self.input_up_proj(x)
        seq_length = x.size(1)
        #根据 x 的序列长度，截取 position_ids，这些位置ID表示每个时间步或单词的位置。
        position_ids = self.position_ids[:, : seq_length ]
        # print(emb_x.shape, emb.shape, self.position_embeddings)
        #将位置嵌入（位置编码）、投影后的输入 emb_x 以及时间嵌入 emb 相加，得到最终的输入嵌入。然后对嵌入应用 LayerNorm 和 Dropout。
        emb_inputs = self.position_embeddings(position_ids) + emb_x + emb.unsqueeze(1).expand(-1, seq_length, -1)
        emb_inputs = self.dropout(self.LayerNorm(emb_inputs))

        #对 desc_state 进行线性变换（desc_down_proj），然后应用 LayerNorm 和 Dropout。
        num_heads = 16
        desc_state = self.dropout(self.LayerNorm(self.desc_down_proj(desc_state)))

        # 确保 desc_mask 的形状符合模型的要求
        desc_mask = desc_mask.unsqueeze(1).unsqueeze(2)  # [batch_size, 1, seq_len, 1]
        desc_mask = desc_mask.expand(-1, num_heads, -1, -1)  # [batch_size, num_heads, seq_len, seq_len]

        # print(f"emb_inputs shape: {emb_inputs.shape}")
        # print(f"desc_state shape: {desc_state.shape}")
        # print(f"desc_mask shape (before model): {desc_mask.shape}")
        # print(f"desc_mask unique values: {desc_mask.unique()}")
        #交叉注意力机制在这里
        # #输入嵌入 emb_inputs 被传递给 input_transformers（一个 BERT 编码器）。编码器的 encoder_hidden_states 是 desc_state，即在时序/状态上条件化的隐藏表示。encoder_attention_mask 是 desc_mask，用于指示哪些位置应该被关注。
        #输入的 emb_inputs 作为查询（Query），而 desc_state 作为键（Key）和值（Value）
        # input_trans_hidden_states = self.input_transformers(emb_inputs,encoder_hidden_states=desc_state,encoder_attention_mask=desc_mask).last_hidden_state
        output = self.input_transformers(emb_inputs, encoder_hidden_states=desc_state, encoder_attention_mask=desc_mask)
        # print(output)
        input_trans_hidden_states = output.last_hidden_state
        # print(input_trans_hidden_states.shape)

        # 加载基因数据
        mu, logvar = self.vae_model(gene_expression)
        # print("mu_size:", mu.shape)
        # print("logvar_size:", logvar.shape)

            # VAE潜在空间映射
        latent_rep = torch.cat([mu, logvar], dim=-1)  # 拼接 mu 和 logvar
        # print("拼接后latent_rep:", latent_rep.shape)
        latent_rep = self.vae_latent_proj(latent_rep)  # 映射到与 hidden_size 相同的维度
        # print("线性变换后尺寸：", latent_rep.shape)

        expanded_gene_expression = latent_rep.unsqueeze(1).expand(-1, seq_length, -1)

        # 输出形状检查
        # print("input_trans_hidden_states shape:", input_trans_hidden_states.shape)
        # print("expanded_gene_expression shape:", expanded_gene_expression.shape)

        # 将潜在表示与输入的隐藏状态结合
        adjusted_output = torch.cat((input_trans_hidden_states, expanded_gene_expression), dim=-1)
        # print(adjusted_output.shape)

        # 计算 cosine similarity loss
        cos_sim_loss = self.cosine_similarity_loss(input_trans_hidden_states, expanded_gene_expression)
        # 进一步的投影操作
        h = self.output_down_proj(adjusted_output)
        h = h.type(x.dtype)

        return h, cos_sim_loss


    # def get_feature_vectors(self, x, timesteps, y=None):
    #     """
    #     Apply the model and return all of the intermediate tensors.
    #
    #     :param x:  输入的张量，形状为 [N x C x ...]，N 是批次大小，C 是通道数，后面是数据的其他维度。
    #     :param timesteps: 一维张量，形状为 [N]，表示每个样本的时间步。通常用于生成时间嵌入。
    #     :param y: 如果模型是条件模型（如分类任务），则这是一个一维张量，表示每个样本的标签，形状为 [N]。如果没有条件标签，y 应为 None。
    #     :return: a dict with the following keys:
    #              - 'down': a list of hidden state tensors from downsampling.
    #              - 'middle': the tensor of the output of the lowest-resolution
    #                          block in the model.
    #              - 'up': a list of hidden state tensors from upsampling.
    #     """
    #     hs = []
    #     emb = self.time_embed(timestep_embedding(timesteps, self.model_channels))
    #     #如果模型是条件的（即具有类别），则根据标签 y 更新时间嵌入 emb。具体来说，标签会通过 label_emb 转换为嵌入，并加到时间嵌入上。
    #     if self.num_classes is not None:
    #         assert y.shape == (x.shape[0],)
    #         emb = emb + self.label_emb(y)
    #     result = dict(down=[], up=[])
    #     h = x.type(self.inner_dtype)
    #     #通过 input_blocks（可能是一个包含多个模块的列表）逐步处理输入数据 h。每经过一个模块，都将当前的隐藏状态 h 添加到 hs 列表，并将其保存在 result["down"] 中。
    #     for module in self.input_blocks:
    #         h = module(h, emb)
    #         hs.append(h)
    #         result["down"].append(h.type(x.dtype))
    #
    #     #经过 input_blocks 后的输出会传递到 middle_block（中间阶段），这是模型中最小分辨率的块，通常代表模型的“瓶颈”层，包含最紧凑的表示。这个中间表示被保存到 result["middle"] 中。
    #     h = self.middle_block(h, emb)
    #     result["middle"] = h.type(x.dtype)
    #
    #     #通过 output_blocks（包含多个模块）进行上采样。在每个模块中，当前的隐藏状态 h 与 hs 列表中存储的隐藏状态（在上采样过程中重新利用）进行拼接。这样做可以将更高分辨率的表示与较低分辨率的表示结合，帮助恢复更多的细节信息。每次上采样的结果都会保存到 result["up"] 中。
    #     for module in self.output_blocks:
    #         cat_in = th.cat([h, hs.pop()], dim=-1)
    #         h = module(cat_in, emb)
    #         result["up"].append(h.type(x.dtype))
    #     return result
    #


