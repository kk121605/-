import numpy as np
import torch as th

from .gaussian_diffusion import GaussianDiffusion


def space_timesteps(num_timesteps, section_counts):
    """
    Create a list of timesteps to use from an original diffusion process,
    given the number of timesteps we want to take from equally-sized portions
    of the original process.

    For example, if there's 300 timesteps and the section counts are [10,15,20]
    then the first 100 timesteps are strided to be 10 timesteps, the second 100
    are strided to be 15 timesteps, and the final 100 are strided to be 20.

    If the stride is a string starting with "ddim", then the fixed striding
    from the DDIM paper is used, and only one section is allowed.

    :param num_timesteps: 这是原始扩散过程中的时间步总数，即你希望从一个原始的扩散过程（例如，1000步的扩散过程）中选取多少个时间步进行训练或测试。
    :param section_counts: 这是一个整数列表，表示将扩散过程分为多少个部分，以及每个部分中应包含多少个时间步。每个部分的大小由 section_counts 中的数字指定。
    例如，section_counts = [10, 15, 20] 表示将时间步分为三段，第一段包含 10 个时间步，第二段包含 15 个，第三段包含 20 个。
    特殊情况：如果 section_counts 是一个字符串并且以 "ddim" 开头，则表示采用 DDIM (Denoising Diffusion Implicit Models) 中描述的特定步幅策略。
    :return: a set of diffusion steps from the original process to use.
    """
    # if isinstance(section_counts, str):
    #     if section_counts.startswith("ddim"):
    #         desired_count = int(section_counts[len("ddim") :])
    #         for i in range(1, num_timesteps):
    #             if len(range(0, num_timesteps, i)) == desired_count:
    #                 return set(range(0, num_timesteps, i))
    #         raise ValueError(
    #             f"cannot create exactly {num_timesteps} steps with an integer stride"
    #         )
    #     section_counts = [int(x) for x in section_counts.split(",")]
    size_per = num_timesteps // len(section_counts)       #计算每个部分的基本大小。
    extra = num_timesteps % len(section_counts)   #计算剩余的部分，这部分会均匀分配给前面的几个部分。
    start_idx = 0
    all_steps = []
    #然后，循环遍历 section_counts 中的每个部分，确保每部分的步数不小于 1，并且每部分的步幅（frac_stride）能够将时间步均匀分布在该部分中。
    #假设：num_timesteps = 300，section_counts = [10, 15, 20]，size_per = 300 // 3 = 100，extra = 300 % 3 = 0
    #那么第一个部分 i = 0 会有 100 步。第二个部分 i = 1 会有 100 步。第三个部分 i = 2 会有 100 步。如果 extra > 0，则前几个部分会多分配 1 个时间步。
    for i, section_count in enumerate(section_counts):
        size = size_per + (1 if i < extra else 0)
        if size < section_count:
            raise ValueError(
                f"cannot divide section of {size} steps into {section_count}"
            )
        #frac_stride 是每个时间步之间的步幅，决定了每个部分内时间步的均匀分布。如果 section_count 是 1 或更小，则没有必要均匀分配时间步，所以直接将步幅 frac_stride 设置为 1。
        #否则，计算步幅：frac_stride = (size - 1) / (section_count - 1)。这个计算是为了保证时间步在这一部分内均匀分布。
        #例如如果 size = 100 且 section_count = 10，则：frac_stride = (100 - 1) / (10 - 1) = 99 / 9 = 11。这意味着每个时间步之间的间隔是 11 个时间步。
        if section_count <= 1:
            frac_stride = 1
        else:
            frac_stride = (size - 1) / (section_count - 1)
        #cur_idx 是当前时间步的索引（从 0 开始）。
        cur_idx = 0.0
        #taken_steps 是当前部分已经选择的时间步索引的列表。
        taken_steps = []
        #这一段是实际选择当前部分的时间步。在每次循环中，round(cur_idx) 计算当前的时间步索引，并将其加入 taken_steps 列表。cur_idx 每次增加 frac_stride，保证时间步之间的均匀间隔。
        #例如，如果 frac_stride = 11，则 cur_idx 会依次增加 11，22，33 等等。经过 section_count 次循环后，taken_steps 就是当前部分所选的时间步索引。
        for _ in range(section_count):
            taken_steps.append(start_idx + round(cur_idx))
            cur_idx += frac_stride
        #将当前部分的所有时间步索引添加到 all_steps 中。更新 start_idx，为下一个部分的起始时间步索引。
        all_steps += taken_steps
        start_idx += size
    return set(all_steps)


class SpacedDiffusion(GaussianDiffusion):
    """
    A diffusion process which can skip steps in a base diffusion process.

    :param use_timesteps: a collection (sequence or set) of timesteps from the
                          original diffusion process to retain.
    :param kwargs: the kwargs to create the base diffusion process.
    """

    def __init__(self, use_timesteps, **kwargs):
        self.use_timesteps = set(use_timesteps)  # 将 use_timesteps 转换为集合，确保其唯一性
        self.timestep_map = []  # 用来存储保留的时间步
        self.original_num_steps = len(kwargs["betas"])  # 获取原始扩散过程的步数

        base_diffusion = GaussianDiffusion(**kwargs)  # 创建一个基础的 GaussianDiffusion 对象
        last_alpha_cumprod = 1.0
        new_betas = []  # 用来存储新的 beta 值
        for i, alpha_cumprod in enumerate(base_diffusion.alphas_cumprod):
            if i in self.use_timesteps:  # 如果当前时间步被保留
                new_betas.append(1 - alpha_cumprod / last_alpha_cumprod)  # 计算新的 beta
                last_alpha_cumprod = alpha_cumprod  # 更新 last_alpha_cumprod
                self.timestep_map.append(i)  # 保存当前时间步
        kwargs["betas"] = np.array(new_betas)  # 将新的 beta 值传递给 kwargs
        super().__init__(**kwargs)  # 调用父类的初始化方法

    #这是对父类 GaussianDiffusion 中 p_mean_variance 方法的重写。它调用了 super()，但传递给它的 model 被包裹在 _wrap_model(model) 中。
    #_wrap_model(model) 方法会返回一个经过封装的模型（_WrappedModel），这个模型会根据 SpacedDiffusion 类的要求修改模型的行为（例如，根据保留的时间步来调整计算）。
    def p_mean_variance(
        self, model, *args, **kwargs
    ):  # pylint: disable=signature-differs
        # print('called p_mean_var')
        return super().p_mean_variance(self._wrap_model(model), *args, **kwargs)

    #类似于 p_mean_variance 方法，training_losses 方法也重写了父类的对应方法，并且将模型传入了 _wrap_model 中，确保模型在计算损失时使用的是经过封装的版本。
    def training_losses(
        self, model, *args, **kwargs
    ):  # pylint: disable=signature-differs
        # print('called training_losses')
        return super().training_losses(self._wrap_model(model), *args, **kwargs)

    #该方法用于封装传入的模型。如果传入的模型已经是 _WrappedModel 类型，则直接返回。如果不是，则将模型封装成 _WrappedModel，并传入以下参数：
    #model：原始的模型。 self.timestep_map：保留的时间步。 self.rescale_timesteps：可能的时间步缩放方式。 self.original_num_steps：原始的扩散步骤数。
    def _wrap_model(self, model):
        if isinstance(model, _WrappedModel):
            return model
        return _WrappedModel(
            model, self.timestep_map, self.rescale_timesteps, self.original_num_steps
        )

    def _scale_timesteps(self, t):
        # Scaling is done by the wrapped model.
        return t


class _WrappedModel:
    def __init__(self, model, timestep_map, rescale_timesteps, original_num_steps):
        #timestep_map: 这是一个列表或集合，指定了哪些时间步会被保留。它是一个索引映射，用于将原始的时间步映射到当前模型需要的时间步。
        #rescale_timesteps: 一个布尔值，指示是否需要将时间步进行缩放。比如，如果原始扩散过程中有 1000 个时间步，而当前过程只有 500 个时间步，则可能需要对时间步进行缩放处理。
        #original_num_steps: 原始扩散过程中时间步的总数，用来辅助缩放计算。
        self.model = model  # 原始的扩散模型
        self.timestep_map = timestep_map  # 时间步的映射表
        self.rescale_timesteps = rescale_timesteps  # 是否需要缩放时间步
        self.original_num_steps = original_num_steps  # 原始的时间步总数

    #这个方法使得 _WrappedModel 对象可以像一个函数一样被调用。在这里，输入是 x（输入数据）和 ts（时间步），以及可选的其他参数 *args 和 **kwargs。
    def __call__(self, x, ts, *args,**kwargs):
        # 将 self.timestep_map 转换为 tensor，并与输入的时间步 ts 对应
        #map_tensor = th.tensor(self.timestep_map, device=ts.device, dtype=ts.dtype)：将 timestep_map 转换为一个 Tensor，并确保它与时间步 ts 在同一设备（例如 GPU）上，并且数据类型一致。
        map_tensor = th.tensor(self.timestep_map, device=ts.device, dtype=ts.dtype)
        #new_ts = map_tensor[ts]：通过索引将时间步 ts 映射到 timestep_map 中，得到新的时间步 new_ts。
        new_ts = map_tensor[ts]
        # print(new_ts)
        #如果 rescale_timesteps 为 True，那么将时间步 new_ts 进行缩放。通常，扩散过程的时间步总数可能会发生变化，缩放可以确保时间步之间的相对间隔一致。
        if self.rescale_timesteps:
            # 将 new_ts 转换为浮点数，并按比例缩放。例如，如果原始扩散过程有 1000 个时间步，但现在希望将其调整为 500 个时间步，可能会将时间步乘以 1000 / 500 = 2。
            new_ts = new_ts.float() * (1000.0 / self.original_num_steps)
        # temp = self.model(x, new_ts, **kwargs)
        # print(temp.shape)
        # return temp
        # print(new_ts)
        return self.model(x, new_ts,*args, **kwargs)
