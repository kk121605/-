# sci_bert_dir = None
# assert(sci_bert_dir is not None,'fill sci_bert_dir fist.')


################################
import torch
import os
from os import path as osp
import regex
import random
################################
def getrandomnumber(numbers,k,weights=None):
    if k==1:
        return random.choices(numbers,weights=weights,k=k)[0]
    else:
        return random.choices(numbers,weights=weights,k=k)


# ##构建了一个smiles  tokenizer，将每个字符视为token##############
# def build_simple_smiles_vocab(dir):
#     # 确保传入的目录参数不为空
#     assert dir is not None, 'dir and smiles_vocab can not be None at the same time.'
#     if not osp.exists(osp.join(dir,'simple_smiles_tokenizer_vocab.txt')):   # 如果指定的词汇表文件不存在，生成新的词汇表
#         # print('Generating Vocabulary for {} ...'.format(dir))
#         # 设置需要读取的文件路径，包括train.txt、validation.txt、test.txt
#         dirs = list(osp.join(dir,i) for i in ['train.txt','validation.txt','test.txt'])
#         smiles = [] # 存储所有SMILES字符串
#         for idir in dirs:   # 遍历每个数据文件（train.txt、validation.txt、test.txt）
#             with open(idir,'r') as f:
#                 for i,line in enumerate(f): # 遍历文件中的每一行
#                     if i==0: continue   # 跳过文件的第一行（通常是表头）
#                     line = line.split('\t') # 按制表符分割每行数据
#                     assert len(line)==3,'Dataset format error.' # 确保每行的数据有3列（即：ID, SMILES, 描述）
#                     if line[1]!='*': smiles.append(line[1].strip())     # 如果SMILES字符串不是 "*"，则将其加入到smiles列表中，# 去掉SMILES字符串两端的空格
#         char_set = set()    # 使用一个集合来存储所有出现的字符，这样可以去除重复字符
#         for smi in smiles:  # 遍历所有的SMILES字符串，提取其中的字符
#             for c in smi:
#                 char_set.add(c) # 添加每个字符到集合中
#         vocabstring = ''.join(char_set) # 添加每个字符到集合中
#         with open(osp.join(dir,'simple_smiles_tokenizer_vocab.txt'),'w') as f:  # 将生成的词汇表保存到文件中
#             f.write(osp.join(vocabstring))  # 写入文件
#         return vocabstring  # 返回生成的词汇表字符串
#     else:
#         print('Reading in Vocabulary...')   # 如果词汇表文件已经存在，打印信息说明正在读取词汇表
#         with open(osp.join(dir,'simple_smiles_tokenizer_vocab.txt'),'r') as f:   # 读取已经存在的词汇表文件
#             vocabstring = f.readline().strip()  # 读取并去除两端空格
#         return vocabstring  # 返回词汇表字符串

################################################原版smiles表达token##################################################
# class regexTokenizer():
#     def __init__(self,path='/DoctoralStudents/yuanhang/AI_drug/diffusion/tgm-dlm-main/datasets/SMILES/generate_vocab.txt',max_len=256):
#         print('Truncating length:',max_len)
#         with open(path,'r') as f:   # 读取提供的路径中的词汇表文件
#             x = f.readlines()
#         # 正则表达式模式，用于提取SMILES中有效的字符或符号
#         pattern =  "(\[[^\]]+]|Br?|Cl?|N|O|S|P|F|I|b|c|n|o|s|p|\(|\)|\.|=|#|-|\+|\\\\|\/|:|~|@|\?|>|\*|\$|\%[0-9]{2}|[0-9])"
#         self.rg = regex.compile(pattern)    # 编译正则表达式
#         self.idtotok  = { cnt+3:i.strip() for cnt,i in enumerate(x)}    # 创建一个字典，将ID映射到词汇表中的字符
#         self.idtotok.update(    # 添加特殊标记：[PAD]， [SOS]， [EOS]
#             {
#                 0:'[PAD]',
#                 1:'[SOS]',
#                 2:'[EOS]'
#             }
#         )
#         self.vocab_size = len(self.idtotok) #SOS, EOS, pad  # 词汇表大小，包含特殊标记
#         self.toktoid = { v:k for k,v in self.idtotok.items()}   # 创建一个字典，将字符映射到ID
#         self.max_len = max_len      # 设置最大长度
#
#     def decode_one(self, iter): # 将一个ID列表转换为SMILES字符串
#         # return "".join([self.ind2Letter(i) for i in iter]).replace('[SOS]','').replace('[EOS]','').replace('[PAD]','')
#         return "".join([self.idtotok[i.item()] for i in iter])
#     # 如果输入是一个张量，则将其解码为对应的SMILES字符串
#     def decode(self,ids:torch.tensor):
#         if len(ids.shape)==1:
#             return [self.decode_one(ids)]
#         else:
#             smiles  = []
#             for i in ids:
#                 smiles.append(self.decode_one(i))
#             return smiles
#     def __len__(self):      # 返回词汇表的大小
#         return self.vocab_size
#     def __call__(self,smis:list):   # 将SMILES字符串转换为对应的张量
#         tensors = []
#         if type(smis) is str:
#             smis = [smis]
#         for i in smis:
#             tensors.append(self.encode_one(i))
#         return torch.concat(tensors,dim=0)
#
#     # 对SMILES字符串进行损坏（加入噪声）并返回张量
#     def corrupt(self,smis:list):
#         tensors = []
#         if type(smis) is str:
#             smis = [smis]
#         for i in smis:
#             tensors.append(self.corrupt_one(i))
#         return torch.concat(tensors,dim=0)
#     def encode_one(self, smi):  # 将一个SMILES字符串编码为对应的张量
#         res = [self.toktoid[i] for i in self.rg.findall(smi)]   # 使用正则表达式将SMILES字符串分解为一个ID列表
#         res = [1] + res + [2]     # 添加特殊标记 [SOS] 和 [EOS]
#         if len(res) < self.max_len:     # 如果编码后的长度小于最大长度，填充0
#             res += [0]*(self.max_len-len(res))
#         else:
#             res = res[:self.max_len]
#             res[-1] = 2     # 确保最后一个位置是 [EOS]
#         return torch.LongTensor([res])
#
#     def encode_one(self, smi):  # 将一个SMILES字符串编码为对应的张量
#         print(self.toktoid.keys())  # 打印所有的键
#         res = [self.toktoid[i] for i in self.rg.findall(smi)]  # 使用正则表达式将SMILES字符串分解为一个ID列表
#         res = [1] + res + [2]  # 添加特殊标记 [SOS] 和 [EOS]
#         if len(res) < self.max_len:  # 如果编码后的长度小于最大长度，填充0
#             res += [0] * (self.max_len - len(res))
#         else:
#             res = res[:self.max_len]
#             res[-1] = 2  # 确保最后一个位置是 [EOS]
#         return torch.LongTensor([res])
#     # 对SMILES字符串进行腐蚀操作（添加噪声）
#     def corrupt_one(self,smi):
#         # res = [self.toktoid[i] for i in self.rg.findall(smi)]
#         # 使用正则表达式将SMILES字符串分解为一个原子列表
#         res = [i for i in self.rg.findall(smi)]
#         total_length = len(res) + 2
#         # 如果总长度超过最大长度，调用 encode_one 进行编码
#         if total_length>self.max_len:
#             return self.encode_one(smi)
#         ######################## start corruption ###########################
#         r = random.random()
#         if r<0.3:   # 随机选择腐蚀操作的类型
#             pa,ring = True,True
#         elif r<0.65:
#             pa,ring = True,False
#         else:
#             pa,ring = False,True
#         #########################
#         max_ring_num  = 1   # 默认最大环数量为1
#         ringpos = []    # 存储环的位置
#         papos = []      # 存储括号的位置
#         for pos,at in enumerate(res):       # 遍历原子列表，寻找环和括号的位置
#             if at=='(' or at==')':
#                 papos.append(pos)       # 括号位置
#             elif at.isnumeric():
#                 max_ring_num = max(max_ring_num,int(at))        # 更新最大环数量
#                 ringpos.append(pos)     # 环位置
#         # ( & ) remove   # 决定是否进行括号的移除操作
#         r = random.random()
#         if r<0.3:
#             remove,padd = True,True
#         elif r<0.65:
#             remove,padd = True,False
#         else:
#             remove,padd = False,True
#         if pa and len(papos)>0:
#             if remove:
#                 # remove pa # 随机移除一定数量的括号
#                 n_remove = getrandomnumber([1,2,3,4],1,weights = [0.6,0.2,0.1,0.1])
#                 p_remove = set(random.choices(papos,weights=None,k=n_remove))   # 随机选择要移除的括号
#                 total_length -= len(p_remove)   # 更新总长度
#                 for p in p_remove:
#                     res[p]=None     # 将这些括号设置为 None
#                     # print('debug pa delete {}'.format(p))
#         # Ring remove# 环的移除操作
#         r = random.random()
#         if r<0.3:
#             remove,radd = True,True
#         elif r<0.65:
#             remove,radd = True,False
#         else:
#             remove,radd = False,True
#         if ring and len(ringpos)>0:
#             if remove:
#                 # remove ring# 随机移除一定数量的环
#                 n_remove = getrandomnumber([1,2,3,4],1,weights = [0.7,0.2,0.05,0.05])
#                 p_remove = set(random.choices(ringpos,weights=None,k=n_remove))     # 随机选择要移除的环
#                 total_length -= len(p_remove)   # 更新总长度
#                 for p in p_remove:
#                     res[p]=None     # 将这些环设置为 None
#                     # print('debug ring delete {}'.format(p))
#         # ring add & ( ) add# 添加括号和环的操作
#         if pa:
#             if padd:
#                 # 随机决定添加括号数量
#                 n_add = getrandomnumber([1,2,3],1,weights = [0.8,0.2,0.1])
#                 n_add = min(self.max_len-total_length,n_add)    # 限制最大长度
#                 for _ in range(n_add):
#                     sele = random.randrange(len(res)+1)     # 随机选择插入位置
#                     res.insert(sele, '(' if random.random()<0.5 else ')')       # 随机插入括号
#                     # print('debug pa add {}'.format(sele))
#                     total_length += 1       # 更新总长度
#         if ring:
#             if radd:
#                 # 随机决定添加环数量
#                 n_add = getrandomnumber([1,2,3],1,weights = [0.8,0.2,0.1])
#                 n_add = min(self.max_len-total_length,n_add)    # 限制最大长度
#                 for _ in range(n_add):
#                     sele = random.randrange(len(res)+1)     # 随机选择插入位置
#                     res.insert(sele, str(random.randrange(1,max_ring_num+1)))   # 随机插入环编号
#                     # print('debug ring add {}'.format(sele))
#                     total_length += 1       # 更新总长度
#
#         ########################## end corruption ###############################
#         # print('test:',res)
#         # print('test:',''.join([i for i in res if i is not None]))
#         # 过滤掉 None 值的符号
#         res = [self.toktoid[i] for i in res if i is not None]
#         res = [1] + res + [2]   # 在序列的开头添加 [SOS]，结尾添加 [EOS]
#         # 如果总长度小于最大长度，进行填充；否则，截断并确保结尾是 [EOS]
#         if len(res) < self.max_len:
#             res += [0]*(self.max_len-len(res))
#         else:
#             res = res[:self.max_len]
#             res[-1] = 2
#         return torch.LongTensor([res])  # 返回最终的张量

#############################################Selies正则化################################################################
class regexTokenizer():
    # def __init__(self, path='/DoctoralStudents/yuanhang/AI_drug/DATA/selfies_vocab.txt', max_len=258):
    def __init__(self, path='/nfs/home/yuanhang/AI_drug/DATA/selfies_vocab.txt', max_len=258):
        print('Truncating length:', max_len)
        with open(path, 'r') as f:
            x = f.readlines()

        # 正则表达式模式，用于提取SELFIES符号
        pattern = r"(\[[^\]]+\]|Br?|Cl?|N|O|S|P|F|I|b|c|n|o|s|p|\(|\)|\.|=|#|-|\+|\\\\|\/|:|~|@|\?|>|\*|\$|\%[0-9]{2}|[0-9]|Branch[0-9]+|Ring[0-9]+)"
        self.rg = regex.compile(pattern)  # 编译正则表达式
        self.idtotok = {cnt + 3: i.strip() for cnt, i in enumerate(x)}  # 映射ID到符号
        self.idtotok.update({
            0: '[PAD]',
            1: '[SOS]',
            2: '[EOS]',
            3: '[Si]' ,
            4: '.'
        })
        self.vocab_size = len(self.idtotok)  # 词汇表大小
        self.toktoid = {v: k for k, v in self.idtotok.items()}  # 映射符号到ID
        self.max_len = max_len  # 最大长度

    def decode_one(self, iter):  # 将一个ID列表转换为SELFIES字符串
        return "".join([self.idtotok[i.item()] for i in iter])

    def decode(self, ids: torch.tensor):
        if len(ids.shape) == 1:
            return [self.decode_one(ids)]
        else:
            selfies = []
            for i in ids:
                selfies.append(self.decode_one(i))
            return selfies

    def __len__(self):  # 返回词汇表的大小
        return self.vocab_size

    def __call__(self, smis: list):  # 将SELFIES字符串转换为对应的张量
        tensors = []
        if type(smis) is str:
            smis = [smis]
        for i in smis:
            tensors.append(self.encode_one(i))
        return torch.concat(tensors, dim=0)

    def encode_one(self, smi):  # 将一个SELFIES字符串编码为对应的张量
        # print(self.toktoid)  # 查看所有映射
        res = [self.toktoid[i] for i in self.rg.findall(smi)]  # 使用正则表达式将SELFIES字符串分解为一个ID列表
        res = [1] + res + [2]  # 添加特殊标记 [SOS] 和 [EOS]
        if len(res) < self.max_len:  # 如果编码后的长度小于最大长度，填充0
            res += [0] * (self.max_len - len(res))
        else:
            res = res[:self.max_len]
            res[-1] = 2  # 确保最后一个位置是 [EOS]
        return torch.LongTensor([res])

    def corrupt(self, smis: list):  # 对SELFIES字符串进行损坏（加入噪声）
        tensors = []
        if type(smis) is str:
            smis = [smis]
        for i in smis:
            tensors.append(self.corrupt_one(i))
        return torch.concat(tensors, dim=0)

    def corrupt_one(self, smi):  # 对一个SELFIES字符串进行腐蚀操作
        res = [i for i in self.rg.findall(smi)]  # 将SELFIES字符串分解为符号列表
        total_length = len(res) + 2  # 总长度（包含 [SOS] 和 [EOS]）
        if total_length > self.max_len:
            return self.encode_one(smi)

        ######################## start corruption ###########################
        r = random.random()
        if r < 0.3:   # 随机选择腐蚀操作的类型
            pa, ring = True, True
        elif r < 0.65:
            pa, ring = True, False
        else:
            pa, ring = False, True
        #########################
        max_ring_num = 1  # 默认最大环数量为1
        ringpos = []  # 存储环的位置
        papos = []  # 存储括号的位置
        for pos, at in enumerate(res):  # 遍历符号列表，寻找环和括号的位置
            if at == '(' or at == ')':
                papos.append(pos)  # 括号位置
            elif at.isnumeric():
                max_ring_num = max(max_ring_num, int(at))  # 更新最大环数量
                ringpos.append(pos)  # 环位置

        # ( & ) remove   # 决定是否进行括号的移除操作
        r = random.random()
        if r < 0.3:
            remove, padd = True, True
        elif r < 0.65:
            remove, padd = True, False
        else:
            remove, padd = False, True
        if pa and len(papos) > 0:
            if remove:
                # 随机移除一定数量的括号
                n_remove = random.choice([1, 2, 3, 4])
                p_remove = set(random.choices(papos, k=n_remove))  # 随机选择要移除的括号
                total_length -= len(p_remove)  # 更新总长度
                for p in p_remove:
                    res[p] = None  # 将这些括号设置为 None

        # Ring remove# 环的移除操作
        r = random.random()
        if r < 0.3:
            remove, radd = True, True
        elif r < 0.65:
            remove, radd = True, False
        else:
            remove, radd = False, True
        if ring and len(ringpos) > 0:
            if remove:
                # 随机移除一定数量的环
                n_remove = random.choice([1, 2, 3, 4])
                p_remove = set(random.choices(ringpos, k=n_remove))  # 随机选择要移除的环
                total_length -= len(p_remove)  # 更新总长度
                for p in p_remove:
                    res[p] = None  # 将这些环设置为 None

        # ring add & ( ) add# 添加括号和环的操作
        if pa and padd:
            n_add = random.choice([1, 2, 3])  # 随机决定添加括号数量
            n_add = min(self.max_len - total_length, n_add)  # 限制最大长度
            for _ in range(n_add):
                sele = random.randrange(len(res) + 1)  # 随机选择插入位置
                res.insert(sele, '(' if random.random() < 0.5 else ')')  # 随机插入括号
                total_length += 1  # 更新总长度

        if ring and radd:
            n_add = random.choice([1, 2, 3])  # 随机决定添加环数量
            n_add = min(self.max_len - total_length, n_add)  # 限制最大长度
            for _ in range(n_add):
                sele = random.randrange(len(res) + 1)  # 随机选择插入位置
                res.insert(sele, str(random.randrange(1, max_ring_num + 1)))  # 随机插入环编号
                total_length += 1  # 更新总长度

        ########################## end corruption ###############################
        res = [self.toktoid[i] for i in res if i is not None]  # 过滤掉 None 值的符号
        res = [1] + res + [2]  # 在序列的开头添加 [SOS]，结尾添加 [EOS]
        if len(res) < self.max_len:
            res += [0] * (self.max_len - len(res))  # 如果总长度小于最大长度，进行填充
        else:
            res = res[:self.max_len]
            res[-1] = 2  # 确保最后一个位置是 [EOS]
        return torch.LongTensor([res])  # 返回最终的张量
