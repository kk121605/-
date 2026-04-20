from torch.utils.data import DataLoader,Dataset
import torch
import random
from rdkit import Chem
from rdkit import RDLogger
from torch.utils.data import DistributedSampler
RDLogger.DisableLog('rdApp.*')
import csv
import random
import torch
from torch.utils.data import Dataset

###创建一个支持 分布式训练 的数据加载器###
def get_dataloader(dataset, batchsize, rank, world_size):
    # 使用分布式采样器来管理数据的分配和顺序
    sampler = DistributedSampler(dataset, num_replicas=world_size, rank=rank, shuffle=True)

    # 这里的collate函数将会处理基因表达数据以及其他现有数据
    def collate(batch):
        # 提取每个样本的SELFIES分子字符串
        toked_smis = [i['tok_smiles'] for i in batch]
        # 提取每个样本的描述性状态
        desc_states = [i['desc_state'] for i in batch]
        # 提取每个样本的描述性掩码
        desc_mask = [i['desc_mask'] for i in batch]
        # 提取每个样本的污染后的SELFIES分子字符串
        corrupted_toked_smis = [i['corrupted_toked_smis'] for i in batch]
        # 提取每个样本的基因表达数据，假设基因表达数据在样本的第5列及其之后
        gene_expression = [i['gene_expression'] for i in batch]  # 基因表达数据

        # print("toked_smis 类型:", type(toked_smis), "形状:", [x.shape for x in toked_smis])
        # print("gene_expression 类型:", type(gene_expression), "形状:", [x.shape for x in gene_expression])

        return torch.concat(toked_smis, dim=0), torch.concat(desc_states, dim=0), torch.concat(desc_mask,dim=0), torch.concat(corrupted_toked_smis, dim=0), torch.concat(gene_expression, dim=0)

    # 创建 DataLoader 对象
    dataloader = DataLoader(
        dataset,
        batch_size=batchsize,
        shuffle=False,  # 数据加载时不需要打乱，因为已经使用了分布式采样器
        collate_fn=collate,  # 使用自定义的collate函数
        sampler=sampler  # 使用分布式采样器
    )

    # 定义循环函数以便在多个epoch之间持续加载数据
    def cycle():
        ec = 0
        while True:
            dataloader.sampler.set_epoch(ec)
            for i in dataloader:
                # print(i[4])
                yield i
            ec += 1

    return iter(cycle())


class ChEBIdataset(Dataset):
    def __init__(self, dir, smi_tokenizer, split, replace_desc=False, pre=None, prob=0, load_state=True,
                 corrupt_prob=0.0, mask_desc=False):
        super().__init__()
        self.dir = dir
        self.smi_tokenizer = smi_tokenizer
        self.split = split
        self.replace_desc = replace_desc
        self.pre = pre
        self.prob = prob
        self.corrupt_prob = corrupt_prob
        print('corruption prob is {}'.format(self.corrupt_prob))
        self.mask_desc = mask_desc
        print('mask_desc is {}'.format(self.mask_desc))
        self.gene_num = None  # **初始化 gene_num**
        assert split in ['train', 'test', 'validation', 'mini', 'train_val_256', 'AKT1', 'AKT2', 'AURKB', 'CTSK', 'EGFR', 'HDAC1', 'MTOR', 'PIK3CA', 'SMAD3', 'TP53']
        self.ori_data = self.get_ori_data()

        if self.gene_num is None:
            raise ValueError("no gene_num")

        print(f"🔹dataset `{split}` gene_num = {self.gene_num}")  # **打印 gene_num**

        self.load_state = load_state
        if load_state:
            self.desc_state = self.get_desc_state()

    def get_desc_state(self):
        import os.path as osp
        file_path = osp.join(self.dir, self.split + '_desc_states_256.pt')
        return torch.load(file_path)

    def get_ori_data(self):
        import os.path as osp
        res = []
        file_path = osp.join(self.dir, self.split + '.csv')

        # 使用csv模块读取没有列名的数据
        with open(file_path, 'r') as f:
            reader = csv.reader(f)
            for i, line in enumerate(reader):
                gene_id = line[0].strip()
                smiles = line[2].strip()
                if len(line) > 3:  # 确保数据中有文本描述
                    desc = line[3].strip()
                else:
                    desc = ""

                # 基因表达数据从第五列开始
                gene_expression = [float(x) for x in line[4:]]  # 从第五列开始读取基因表达数据
                if i == 0:  # **记录 gene_num**
                    self.gene_num = len(gene_expression)  # **存储 gene_num**

                if self.replace_desc:
                    import spacy
                    nlp = spacy.load('en_core_web_sm')
                    doc = nlp(desc)
                    for token in doc:
                        if token.text == 'is':
                            desc = 'The molecule ' + desc[token.idx:]
                            break

                res.append((gene_id, smiles, desc, gene_expression))
                # print(res)

        return res

    def __len__(self):
        return len(self.ori_data)

    def permute(self, smiles):
        p = random.random()
        if p < self.prob:
            print("PERMUTE SMILE")
            return changeorder(smiles, shuffle=True)
        else:
            return smiles

    def __getitem__(self, idx):
        data = self.ori_data[idx]
        dic = {'cid': data[0], 'smiles': self.permute(data[1]), 'desc': data[2], 'gene_expression': data[3]}
        dic['gene_expression'] = torch.tensor(dic['gene_expression'], dtype=torch.float32).unsqueeze(0)
        dic['tok_smiles'] = self.smi_tokenizer(dic['smiles'])
        dic['corrupted_toked_smis'] = self.smi_tokenizer.corrupt(
            dic['smiles']) if random.random() < self.corrupt_prob else dic['tok_smiles']
        dic['tok_desc'] = None
        dic['desc_mask'] = None
        if self.load_state:
            dic['desc_state'] = self.desc_state[data[0]]['states']
            dic['desc_mask'] = self.desc_state[data[0]]['mask']
            if self.mask_desc:
                dic['desc_state'] = torch.zeros_like(dic['desc_state'])
                dic['desc_mask'] = torch.ones_like(dic['desc_mask'])
        return dic

def changeorder(smiles,shuffle):
    original_smiles = smiles # Replace with your original SMILES string
    # Convert the original SMILES string to an RDKit molecule object
    mol = Chem.MolFromSmiles(original_smiles)
    if mol is None:
        print("Wrong in original dataset")
    Chem.Kekulize(mol)
    # Get the atom indices in the molecule
    atom_indices = [atom.GetIdx() for atom in mol.GetAtoms()]
    if shuffle:
        random.shuffle(atom_indices)
    reordered_mol = Chem.RenumberAtoms(mol, atom_indices)
    # if k:
    #     print(reordered_mol)
    # Generate the new SMILES string
    new_smiles = Chem.MolToSmiles(reordered_mol,kekuleSmiles=True)
    return new_smiles
