from mydatasets import get_dataloader,ChEBIdataset
import torch
import transformers
from mytokenizers import regexTokenizer
from transformers import AutoModel
from transformers import AutoTokenizer
import argparse

parser = argparse.ArgumentParser()
parser.add_argument("-i","--input",required=True)
args = parser.parse_args()
split = args.input
smtokenizer = regexTokenizer()
train_dataset = ChEBIdataset(
        dir='/nfs/home/yuanhang/AI_drug/DATA/',
        # dir='/DoctoralStudents/yuanhang/AI_drug/DATA/new_ligands/',
        smi_tokenizer=smtokenizer,
        split=split,
        replace_desc=False,
        load_state=False
        # pre = pre
    )
model = AutoModel.from_pretrained('/nfs/home/yuanhang/AI_drug/scibert')
tokz = AutoTokenizer.from_pretrained('/nfs/home/yuanhang/AI_drug/scibert')

volume = {}


model = model.cuda()
    # alllen = []
model.eval()
with torch.no_grad():
    for i in range(len(train_dataset)):
        if i%190 == 0:
            print(i)
        id = train_dataset[i]['cid']
        desc =train_dataset[i]['desc']
        #原始216，现在改为256
        tok_op = tokz(
            desc,max_length=256, truncation=True,padding='max_length'
            )
        toked_desc = torch.tensor(tok_op['input_ids']).unsqueeze(0)
        toked_desc_attentionmask = torch.tensor(tok_op['attention_mask']).unsqueeze(0)
        #原始216，现在改为256
        assert(toked_desc.shape[1]==256)
        lh = model(toked_desc.cuda()).last_hidden_state
        # print(f"desc_state shape: {lh.shape}")
        # print(f"desc_mask shape: {toked_desc_attentionmask.shape}")

        volume[id] = {'states':lh.to('cpu'),'mask':toked_desc_attentionmask}

torch.save(volume,'/nfs/home/yuanhang/AI_drug/DATA/'+split+'_desc_states_256.pt')
