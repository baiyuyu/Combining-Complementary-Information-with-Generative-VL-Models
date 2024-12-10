
import urllib.request


import torch
from PIL import Image
import requests

import random

import pickle
import time
import pandas as pd
import warnings
warnings.filterwarnings('ignore')

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(device)

model = torch.hub.load('mair-lab/mapl', 'mapl')
model.eval()
model.to(device)

img_path = '/var/scratch/ybi530/data/data/train2014/'
file_path = '/var/scratch/ybi530/result_Nov/'
df = pd.read_csv(file_path+'df_with_prompts.csv',index_col=None)




# #  ok
# prompt = 'Question: What am I supposed to do if I feel Happy? Answer: '
# # ok
# prompt = 'Q: What am I supposed to do if I feel Happy? A: '
#  prompt = 'Q: What am I supposed to do if I feel Happy? \nA: '


for j in range(30):

    j = str(j)
    print('start prompt '+j)
    df['answer_prompt_'+j] = ''
    s1 = 0
    s2 = 0
    for i in df.index[:]:
        raw_image = Image.open(img_path+df.image_url[i]).convert('RGB')
        pixel_values = model.image_transform(raw_image).unsqueeze(0).to(device)
        
        question = 'Question:'+df['prompt_'+j][i]+" Answer:"
#         question = 'Q:'+df['prompt_'+j][i]+" A:"
        input_ids = model.text_transform(question).input_ids.to(device)

        # print('input generate')
#         out = model.generate(**inputs)
        with torch.cuda.amp.autocast(dtype=torch.bfloat16):
            generated_ids = model.generate(
            pixel_values=pixel_values,
            input_ids=input_ids,
            max_new_tokens=50,
            num_beams=5)

        result = model.text_processor.batch_decode(generated_ids, skip_special_tokens=True)[0].strip()
        df['answer_prompt_'+j][i] = result
        print(i,result[0],question)
        if i>s1+50:
            print(i)
            s1 = s1+50
        if i>s2+500:
            df.to_csv(file_path+'temp.csv')
            s2 = s2+500
    df.to_csv('temp'+j+'.csv')
    print('finish prompt'+j)

df.to_csv(file_path+'dis_mapl_answer_all.csv')