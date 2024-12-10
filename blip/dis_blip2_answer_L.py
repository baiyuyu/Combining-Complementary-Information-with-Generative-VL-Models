import torch
from PIL import Image
import requests

from transformers import Blip2Processor, Blip2ForConditionalGeneration
import random

import pickle
import time
import pandas as pd
import warnings
warnings.filterwarnings('ignore')
#os.environ['HF_HOME'] = '/var/scratch/ybi530/data'

device = torch.device("cuda") if torch.cuda.is_available() else "cpu"
print(device)
img_path = '/var/scratch/ybi530/data/data/train2014/'
file_path = '/var/scratch/ybi530/result_Nov/'
white_image = Image.open(img_path+'white.jpg').convert('RGB')

## generate prompt
df = pd.read_csv(file_path+'df_with_prompts.csv',index_col=None)
# df.head(1).T

processor = Blip2Processor.from_pretrained("Salesforce/blip2-flan-t5-xxl",cache_dir = '/var/scratch/ybi530/data/model')
print('finishload processor')

model = Blip2ForConditionalGeneration.from_pretrained("Salesforce/blip2-flan-t5-xxl", cache_dir = '/var/scratch/ybi530/data/model',device_map="auto")
print('finishload model')

for j in range(0,30,5):

    j = str(j)
    print('start prompt '+j)
    df['answer_prompt_'+j] = ''
    s1 = 0
    s2 = 0
    for i in df.index[:]:
        raw_image = white_image
        question = df['prompt_'+j][i]

        inputs = processor(raw_image, question, return_tensors="pt").to(device)
        # print('input generate')
        out = model.generate(**inputs)
        result = processor.decode(out[0], skip_special_tokens=True)
        df['answer_prompt_'+j][i] = result
        print(i,result,question)
        if i>s1+50:
            print(i)
            s1 = s1+50
        if i>s2+500:
            df.to_csv(file_path+'temp.csv')
            s2 = s2+500
    df.to_csv('temp'+j+'.csv')
    print('finish prompt'+j)

df.to_csv(file_path+'dis_blip_answer_all_L.csv')






# df.to_csv('result/gen_action_prompt_all_2(with_bert_score).csv')
