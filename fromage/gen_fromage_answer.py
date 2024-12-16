import numpy as np
import copy
import torch
from transformers import logging
logging.set_verbosity_error()


from PIL import Image
import matplotlib.pyplot as plt

from fromage import models
from fromage import utils

from PIL import Image
import requests

import random

import pickle
import time
import pandas as pd
import warnings
warnings.filterwarnings('ignore')
#os.environ['HF_HOME'] = '/var/scratch/ybi530/data'
device = torch.device("cuda") if torch.cuda.is_available() else "cpu"

# device = torch.device("cuda") if torch.cuda.is_available() else "cpu"
print(device)

img_path = '/var/scratch/ybi530/data/data/train2014/'
file_path = '/var/scratch/ybi530/result_Nov/'
df = pd.read_csv(file_path+'df_with_prompts(gen).csv',index_col=None)


# Load model used in the paper.
model_dir = './fromage_model/'
model = models.load_fromage(model_dir)

ret_scale_factor = 0



for j in range(5):

    j = str(j)
    print('start prompt '+j)
    df['answer_prompt_'+j] = ''
    s1 = 0
    s2 = 0
    for i in df.index[:]:
        raw_image = Image.open(img_path+df.image_url[i]).convert('RGB')
        question = df['prompt_'+j][i]
#         question = 'Q:'+df['prompt_'+j][i]+" \nA:"

        inputs = [raw_image] + [question]
        # print('input generate')
#         out = model.generate(**inputs)
        result = model.generate_for_images_and_texts(inputs, num_words=32, ret_scale_factor=ret_scale_factor, max_num_rets=3)
        df['answer_prompt_'+j][i] = result[0]
        print(i,result[0],question)
        if i>s1+50:
            print(i)
            s1 = s1+50
        if i>s2+500:
            df.to_csv(file_path+'temp.csv')
            s2 = s2+500
    df.to_csv('temp(gen)'+j+'.csv')
    print('finish prompt'+j)

df.to_csv(file_path+'gen_fromage_answer_all.csv')