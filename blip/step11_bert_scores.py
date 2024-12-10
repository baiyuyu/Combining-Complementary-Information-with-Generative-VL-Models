import pandas as pd

import bert_score
import time

import torch
from PIL import Image
import requests

from transformers import Blip2Processor, Blip2ForConditionalGeneration
import random

import pickle
import time
import pandas as pd
import warnings

import nltk
from nltk.translate.bleu_score import sentence_bleu
from nltk.translate.bleu_score import corpus_bleu

import numpy as np

warnings.filterwarnings('ignore')
img_path = '/var/scratch/ybi530/data/data/train2014/'
# !pwd
df = pd.read_csv('result/bert_score_distribution.csv',index_col='Unnamed: 0')


device = torch.device("cuda") if torch.cuda.is_available() else "cpu"
print(device)
# df.rename(columns={'result_no_intention':"action_generated_nointention"},inplace=True)

# df.to_csv('result/generated_actions.csv')


# df.loc[:,'action_generated'] = df['action_generated'].apply(lambda x:"I will "+x)
# df.loc[:,'action_generated_noimage'] = df['action_generated_noimage'].apply(lambda x:"I will "+x)
cands1 = list(df.sample_generated_actions.values)
cands2 = list(df.sample_generated_actions_noimage.values)
cands3 = list(df.sample_generated_actions_nointention.values)
refs = list(df.target_action.values)

cands4 = list(df.sample_generated_actions2.values)
cands5 = list(df.sample_generated_actions2_noimage.values)
cands6 = list(df.sample_generated_actions2_nointention.values)
df.columns
 
P2, R2, F1_2 = bert_score.score(cands1, refs, lang="en", verbose=True)
df['bertscore_sample1'] = F1_2
print('1')

P2, R2, F1_2 = bert_score.score(cands2, refs, lang="en", verbose=True)
df['bertscore_sample1_noimage'] = F1_2
print('2')

P2, R2, F1_2 = bert_score.score(cands3, refs, lang="en", verbose=True)
df['bertscore_sample1_nointention'] = F1_2
print('3')

P2, R2, F1_2 = bert_score.score(cands4, refs, lang="en", verbose=True)
df['bertscore_sample2'] = F1_2
print('4')

P2, R2, F1_2 = bert_score.score(cands5, refs, lang="en", verbose=True)
df['bertscore_sample2_noimage'] = F1_2
print('5')

P2, R2, F1_2 = bert_score.score(cands6, refs, lang="en", verbose=True)
df['bertscore_sample2_nointention'] = F1_2
print('6')


df.to_csv('result/bertscore_generated_actions.csv')
# df = pd.read_csv('result/generate_action_bert_score_part2.csv',index_col='Unnamed: 0')