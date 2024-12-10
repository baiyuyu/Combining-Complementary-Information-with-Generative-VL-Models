import torch
from PIL import Image
import torch.nn as nn
import requests

from transformers import Blip2Processor, Blip2Model
from torch.nn import CrossEntropyLoss
from tqdm import notebook

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

from typing import Any, Optional, Tuple, Union



class CustomBlipModel(Blip2Model):
    def forward(
        self,
        pixel_values: torch.FloatTensor,
        input_ids: torch.FloatTensor,
        attention_mask: Optional[torch.LongTensor] = None,
        decoder_input_ids: Optional[torch.LongTensor] = None,
        decoder_attention_mask: Optional[torch.LongTensor] = None,
        output_attentions: Optional[bool] = None,
        output_hidden_states: Optional[bool] = None,
        labels: Optional[torch.LongTensor] = None,
        return_dict: Optional[bool] = None,
    ) -> Union:

        return_dict = return_dict if return_dict is not None else self.config.use_return_dict

        # step 1: forward the images through the vision encoder,
        # to get image embeddings of shape (batch_size, seq_len, hidden_size)
        vision_outputs = self.vision_model(
            pixel_values=pixel_values,
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states,
            return_dict=return_dict,
        )
        image_embeds = vision_outputs[0]

        # step 2: forward the query tokens through the QFormer, using the image embeddings for cross-attention
        image_attention_mask = torch.ones(image_embeds.size()[:-1], dtype=torch.long, device=image_embeds.device)

        query_tokens = self.query_tokens.expand(image_embeds.shape[0], -1, -1)
        query_outputs = self.qformer(
            query_embeds=query_tokens,
            encoder_hidden_states=image_embeds,
            encoder_attention_mask=image_attention_mask,
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states,
            return_dict=return_dict,
        )
        query_output = query_outputs[0]

        # step 3: use the language model, conditioned on the query outputs and the prompt
        language_model_inputs = self.language_projection(query_output)
        language_model_attention_mask = torch.ones(
            language_model_inputs.size()[:-1], dtype=torch.long, device=language_model_inputs.device
        )
        inputs_embeds = self.language_model.get_input_embeddings()(input_ids)
        inputs_embeds = torch.cat([language_model_inputs, inputs_embeds], dim=1)
        labels = torch.cat([torch.zeros(language_model_inputs.shape[:-1],device=input_ids.device, dtype=input_ids.dtype) -100,
                            input_ids], dim=1)

        
        

        if attention_mask is None:
            attention_mask = torch.ones_like(input_ids)
        expected_device = language_model_attention_mask.device
        attention_mask = torch.cat([language_model_attention_mask, attention_mask.to(expected_device)], dim=1)

        if self.config.use_decoder_only_language_model:
            outputs = self.language_model(
                inputs_embeds=inputs_embeds,
                attention_mask=attention_mask,
                output_attentions=output_attentions,
                output_hidden_states=output_hidden_states,
                return_dict=return_dict,
            )
            logits = outputs.logits if return_dict else outputs[0]
            loss = None
            # we compute the loss here since we need to take into account the sequence length of the query embeds
#             if labels is not None:
#                 labels = labels.to(logits.device)
#                 logits = logits[:, -labels.size(1) :, :]
#                 # Shift so that tokens < n predict n
#                 shift_logits = logits[..., :-1, :].contiguous()
#                 shift_labels = labels[..., 1:].contiguous().to(logits.device)

#                 # Flatten the tokens
#                 loss_fct = CrossEntropyLoss(reduction="mean")

#                 loss = loss_fct(shift_logits.view(-1, self.config.text_config.vocab_size), shift_labels.view(-1))
        else:
            outputs = self.language_model(
                inputs_embeds=inputs_embeds,
                attention_mask=attention_mask,
                decoder_input_ids=decoder_input_ids,
                decoder_attention_mask=decoder_attention_mask,
                output_attentions=output_attentions,
                output_hidden_states=output_hidden_states,
                return_dict=return_dict,
                labels=labels,
            )
            loss = outputs.loss if return_dict else outputs[0]
            logits = outputs.logits if return_dict else outputs[1]

        if not return_dict:
            output = (logits, vision_outputs, query_outputs, outputs)
            return ((loss,) + output) if loss is not None else output

        return {
            'loss':loss,
            'labels':labels,
            'logits':logits,
            "vision_outputs":vision_outputs,
            'qformer_outputs':query_outputs,
            'language_model_outputs':outputs,
        }

# 创建一个CustomBertModel的实例

# # 调用模型的forward方法执行前向传播
# input_ids = torch.tensor([[1, 2, 3, 4, 5]])
# attention_mask = torch.ones_like(input_ids)
# outputs = model(input_ids, attention_mask=attention_mask)

## generate prompt
# df = pd.read_csv(file_path+'df_with_prompts.csv',index_col=None)


processor = Blip2Processor.from_pretrained("Salesforce/blip2-flan-t5-xxl",cache_dir = '/var/scratch/ybi530/data/model')
print('finishload processor')

# model = Blip2Model.from_pretrained("Salesforce/blip2-opt-2.7b", cache_dir = '/var/scratch/ybi530/data/model',device_map="auto")
# print('finishload model')

model = CustomBlipModel.from_pretrained("Salesforce/blip2-flan-t5-xxl",cache_dir = '/var/scratch/ybi530/data/model',device_map="auto")
print('finishload model')

img_path = '/var/scratch/ybi530/data/data/train2014/'
import pandas as pd


## generate prompt
df = pd.read_csv('/var/scratch/ybi530/result/no_intention.csv',index_col=None)
df.head(3)



df = df[['image_url', 'intention', 'target_action', 'action1', 'action2',
       'action3', 'action4']]


df.head(3)



def create_prompt(row):
    use_columns = ['target_action', 'action1', 'action2',
       'action3', 'action4']
    choices = ''
    for i in use_columns:
        choices += row[i]+'.'
    return 'Question: What am I supposed to do '+row['intention']+'?'

df['prompt'] = df.apply(create_prompt, axis=1)

print('Create prompt')


df.loc[:,'target_action'] = df['target_action'].apply(lambda x:x[3:])

df.loc[:,'action1'] = df['action1'].apply(lambda x:x[3:])
df.loc[:,'action2'] = df['action2'].apply(lambda x:x[3:])
df.loc[:,'action3'] = df['action3'].apply(lambda x:x[3:])
df.loc[:,'action4'] = df['action4'].apply(lambda x:x[3:])

prompt = df.prompt[0]
prompt

i =1
raw_image = Image.open(img_path+df.image_url[i]).convert('RGB')
question = df['prompt'][i]

ce_loss = torch.nn.CrossEntropyLoss(ignore_index=-100, reduction='none').cuda()

loss_df=[]


for i in notebook.tqdm(range(len(df))):
    raw_image = Image.open(img_path+df.image_url[i]).convert('RGB')
    question = df['prompt'][i]
    answer_options = [df.action1[i],df.action2[i],df.action3[i],df.action4[i],df.target_action[i]]
    loss_list = []
    for j in answer_options:
        prompt = question + ' ' + j
        inputs = processor(raw_image, prompt, return_tensors="pt").to(device)
        out = model(**inputs)
        shift_logits = out['logits'][..., :-1, :].contiguous()
        shift_labels = out['labels'][..., 1:].contiguous()
        condition_length = inputs['input_ids'].shape[1]
        losses = ce_loss(shift_logits.reshape((-1, shift_logits.shape[-1])), shift_labels.reshape((-1,)))
        loss_list.append(losses[condition_length:].sum())
    loss_df.append([i.cpu().detach().numpy() for i in loss_list])
    torch.cuda.empty_cache()
        
        

loss_df = pd.DataFrame(loss_df)

(loss_df.min(axis=1) == loss_df.iloc[:,4]).mean()

(loss_df.min(axis=1) == loss_df.iloc[:,0]).mean()

(loss_df.min(axis=1) == loss_df.iloc[:,1]).mean()

(loss_df.min(axis=1) == loss_df.iloc[:,2]).mean()

(loss_df.min(axis=1) == loss_df.iloc[:,3]).mean()

loss_df.to_csv('blip_cross_loss_df_v1.csv')