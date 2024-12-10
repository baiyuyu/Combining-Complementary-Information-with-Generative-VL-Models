import torch
from datasets import load_dataset
from peft import LoraConfig, get_peft_model
from PIL import Image
from transformers import IdeficsForVisionText2Text, AutoProcessor, Trainer, TrainingArguments, BitsAndBytesConfig
import torchvision.transforms as transforms

from transformers import IdeficsModel
from typing import Any, Optional, Tuple, Union,List
from modeling_attn_mask_utils import _prepare_4d_causal_attention_mask_for_sdpa

device = "cuda" if torch.cuda.is_available() else "cpu"

# checkpoint = "HuggingFaceM4/tiny-random-idefics"
checkpoint = "HuggingFaceM4/idefics-9b-instruct"
# 这个效果比较好
# checkpoint = "HuggingFaceM4/idefics-9b"


# Here we skip some special modules that can't be quantized properly
bnb_config = BitsAndBytesConfig(
    load_in_4bit=True,
    bnb_4bit_use_double_quant=True,
    bnb_4bit_quant_type="nf4",
    bnb_4bit_compute_dtype=torch.float16,
    llm_int8_skip_modules=["lm_head", "embed_tokens"],
)

processor = AutoProcessor.from_pretrained(checkpoint, use_auth_token=True)
tokenizer = processor.tokenizer
bad_words = ["<image>", "<fake_token_around_image>"]
if len(bad_words) > 0:
    bad_words_ids = tokenizer(bad_words, add_special_tokens=False).input_ids

eos_token = "</s>"
eos_token_id = tokenizer.convert_tokens_to_ids(eos_token)



import transformers




class CustomideficsModel(IdeficsForVisionText2Text):
    def __init__(self, config, vision_model=None):
        super().__init__(config)
        self.model = Mysubclass2(config)
        print('haha2')
#         self.model = IdeficsModel(config)
    
    def forward(
        self,
        input_ids: torch.LongTensor = None,
        attention_mask: Optional[torch.Tensor] = None,
        position_ids: Optional[torch.LongTensor] = None,
        past_key_values: Optional[List[torch.FloatTensor]] = None,
        inputs_embeds: Optional[torch.FloatTensor] = None,
        pixel_values: Optional[torch.FloatTensor] = None,
        image_encoder_embeddings: Optional[torch.FloatTensor] = None,
        perceiver_embeddings: Optional[torch.FloatTensor] = None,
        image_attention_mask: Optional[torch.Tensor] = None,
        labels: Optional[torch.LongTensor] = None,
        use_cache: Optional[bool] = None,
        output_attentions: Optional[bool] = None,
        output_hidden_states: Optional[bool] = None,
        interpolate_pos_encoding: Optional[bool] = False,
        return_dict: Optional[bool] = None,
    ) -> Union:


        output_attentions = output_attentions if output_attentions is not None else self.config.output_attentions
        output_hidden_states = (
            output_hidden_states if output_hidden_states is not None else self.config.output_hidden_states
        )
        return_dict = return_dict if return_dict is not None else self.config.use_return_dict

        # decoder outputs consists of (dec_features, layer_state, dec_hidden, dec_attn)
        print('调用模型下一级产生输出')
        outputs = self.model(
            input_ids=input_ids,
            attention_mask=attention_mask,
            position_ids=position_ids,
            past_key_values=past_key_values,
            inputs_embeds=inputs_embeds,
            pixel_values=pixel_values,
            image_encoder_embeddings=image_encoder_embeddings,
            perceiver_embeddings=perceiver_embeddings,
            image_attention_mask=image_attention_mask,
            use_cache=use_cache,
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states,
            interpolate_pos_encoding=interpolate_pos_encoding,
            return_dict=return_dict,
        )

        hidden_states = outputs[0]
        logits = self.lm_head(hidden_states)

        loss = None
        if labels is not None:
            labels = labels.to(logits.device)
            # Shift so that tokens < n predict n
            if attention_mask is not None:
                shift_attention_mask = attention_mask[..., 1:].to(logits.device)
                shift_logits = logits[..., :-1, :][shift_attention_mask != 0].contiguous()
                shift_labels = labels[..., 1:][shift_attention_mask != 0].contiguous()
            else:
                shift_logits = logits[..., :-1, :].contiguous()
                shift_labels = labels[..., 1:].contiguous()
            # Flatten the tokens
            loss_fct = CrossEntropyLoss()
            loss = loss_fct(shift_logits.view(-1, shift_logits.size(-1)), shift_labels.view(-1))

        if not return_dict:
            output = (logits,) + outputs[1:]
            return (loss,) + output if loss is not None else output

        return {
            'logits':logits,}



model = CustomideficsModel.from_pretrained(
    checkpoint,
    quantization_config=bnb_config,
    device_map="auto"
)



## generate prompt

img_path = '/var/scratch/ybi530/data/data/train2014/'
file_path = '/var/scratch/ybi530/result_Nov/'
import pandas as pd
df = pd.read_csv(file_path+'df_with_prompts.csv',index_col=None)
# df.head(1).T

df = df.head(5)

# df

i =3
image = df.loc[i,'image_url']
text =  df.loc[i,'prompt_5']


raw_image = Image.open(img_path+image).convert('RGB')

raw_image

text = "Question: "+text+" Answer:"


input_prompt = [text,raw_image]

inputs = processor(input_prompt, return_tensors="pt").to(device)
# generated_ids = model.generate(**inputs, eos_token_id=[eos_token_id], bad_words_ids=bad_words_ids, max_new_tokens=100, early_stopping=True)
# generated_text = processor.batch_decode(generated_ids, skip_special_tokens=True)[0]



inputs.keys()

inputs.pixel_values.shape

inputs.input_ids.shape

out = model(**inputs)



