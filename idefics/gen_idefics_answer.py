import torch
from datasets import load_dataset
from peft import LoraConfig, get_peft_model
from PIL import Image
from transformers import IdeficsForVisionText2Text, AutoProcessor, Trainer, TrainingArguments, BitsAndBytesConfig
import torchvision.transforms as transforms
import warnings
warnings.filterwarnings('ignore')

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



model = IdeficsForVisionText2Text.from_pretrained(
    checkpoint,
    quantization_config=bnb_config,
    device_map="auto"
)

def check_inference(model, processor, prompts, max_new_tokens=50):
    tokenizer = processor.tokenizer
    bad_words = ["<image>", "<fake_token_around_image>"]
    if len(bad_words) > 0:
        bad_words_ids = tokenizer(bad_words, add_special_tokens=False).input_ids

    eos_token = "</s>"
    eos_token_id = tokenizer.convert_tokens_to_ids(eos_token)

    inputs = processor(prompts, return_tensors="pt").to(device)
    generated_ids = model.generate(**inputs, eos_token_id=[eos_token_id], bad_words_ids=bad_words_ids, max_new_tokens=max_new_tokens, early_stopping=True)
    generated_text = processor.batch_decode(generated_ids, skip_special_tokens=True)[0]
#     print(generated_text)
    return generated_text



## generate prompt
device = torch.device("cuda") if torch.cuda.is_available() else "cpu"
print(device)
img_path = '/var/scratch/ybi530/data/data/train2014/'
file_path = '/var/scratch/ybi530/result_Nov/'
import pandas as pd
df = pd.read_csv(file_path+'df_with_prompts(gen).csv',index_col=None)
# df.head(1).T


for j in range(5):

    j = str(j)
    print('start prompt '+j)
    df['answer_prompt_'+j] = ''
    s1 = 0
    s2 = 0
    for i in df.index[:]:
        raw_image = Image.open(img_path+df.image_url[i]).convert('RGB')
        question = df['prompt_'+j][i]

        inputs = [question,raw_image]
        # print('input generate')
        
        result = check_inference(model, processor, inputs, max_new_tokens=20)
        df['answer_prompt_'+j][i] = result
        print(i,result)
        if i>s1+50:
            print(i)
            s1 = s1+50
        if i>s2+500:
#             df.to_csv(file_path+'temp.csv')
            s2 = s2+500
    df.to_csv('temp(gen)'+j+'.csv')
    print('finish prompt'+j)

df.to_csv(file_path+'gen_idefics_answer.csv')