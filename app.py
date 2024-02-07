import streamlit as st
import random
import time
from torch import cuda, bfloat16 
from transformers import AutoTokenizer, AutoModelForCausalLM
import transformers
from PIL import Image

#Creating the model 
fmodel = AutoModelForCausalLM.from_pretrained( 
    'tiiuae/falcon-40b-instruct', 
    device_map='auto', #Loading the LLM on multiple GPUs. 
    trust_remote_code=True, 
    torch_dtype=bfloat16 ) 
fmodel.eval() 
#Tokenizer for the model 
tokenizer = AutoTokenizer.from_pretrained('tiiuae/falcon-40b-instruct')

#Reading the image 
image = Image.open('data_dir/aurak.png') 
st.image(image)
st.title(':red[AURAK] white:[GPT]')
gen_text = transformers.pipeline( 
    model=fmodel, 
    tokenizer=tokenizer, 
    task='text-generation', 
    return_full_text=True, 
    #device=device, 
    max_length=10000, 
    temperature=0.1, 
    top_p=0.15, #select from top tokens whose probability adds up to 15% 
    top_k=0, #selecting from top 0 tokens 
    repetition_penalty=1.1, #without a penalty, output starts to repeat 
    do_sample=True, 
    num_return_sequences=1, 
    eos_token_id=tokenizer.eos_token_id,)