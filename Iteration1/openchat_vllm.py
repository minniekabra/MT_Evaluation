MODEL_PATH = "./openchat_3.5"
MODEL_TYPE = 'mistral'
METHOD = 'vllm' # Use 'vllm' to run with vLLM. Leave blank to use transformer

# Data related variables
INPUT_DATA_PATH = './pe_enhi_qe_2FEB.xlsx' # abs path to input data
#INPUT_COLUMN = 'prompt1_output' # column name in input data to run inference on
OUTPUT_PATH = './pe_enhi_qe_minnie_oup.xlsx' # abs path to output data with appropriate file extension
OUTPUT_PATH_CONTINOUS = './pe_enhi_qe_2FEB_oup1_minnie_oup_cont.csv' # abs path to output data with appropriate file extension to save intermediate output

SAVE_COUNTER = 100 # Intermediate output save counter - SET THIS TO HIGH VALUE FOR VLLM


# Model Related variables
TEMPERATURE = 0
TOP_P = 1
MAX_TOKENS = 5000
MAX_NEW_TOKENS = 700
REPETITION_PENALTY = 1.0
TOP_K = 50

import os
import sys
import time
import random
import pickle
import re
import pandas as pd
import numpy as np

import torch
from transformers import LlamaForCausalLM, LlamaTokenizer
from vllm import LLM, SamplingParams


#sys.stdin.reconfigure(encoding='utf-8')
#sys.stdout.reconfigure(encoding='utf-8')

os.environ['CUBLAS_WORKSPACE_CONFIG']=':16:8'


goldml='lgpbd7220.gso.aexp.com'
dgx='lgpbddgx03.gso.aexp.com'

myhost = os.uname()[1]
if myhost.find('dgx')>=0:
    num_gpus=4
    s=1
    #gpu_mem = {0: "4GiB" , 1:"3GiB", 2:"4GiB", 3: "10GiB", 4:"3GiB", 5: "10GiB" , 6:"10GiB", 7:"GiB"} #change as per requirements
    gpu_mem = {i: "10GiB" for i in range(s, s+num_gpus)} #ideal case. P.S. in case of llama2 use 2, 4 or 5 (num_gpus) GPUs.
else:
    num_gpus = 1
    gpu_mem = {0: "35GiB" }

print(gpu_mem)

def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True

def load_data(PATH):
    if PATH.endswith('.csv'):
        df = pd.read_csv(PATH)
    elif PATH.endswith('.xlsx'):
        df = pd.read_excel(PATH)
        
    elif PATH.endswith('.pkl'):
        df = pd.read_pickle(PATH)
    elif PATH.endswith('.tsv'):
        df = pd.read_csv(PATH, sep='\t')
    else:
        raise ValueError("File extension not supported")
    #df=df.loc[:10]
    #df[INPUT_COLUMN] = df[INPUT_COLUMN].astype(str)
    df.reset_index(drop=True, inplace=True)
    print(len(df), flush=True)
    return df

def write_data(df, step=False):
    if step:
        OUT_FILE = OUTPUT_PATH_CONTINOUS
    else:
        OUT_FILE = OUTPUT_PATH
    
    if OUT_FILE.endswith('.csv'):
        df.to_csv(OUT_FILE, index=False)
    elif OUT_FILE.endswith('.xlsx'):
        df.to_excel(OUT_FILE, index=False)
    elif OUT_FILE.endswith('.pkl'):
        df.to_pickle(OUT_FILE)
    elif OUT_FILE.endswith('.tsv'):
        df.to_csv(OUT_FILE, sep='\t', index=False)
    else:
        raise ValueError("File extension not supported") 
    
    print("Output written to ", OUT_FILE, flush=True)

def load_model_llama():
    start = time.time()
    tokenizer = LlamaTokenizer.from_pretrained(MODEL_PATH)
    model = LlamaForCausalLM.from_pretrained(MODEL_PATH, device_map = 'auto', 
                                             max_memory=gpu_mem, torch_dtype=torch.float16)
    end=time.time()
    print('Time taken in model loading is ', (end-start)/60, ' mins')    
    return tokenizer, model

def load_model_vllm():
    start = time.time()
    
    sampling_params = SamplingParams(temperature=TEMPERATURE, 
                                     top_p=TOP_P, max_tokens=MAX_TOKENS)
    model = LLM(model=MODEL_PATH, tensor_parallel_size=num_gpus,
                gpu_memory_utilization=0.85)
    end=time.time()
    print('Time taken in model loading is ', (end-start)/60, ' mins')    
    return sampling_params, model




# change Minnie suggested (errors fixed)
def first_prompt(df):
       
    #llm_oup_inst='A large language model did a step by step evaluation of machine translation quality for the English-Chinese sentences which is given as below:-'

    prompt1="There is an English to Hindi translator which translated a sentence from English to Hindi. For Hindi sentence, the corresponding Reference translation has also been given. Below are the source, machine translation & reference translation:"

    prompt2="Evaluate the quality for this English-Hindi translation. You need to think step by step. First read and analyse the following source, machine translation and reference translation. List where the machine translation is bad or wrong compared with the reference translation"

    source=df['original']
    MT=df['translation']
    reference=df['post_edit']

    first_prompt ='GPT4 Correct User: ' + prompt1 + "\n " + "Source: " + source + "\n " + "Machine translation: " + MT + "\n " + "Reference translation: " + reference + "\n" + prompt2 + "<|end_of_turn|>GPT4 Correct Assistant: "
    
    df['first_prompt']=first_prompt

    return df


def second_prompt(df):

    prompt1="There is an English to Hindi translator which translated a sentence from English to Hindi. For Hindi sentence, the corresponding Reference translation has also been given. Below are the source, machine translation & reference translation:"

    prompt2="A large language model did a step by step evaluation of the translation quality which is given as below:-"

    prompt3='''Now based on the above  analysis and comparison with the reference by the large language model, list a single "score" between 0 and 100 to indicate the quality of the machine translation quality for the English-Hindi sentence. The score of 0 stands for the worst translation and 100 stands for perfect translation. Do not add external information or interpretation unless present in the text explicitly'''



    source=df['original']
    MT=df['translation']
    reference=df['post_edit']

    first_prompt_output=df['first_prompt_output']


    second_prompt = 'GPT4 Correct User: '+ prompt1+"\n "+ "\n " + "Source: " + source + "\n " + "Machine translation: " + MT + "\n " + "Reference translation: " + reference + "\n"  + prompt2 + "\n "+ first_prompt_output+ "\n "+ prompt3 + " \n" +" <|end_of_turn|>GPT4 Correct Assistant: "

    
    df['second_prompt']=second_prompt

    return df

def generate_vllm(sampling_params, model, df, prompt_type):
    all_outputs = []
    for _chat in range(0, len(df), SAVE_COUNTER):
        batch_output = []
        chat = df[prompt_type][_chat:_chat+SAVE_COUNTER].tolist()
        try:
            output = model.generate(chat, sampling_params)
            for _out in output:
                batch_output.append(_out.outputs[0].text)
        except Exception as e:
            output = [e for _ in range(SAVE_COUNTER)]
            batch_output.extend(output)
        all_outputs.extend(batch_output)
        inter_df = df[0:_chat+SAVE_COUNTER]
        inter_df[prompt_type+'_output'] = all_outputs
        write_data(inter_df, True)
    df[prompt_type+'_output'] = all_outputs
    return df


if __name__=='__main__':
    set_seed(42)
    start=time.time()
    df = load_data(INPUT_DATA_PATH)
    #df=df.loc[:10]
    df['unique_id'] = df.index
    df=df.apply(first_prompt, axis=1)

    if METHOD=='vllm':
        sampling_params, model = load_model_vllm()
        runner_start=time.time()

        prompt_type='first_prompt'
        df = generate_vllm(sampling_params, model, df, prompt_type)

        df=df.apply(second_prompt, axis=1) 
        prompt_type='second_prompt'
        df = generate_vllm(sampling_params, model, df, prompt_type) 
        runner_end=time.time()
        print("time taken for inference of 10 examples is",(runner_end-runner_start)/60,"mins")
    else:  
        a=1
    write_data(df)
    
    end=time.time()
    print('Time taken in inference is ', (end-start)/60, ' mins')




