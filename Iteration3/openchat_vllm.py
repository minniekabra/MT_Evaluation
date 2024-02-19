MODEL_PATH = "./openchat_3.5"
MODEL_TYPE = 'mistral'
METHOD = 'vllm' # Use 'vllm' to run with vLLM. Leave blank to use transformer

# Data related variables

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

    
    prompt1=f"""There is an {source_language} to {translation_language} translator which translated a sentence from {source_language}  to {translation_language}. For the {translation_language} translation, the corresponding Reference sentence has also been given. Below are the source, machine translation & reference sentence:"""

    prompt2="""In the format as pointers, compare and analyze the differences between machine translation and human-made reference sentence in terms of:-

    1. Accuracy 
    2. Context understanding
    3. Grammar
    4. Syntax
    5. Overall readability. 

    Do not output an introduction and the concluding statement."""

    source=df['src']
    MT=df['mt']
    reference=df['ref']

    first_prompt ='GPT4 Correct User: ' + prompt1 + "\n " + "Source: " + source + "\n " + "Machine translation: " + MT + "\n " + "Reference translation: " + reference + "\n" + prompt2 + "<|end_of_turn|>GPT4 Correct Assistant: "
    
    df['first_prompt']=first_prompt

    return df


def second_prompt(df):

    prompt1=f"""There is an {source_language} to {translation_language} translator which translated a sentence from {source_language} to {translation_language}. For the {translation_language} translation, the corresponding Reference sentence has also been given. Below are the source, machine translation & reference sentence:"""

    prompt2="A large language model did a step by step evaluation of the translation quality which is given as below:-"

    prompt3=f"""Now based on the above  analysis and comparison with the reference by the large language model, list a single "score" between 0 and 100 to indicate the quality of the machine translation quality for the {source_language}-{translation_language} sentence. The score of 0 stands for the worst translation and 100 stands for perfect translation. Do not add any other information"""




    source=df['src']
    MT=df['mt']
    reference=df['ref']

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

    #en-de  en-mr  en-zh  et-en  ne-en  ro-en  ru-en  si-en

    sampling_params, model = load_model_vllm()
    language_mapping={'en-de': ['English', 'German'], 'en-mr': ['English', 'Marathi'], 'en-zh': ['English', 'Chinese'], 'et-en': ['Estonian', 'English' ], 'ne-en':['Norwegian', 'English'],   'ro-en': ['Romanian','English'] , 'ru-en': ['Russian','English'] , 'si-en': ['Sinhalese', 'English'] }
    dir_given='./Feb16/datasets/'
    dir_list=os.listdir(dir_given)
    for dir_name in dir_list:
        expected_file=dir_name + "_overlaps_test"        
        INPUT_DATA_PATH = dir_given+dir_name+'/'+expected_file + ".tsv"
        OUTPUT_PATH = 'oup_test_' + expected_file + '.xlsx'
        OUTPUT_PATH_CONTINOUS = 'oup_cont_test_' + expected_file + '.csv'
        set_seed(42)
        start=time.time()
        df = load_data(INPUT_DATA_PATH)
        #df=df.loc[:10]
        df['unique_id'] = df.index
        
        source_language=language_mapping[dir_name][0]
        translation_language=language_mapping[dir_name][1]
        df=df.apply(first_prompt, axis=1)

        if METHOD=='vllm':
            
            runner_start=time.time()
            prompt_type='first_prompt'
            df = generate_vllm(sampling_params, model, df, prompt_type)

            df=df.apply(second_prompt, axis=1) 
            prompt_type='second_prompt'
            df = generate_vllm(sampling_params, model, df, prompt_type) 
            runner_end=time.time()
            print("time taken for inference is",(runner_end-runner_start)/60,"mins")
        else:  
            a=1
        write_data(df)
        
        end=time.time()
        print('Time taken in inference of ', len(df), ' rows is ', (end-start)/60, ' mins')





