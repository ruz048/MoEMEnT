from datasets import load_dataset
import json
import math
import time
import os
from openai import AzureOpenAI

import base64  

def encode_to_base64(input_string):  
    # Convert string to bytes  
    message_bytes = input_string.encode('utf-8')  
      
    # Encode bytes to base64  
    base64_bytes = base64.b64encode(message_bytes)  
      
    # Convert base64 bytes back to string  
    base64_string = base64_bytes.decode('utf-8')  
      
    return base64_string  

def encode_atbash(text):
    ans = ''
    N = ord('z') + ord('a')
    for s in text:
        try:
            if s.isalpha():
                ans += chr(N - ord(s))
            else:
                ans += s
        except:
            ans += s
    return ans

def encode_caesar_3(s):
    ans = ''
    shift = 3
    for p in s:
        if 'a' <= p <= 'z':
            ans += chr(ord('a') + (ord(p) - ord('a') + shift) % 26)
        elif 'A' <= p <= 'Z':
            ans += chr(ord('A') + (ord(p) - ord('A') + shift) % 26)
        else:
            ans += p

    return ans

client = AzureOpenAI(
    api_key="YOUR API KEY",  
    api_version="2024-05-01-preview",
    azure_endpoint = "YOUR ENDPOINT"
    )

def get_gpt_response(model, prompt):
    
    while True:
        time.sleep(0.2)
        try:
            c = client.chat.completions.create(   
            model=model,  
            messages=[  
                {"role": "user", "content": prompt}  
            ],  
            max_tokens=50,  
            logprobs=True,
            temperature=0 
            )  
            pred_c = c.choices[0].message.content
            prob_c = math.exp(c.choices[0].logprobs.content[0].logprob)
            break
            
        except Exception as e:
            print(e)
            print("pausing")
            print(prompt)
            if "content_filter" in (e.args[0]):
                pred_c = 'content_filter'
                break
            time.sleep(5)
            continue

    return pred_c, prob_c

split_list = ['test']
model_list = ["gpt-4-turbo-v", "gpt-4o"]
subject_list = sorted([f.split("_test.csv")[0] for f in os.listdir(os.path.join('data', "test")) if "_test.csv" in f])

print(subject_list, len(subject_list))


for split in split_list:
    for model in model_list:
        print('model: '+model)
        for subject in subject_list:
            ds = load_dataset("cais/mmlu", subject, split=split)
            
            for d in ds:
                question_input = ''
                for k, v in enumerate(d['choices']):
                    question_input+=str(k)+': '+v+'.\n'

                prompt_base64 = 'Given this question (encoded in Base64 format): {}\nWhich of the following answers is the best for this question?\n{}Only reply with 0, 1, 2, or 3.'.format(encode_to_base64(d['question']) ,question_input) 
                prompt_caesar = 'Given this question (encoded in Caesar cipher with shift 3): {}\nWhich of the following answers is the best for this question?\n{}Only reply with 0, 1, 2, or 3.'.format(encode_caesar_3(d['question']) ,question_input)
                prompt_atbash = 'Given this question (encoded in Atbash cipher): {}\nWhich of the following answers is the best for this question?\n{}Only reply with 0, 1, 2, or 3.'.format(encode_atbash(d['question']) ,question_input)
                prompt = 'Given this question: {}\nWhich of the following answers is the best for this question?\n{}Only reply with 0, 1, 2, or 3.'.format((d['question']) ,question_input)

                pred_c, prob_c = get_gpt_response(model, prompt)
                pred_c_base64, prob_c_base64 = get_gpt_response(model, prompt_base64)
                pred_c_caesar, prob_c_caesar = get_gpt_response(model, prompt_caesar)
                pred_c_atbash, prob_c_atbash = get_gpt_response(model, prompt_atbash)

                res_dic = {'pred':pred_c, 
                            'prob':prob_c,
                            'pred_base64':pred_c_base64, 
                            'prob_base64':prob_c_base64,
                            'pred_caesar':pred_c_caesar, 
                            'prob_caesar':prob_c_caesar,
                            'pred_atbash':pred_c_atbash, 
                            'prob_atbash':prob_c_atbash,
                            'label':d['answer'],
                            }

                with open("generation/mmlu_{}_{}.json".format(model, subject), "a") as outfile:
                    outfile.write(json.dumps(res_dic)+'\n')
