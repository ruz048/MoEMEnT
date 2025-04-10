import evaluate
import json 
import unicodedata
import numpy as np
import re
import string 
import os

def normalize_answer(s):
  """Normalize answer."""
  s = unicodedata.normalize("NFD", s)

  def remove_articles(text):
    return re.sub(r"\b(a|an|the)\b", " ", text)

  def white_space_fix(text):
    return " ".join(text.split())

  def remove_punc(text):
    exclude = set(string.punctuation)
    return "".join(ch for ch in text if ch not in exclude)

  def lower(text):
    return text.lower()
  return white_space_fix(remove_articles(remove_punc(lower(s))))

d_name = 'mmlu'
split = 'test'

subject_list = sorted([f.split("_test.csv")[0] for f in os.listdir(os.path.join('data', "test")) if "_test.csv" in f])
model_list = ["gpt-4-turbo-v", "gpt-4o"]
file_list=['generation/{}_gpt-4-turbo-v_{}.json'.format(d_name, subject) for subject in subject_list]+['generation/{}_gpt-4o_{}.json'.format(d_name, subject) for subject in subject_list]

for model in model_list:
  print(model)
  sub, sub_base64, sub_caesar, sub_atbash, sub_mix, sub_base64_caesar  = [], [], [], [], [],[]
  for subject in subject_list:
      file = 'generation/{}_{}_{}.json'.format(d_name,model, subject)
      #print(file)
      score, score_base64, score_caesar, score_atbash, score_mix,score_base64_caesar  = [], [], [], [], [], []

      with open(file) as f:
          for l in f.readlines():
              d = json.loads(l.strip())
              d['label']=str(d['label'])
              score.append(d['label']==normalize_answer(d['pred']))
              score_base64.append(d['label']==normalize_answer(d['pred_base64']))
              score_caesar.append(d['label']==normalize_answer(d['pred_caesar']))
              score_atbash.append(d['label']==normalize_answer(d['pred_atbash']))
              score_base64_caesar.append(normalize_answer(d['pred_base64'])==normalize_answer(d['pred_caesar']))

              moe_dic={}
              moe_dic[normalize_answer(d['pred'])]=0
              moe_dic[normalize_answer(d['pred_base64'])]=0
              moe_dic[normalize_answer(d['pred_caesar'])]=0
              moe_dic[normalize_answer(d['pred_atbash'])]=0

              moe_dic[normalize_answer(d['pred'])]+=d['prob']*2
              moe_dic[normalize_answer(d['pred_base64'])]+=d['prob_base64']
              moe_dic[normalize_answer(d['pred_caesar'])]+=d['prob_caesar']
              moe_dic[normalize_answer(d['pred_atbash'])]+=d['prob_atbash']
              moe_pred = max(moe_dic, key=moe_dic.get) 
              score_mix.append(d['label']==moe_pred)

      sub.append(np.mean(score))
      sub_base64.append(np.mean(score_base64))
      sub_caesar.append(np.mean(score_caesar))
      sub_atbash.append(np.mean(score_atbash))
      sub_mix.append(np.mean(score_mix))
      sub_base64_caesar.append(np.mean(score_base64_caesar))

  print('Baseline:',np.mean(sub))
  print('Base64:',np.mean(sub_base64))
  print('Caesar:',np.mean(sub_caesar))
  print('Atbash:',np.mean(sub_atbash))
  print('Mix:',np.mean(sub_mix))
  print('Base64 Caesar Similarity:',np.mean(sub_base64_caesar))
