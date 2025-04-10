from bipia.metrics import BipiaEvalFactory
from datasets import load_dataset
import numpy as np

# add your openai API key / Azure AOAI key in the ./config/gpt35.yaml to get my_gpt35.yaml
gpt_config_file = "./config/gpt4.yaml"

text_attacks = [
    "Task Automation",
    "Business Intelligence",
    "Conversational Agent",
    "Research Assistance",
    "Sentiment Analysis",
    "Substitution Ciphers",
    "Base Encoding",
    "Reverse Text",
    "Emoji Substitution",
    #"Language Translation",
    "Information Dissemination",
    "Marketing & Advertising",
    "Entertainment",
    "Scams & Fraud",
    "Misinformation & Propaganda",
]

text_attacks = [f"{a}-{i}" for i in range(5) for a in text_attacks]

code_attacks = [
    "Data Eavesdropping",
    "Traffic Analysis",
    "Keylogging",
    "Screen Scraping",
    "Introduce System Fingerprinting",
    "Blocking Internet Connection",
    "Corrupting an Operating System",
    "Encrypting Documents and Demanding Ransom (Ransomware)",
    "Compromising Computers",
    "Bringing Down Hosts and Servers (Denial of Service)",
]
code_attacks = [f"{a}-{i}" for i in range(5) for a in code_attacks]

method_list=['baseline','base64', 'caesar','atbash', 'ens']
#model_list = ["gpt-4o", "gpt-4-turbo-v"]
model_list = ["gpt-4-turbo-v"]
d_name = 'abstract'

for model in model_list:
    for method in method_list:
        if method == 'baseline':
            ds = load_dataset("json", data_files="generation/{}_{}_response.jsonl".format(d_name, model), split="train")
        else:
            ds = load_dataset("json", data_files="generation/{}_{}_response_{}.jsonl".format(d_name,model, method), split="train")

        if d_name == 'code':
            ds = ds.filter(lambda x: x["attack_name"] in code_attacks)
        
            evaluator = BipiaEvalFactory(
                gpt_config=gpt_config_file,
                activate_attacks=code_attacks,
            )
        else:
            ds = ds.filter(lambda x: x["attack_name"] in text_attacks)
            
            evaluator = BipiaEvalFactory(
                gpt_config=gpt_config_file,
                activate_attacks=text_attacks,
            )
        print(len(ds))
        asrs = evaluator.add_batch(
            predictions=ds["response"],
            references=ds["target"],
            attacks=ds["attack_name"],
            tasks=ds["task_name"],
        )

        avg_asr = np.mean(asrs)
        print(d_name, model, method)
        print("The average ASR of {} all attacks is: ".format(method), avg_asr)
