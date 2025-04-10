# MoEMEnT

The code of our work "Defense against Prompt Injection Attacks via Mixture of Encodings" published in NAACL 2025.

## Overview
Large Language Models (LLMs) have emerged as a dominant approach for a wide range of NLP tasks, with their access to external information further enhancing their capabilities. However, this introduces new vulnerabilities, known as prompt injection attacks, where external content embeds malicious instructions that manipulate the LLM’s output. Recently, the Base64 defense has been recognized as one of the most effective methods for reducing success rate of prompt injection attacks. Despite its efficacy, this method can degrade LLM performance on certain NLP tasks. 

![Flowchart](flowchart.png)


To address this challenge, we propose a novel defense mechanism: mixture of encodings, which utilizes multiple character encodings, including Base64. Extensive experimental results show that our method achieves one of the lowest attack success rates under prompt injection attacks, while maintaining high performance across all NLP tasks, outperforming existing character encoding-based defense methods. This underscores the effectiveness of our mixture of encodings strategy for both safety and task performance metrics.


## Harmfulness Evaluation

The code is built on the [BIPIA repo](https://github.com/microsoft/BIPIA), please refer to their repo for processing datasets and environment requirements.

After finish setting up environment for BIPIA, first run ```gen_bipia.py``` followed by ```bipia_mix.py``` to get the response of our method to BIPIA prompt injection attack. Then use ```eval_asr.py``` to get the resulting ASR. 

## Helpfulness Evaluation

We release code for applying our method on MMLU benchmark as an example: run ```gen_mmlu.py``` to get the response of our method to questions in MMLU. Then use ```eval_acc.py``` to get the resulting accuracy. 


## Paper
If you use this code in your research please cite the following:
```
@inproceedings{
zhang2025defense,
title={Defense against Prompt Injection Attacks via Mixture of Encodings},
author={Ruiyi Zhang and David Sullivan and Kyle Jackson and Pengtao Xie and Mei Chen},
booktitle={The 2025 Annual Conference of the Nations of the Americas Chapter of the ACL},
year={2025}
}
```
