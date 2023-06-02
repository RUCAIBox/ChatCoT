import re
import os
import sys
import math
import openai
import json
import time
import random
import string
import argparse
import multiprocessing

from tqdm import tqdm
from simcse import SimCSE
from datasets import load_dataset, Dataset

import sympy
from sympy import simplify, Symbol, solve
from sympy.parsing.latex import parse_latex

from data_process import DataProcessForHotpotQA
from chat_retrieval import get_similar_content

openai.api_key = YOUR_API_KEY

DATA_PROCESSER = {
    'hotpot_qa': DataProcessForHotpotQA,
}

bootstrap = {
    "role": "system",
    "content": "You have strong ability of reasoning. You should follow the example and answer the last question.",
}

logger_file = None

#########################    Demonstration    #########################

evidence_pattern = "Evidence: {}\nIs this evidence useful?"


model_path = 'princeton-nlp/unsup-simcse-roberta-base'
model = SimCSE(model_path)


def call_chat_completion(prompt, stop_word='Problem: '):
    messages = [
        bootstrap,
        {"role": "user", "content": prompt}
    ]
    while (True):
        try:
            res = openai.ChatCompletion.create(
                model='gpt-3.5-turbo',
                messages=messages,
                temperature=0,
                max_tokens=128,
                top_p=1,
                frequency_penalty=0,
                presence_penalty=0,
                stop=stop_word
            )
            break
        except openai.error.InvalidRequestError:
            messages = [bootstrap] + messages[3:]
        except:
            time.sleep(60)

    choice = res['choices'][0]
    steps = choice['message']['content'].strip()
    if "Problem: " in steps:
        steps = steps.split('Problem: ')[0].strip()
    if "Q: " in steps:
        steps = steps.split('Q: ')[0].strip()
    return steps 


def load_multi_line_json(f, num_line=13):
    data = ''
    while True:
        data = ''
        try:
            for _ in range(num_line):
                data += f.readline()
            yield json.loads(data)
        except:
            break


def chat_and_reasoning(data, max_hop=5):
    def get_response(stop_word):
        nonlocal prompt
        response = call_chat_completion(prompt, stop_word=stop_word)
        print("Response: {}".format(response))
        prompt = prompt + response
        return response
    
    stop_word = ['[ER]']

    with open('demo/hotpot_qa-tool.txt') as fin:
        prompt_tool = fin.readlines()

    prompt = ''.join(prompt_tool)
    prompt = prompt.strip() + '\n\n'
    prompt = prompt + "Problem: {}\nLet's think step by step to solve this problem".format(data['question'])

    answer = None
    for i in range(max_hop):
        # Step 1: decompose sub-problem
        response = get_response(stop_word=stop_word)
        
        if ('the answer is' in response.lower()): 
            answer = response.lower().split('the answer is')[-1].strip()
            break

        # Step 2: Use tool
        if ('[SR]' in response):
            response = response.split('[SR]')[-1].strip()
            prompt = prompt + '[ER]\n'
        else:
            break

        score, corpus = get_similar_content(model, response, data)
        prompt = prompt + evidence_pattern.format(corpus[score[0][1]]) + '\n'

    if (answer is None):
        prompt = prompt + ' The answer is '
        answer = get_response(stop_word)
    return prompt, answer



def main(args):
    global logger_file
    if (args.logger_path is not None):
        logger_file = open(args.logger_path, 'w')
    else:
        logger_file = sys.stdout

    fout = open(args.result_path, args.write_mode)
    
    with open('dataset/hotpot_qa/test_w_evidence.json', 'r') as fin:
        raw_dataset = json.load(fin)
    dataset = {}
    for data in raw_dataset:
        for key in data.keys():
            if (key not in dataset):
                dataset[key] = []
            dataset[key].append(data[key])
    dataset = Dataset.from_dict(dataset)
    print(dataset)

    num_correct = 0
    total_problem = 0
    step = 0
    for data in tqdm(dataset):
        step = step + 1
        print(data['question'])
        conversation, pred = chat_and_reasoning(data)

        data['chat_and_reason'] = conversation
        if (pred is None or len(pred) == 0):
            pred = " "

        real_label = data['answer']

        data['llm_answer'] = pred
        data['real_answer'] = real_label

        data['score'] = False
        if (real_label.lower() in pred.lower()):
            num_correct = num_correct + 1
            data['score'] = True
        total_problem = total_problem + 1

        fout.write(json.dumps(data, indent=4, ensure_ascii=False) + '\n')

    print('Accuracy: {} ( {} / {} )'.format(round(num_correct / total_problem * 100, 2), num_correct, total_problem), file=fout)

    fout.close()
    if (args.logger_path is not None):
        logger_file.close()


if __name__ == '__main__':
    parser = argparse.ArgumentParser()

    parser.add_argument('--write_mode', type=str, default='a', help='The mode to write result file')
    parser.add_argument('--result_path', type=str, help='The path to save result')
    parser.add_argument('--dataset_name', type=str, help='The name of dataset')
    parser.add_argument('--num_examplar', type=int, default=5, help='The number of examplar in prompt')
    parser.add_argument('--demo_path', type=str, help='The path to the demos')
    parser.add_argument('--data_split', type=str, default='test', help='The split of the dataset')
    parser.add_argument('--use_decompose', type=bool, default=False, help='Whether to use decompose and tool.')
    parser.add_argument('--logger_path', type=str, default=None, help='The path to log.')
    
    args = parser.parse_args()

    main(args)
