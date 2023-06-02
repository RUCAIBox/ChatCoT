import re
import os
import sys
import openai
import json
import time
import random
import string
import argparse

from tqdm import tqdm
from datasets import load_dataset, Dataset

import sympy
from sympy import simplify, Symbol, solve
from sympy.parsing.latex import parse_latex

from data_process import DataProcessForHotpotQA

openai.api_key = YOUR_API_KEY

DATA_PROCESSER = {
    'hotpot_qa': DataProcessForHotpotQA,
}

bootstrap = {
    "role": "system",
    "content": "You have strong ability of reasoning. You should follow the example and answer the last question.",
}

logger_file = None

def call_chat_completion(prompt, stop_word='Problem: '):
    messages = [
        bootstrap,
        {"role": "user", "content": prompt},
    ]
    while (True):
        try:
            res = openai.ChatCompletion.create(
                model='gpt-3.5-turbo',
                messages=messages,
                temperature=0,
                max_tokens=256,
                top_p=1,
                frequency_penalty=0,
                presence_penalty=0,
                stop=stop_word
            )
            break
        except:
            time.sleep(1)

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


def main(args):
    global logger_file
    if (args.logger_path is not None):
        logger_file = open(args.logger_path, 'w')
    else:
        logger_file = sys.stdout

    fout = open(args.result_path, args.write_mode)
    data_processer = DATA_PROCESSER[args.dataset_name](
        args.demo_path, 
        num_examplar=args.num_examplar, 
        is_ground_truth=args.is_ground_truth
    )
    
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
        for i in range(args.num_examplar):
            prompt, real_label = data_processer.process(data, i)
            if (len(prompt) < int(4096 * 1.5)):
                break
        prompt = prompt
        llm_step = call_chat_completion(prompt)
        
        data['llm_step'] = llm_step
        if ('the answer is' in llm_step.lower()):
            pred = llm_step.lower().split('the answer is')[-1].strip()
        else:
            pred = llm_step = call_chat_completion(prompt + llm_step + 'The answer is')

        data['prompt'] = prompt
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
    parser.add_argument('--is_ground_truth', type=bool, default=False, help='Whether use ground truth in prompt.')
    
    args = parser.parse_args()

    main(args)
