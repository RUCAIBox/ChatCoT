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

from data_process import (
    DataProcessForMATH,
)

openai.api_key = YOUR_API_KEY

DATA_PROCESSER = {
    'math': DataProcessForMATH,
}

bootstrap = {
    "role": "system",
    "content": "You are an expert in mathematical problem. You should follow the example and answer the last question.",
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
                temperature=0.7,
                max_tokens=512,
                top_p=1,
                frequency_penalty=0,
                presence_penalty=0,
                stop=stop_word,
                n=5
            )
            break
        except:
            time.sleep(1)

    steps = []
    for choice in res['choices']:
        step = choice['message']['content'].strip()
        if "Problem: " in step:
            step = step.split('Problem: ')[0].strip()
        if "Q: " in step:
            step = step.split('Q: ')[0].strip()
        steps.append(step)

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


def clean(content):
    content = content.replace(' ', '')
    return content


def get_answer_boxed(content):
    pattern = '\\boxed'
    start_pos = content.rfind(pattern)
    if (start_pos == -1): return None
    answer = ''
    num_left = 0
    for i in range(start_pos + 7, len(content)):
        if (content[i] == '}' and num_left == 0):
            break
        if (content[i] == '{'):
            num_left = num_left + 1
        elif (content[i] == '}'):
            num_left = num_left - 1
        answer = answer + content[i]
    return answer


def clean_problem(content):
    # a\\frac{b}{c} -> a+\\frac{b}{c}
    mix_frac_pattern = '\d+\\\\frac'
    mix_frac = re.findall(mix_frac_pattern, content)
    for mf in mix_frac:
        tmp_digit = re.findall('\d+', mf)[0]
        content = content.replace(mf, '{}+\\frac'.format(tmp_digit))

    # _a -> _{a}
    subscript_pattern = '_\d+'
    subscript = re.findall(subscript_pattern, content)
    for ss in subscript:
        tmp_digit = ss[1:]
        content = content.replace(ss, '_{' + tmp_digit + '}')

    return content


def main(args):
    global logger_file
    if (args.logger_path is not None):
        logger_file = open(args.logger_path, 'w')
    else:
        logger_file = sys.stdout

    fout = open(args.result_path, args.write_mode)
    data_processer = DATA_PROCESSER[args.dataset_name](args.demo_path, args.num_examplar)
    
    with open('dataset/math/test.json', 'r') as fin:
        raw_dataset = json.load(fin)
    dataset = {}
    for data in raw_dataset:
        if (args.data_split != 'test' and data['knowledge_point'] != args.data_split):
            continue
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
        # if (step <= 5): continue
        data['problem'] = clean_problem(data['problem'])

        for i in range(args.num_examplar):
            prompt, real_label = data_processer.process(data, i)
            if (len(prompt) < int(4096 * 1.5)):
                break
        llm_step = call_chat_completion(prompt)
        
        data['llm_step'] = llm_step

        preds = {}
        final_pred = None
        for ls in llm_step:
            if ('The answer is' in ls):
                pred = ls.split('The answer is')[-1].strip()
            else:
                pred = get_answer_boxed(ls)
            
            if (pred is None):
                pred = ls

            if (len(pred) >= 1 and pred[-1] == '.'):
                pred = pred[:-1]
            if (pred.rfind('$') != -1):
                if (pred[pred.rfind('$') + 1:].isalpha() == True):
                    pred = pred[:pred.rfind('$') + 1]
            if (len(pred) > 2 and pred[0] == '$' and pred[-1] == '$'):
                pred = pred[1:-1]
            
            pred = clean(pred)

            if (pred not in preds):
                preds[pred] = 0
            preds[pred] = preds[pred] + 1

            if (final_pred is None or preds[pred] > preds[final_pred]):
                final_pred = pred

        data['prompt'] = prompt
        real_label = clean(real_label)
        data['llm_answer'] = final_pred
        data['real_answer'] = real_label

        data['score'] = False
        if (final_pred == real_label):
            num_correct = num_correct + 1
            data['score'] = True
        total_problem = total_problem + 1

        fout.write(json.dumps(data, indent=4, ensure_ascii=False) + '\n')

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
