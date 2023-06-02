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

use_retriever = {"role": "user", "content": "To solve this sub-problem, which sentence can we use to retrieve useful information?"}
continue_reasoning = {"role": "user", "content": "Continue reasoning"}
final_get_answer = {"role": "user", "content": "Base on the context, what is the answer?"}
evidence_pattern = "Evidence: {}\nIs this evidence useful?"


begin_prompt = {
    "user_begin": "You should decompose the problem into sub-problem and solve it step by step. You should follow the react in the history. In each reasoning step, you can use retriever to retrieve useful evidence to help you solve problem. Do you understand?",
    "assistant_begin": "Yes, I understand. I will follow my response in the conversation history and solve the problem step by step.",
}


model_path = 'princeton-nlp/unsup-simcse-roberta-base'
model = SimCSE(model_path)


def call_chat_completion(messages, stop_word='Problem: '):
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
    def clean_response(response):
        stop_word_list = ['To solve this sub-problem,', 'Tool (choice: ', 'Based on the context,', 'Result: ', 'Equation: ', 'Unknown variable: ', 'Solution: ']
        for sw in stop_word_list:
            if (sw in response):
                response = response.split(sw)[0].strip()
        return response
    
    def get_response(stop_word):
        nonlocal messages
        response = call_chat_completion(messages, stop_word=stop_word)
        print("Response: {}".format(response))
        messages.append({"role": "assistant", "content": response})
        return response
    
    stop_word = ['To solve this sub-problem,', 'Tool (choice: ', 'Based on the context,']
    stop_word = ['fasdfasdfasdfsad']

    with open('demo/hotpot_qa-chat.json') as fin:
        prompt_chat = json.load(fin)

    messages = [bootstrap]
    messages = messages + prompt_chat[0]['chat']
    messages = messages + prompt_chat[1]['chat']
    messages = messages + prompt_chat[2]['chat']
    messages = messages + prompt_chat[3]['chat']
    messages = messages + [
        {"role": "user", "content": begin_prompt["user_begin"]},
        {"role": "assistant", "content": begin_prompt["assistant_begin"]},
        {"role": "user", "content": "Problem: {}\nLet's think step by step to solve this problem".format(data['question'])},
    ]

    answer = None
    for i in range(max_hop):
        # Step 1: decompose sub-problem
        response = get_response(stop_word=stop_word)
        
        if ('the answer is' in response.lower()): 
            answer = response.lower().split('the answer is')[-1].strip()
            break

        # Step 2: Use tool
        messages = messages + [use_retriever]
        response = get_response(stop_word=stop_word)

        score, corpus = get_similar_content(model, response, data)

        for i in range(min(5, len(score))):
            evidence = {
                "role": "user",
                "content": evidence_pattern.format(corpus[score[i][1]])
            }
            messages = messages + [evidence]
            response = get_response(stop_word=stop_word)
            if (response.lower().startswith('no') == False):
                break
            else:
                messages = messages[:-2]
            
        messages = messages + [continue_reasoning]

    if (answer is None):
        messages = messages[:-1] + [final_get_answer]
        answer = get_response(stop_word)
    return messages[len(prompt_chat[0]['chat'] + prompt_chat[1]['chat'] + prompt_chat[2]['chat'] + prompt_chat[3]['chat']) + 1:], answer



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
