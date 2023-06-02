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

#########################    Demonstration    #########################

prompt_1_cot = [
    "Problem: Ramanujan and Hardy played a game where they both picked a complex number. If the product of their numbers was $32-8i$, and Hardy picked $5+3i$, what number did Ramanujan pick?\nLet's think step by step to solve this problem",
    "First, we need to find out how to use Hardy's number and the product to find Ramanujan's number",
    "Second, we can start by setting Ramanujan's number as $a+bi$ and then use the fact that the product of their numbers is $32-8i$ to set up an equation. We have:\n\n$(5+3i)(a+bi) = 32-8i$\n\nExpanding the left side, we get:\n\n$(5a-3b) + (3a+5b)i = 32-8i$\n\nEquating the real and imaginary parts, we get the system of equations:\n\n$5a-3b = 32$\n$3a+5b = -8$",
    "[ES]",
    "$a$, $b$ [SEP]",
    "$5*a-3*b = 32$, $3*a+5*b = -8$ [\ES]",
    "Results: $a = 4$, $b = -4$",
    "Finally, we can conclude that Ramanujan picked the complex number $4-4i$. Therefore, the answer is 4-4i"
]

prompt_2_cot = [
    "Problem: Simplify\n\\[\\frac{1}{\\log_{15} 2 + 1} + \\frac{1}{\\log_{10} 3 + 1} + \\frac{1}{\\log_{6} 5 + 1}.\\]\nLet's think step by step to solve this problem",
    "First, we need to find out the values of $\\frac{1}{\\log_{15} 2 + 1} + \\frac{1}{\\log_{10} 3 + 1} + \\frac{1}{\\log_6 5 + 1}$",
    "[CALC]",
    "$\\frac{1}{\\log_{15} 2 + 1} + \\frac{1}{\\log_{10} 3 + 1} + \\frac{1}{\\log_{6} 5 + 1}$ [\CALC]",
    "Results: $\\frac{1}{\\log_{15} 2 + 1} + \\frac{1}{\\log_{10} 3 + 1} + \\frac{1}{\\log_{6} 5 + 1} = 2$",
    "Finally, we can simplify the expression to just 2. Therefore, the answer is 2"
]

prompt_3_cot = [
    "Problem: A parabola with equation $y=x^2+bx+c$ passes through the points $(-1,-11)$ and $(3,17)$. What is $c$?\nLet's think step by step to solve this problem",
    "First, we need to find out the values of $c$ using the given points and the equation of the parabola",
    "[ES]",
    "$b$, $c$ [SEP]",
    "$(-1)^2 - b + c = -11$, $(3)^2 + 3b + c = 17$ [\ES]",
    "Results: $b = 3$, $c = -7$",
    "Finally, we can conclude that $c=-7$. Therefore, the answer is -7",
]

prompt_4_cot = [
    "Problem: A sequence consists of $2010$ terms.  Each term after the first is 1 larger than the previous term.  The sum of the $2010$ terms is $5307$.  When every second term is added up, starting with the first term and ending with the second last term, what is the sum?\nLet's think step by step to solve this problem",
    "First, we need to find out the relation between each odd-numbered term with the following even-numbered term",
    "Second, we pair the first term with the second, the third term with the fourth, and so on, until we pair the 2009th term with the 2010th term. There are 1005 such pairs.",
    "Third, we suppose that $S$ is the sum of the odd-numbered terms in the sequence. In each pair, the even-numbered term is one bigger than the odd-numbered term. That is, $x_2-x_1=1$, $x_4-x_3=1$, and so on.  Therefore, the sum of the even-numbered terms is 1005 greater than the sum of the odd-numbered terms. Thus, the sum of the even-numbered terms is $S+1005$.\n\nSince the sum of all of the terms equals the sum of the odd-numbered terms plus the sum of the even-numbered terms, then $S+(S+1005)=5307$.",
    "[ES]",
    "$S$ [SEP]",
    "$S+(S+1005)=5307$ [\ES]",
    "Results: $S = 2151$",
    "Finally, we know that $S$ is the sum of the odd-numbered terms in the sequence. Therefore, the answer is 2151",
]

prompt_5_cot = [
    "Problem: A function $f$ has the property that $f(3x-1)=x^2+x+1$ for all real numbers $x$.  What is $f(5)$?\nLet's think step by step to solve this problem",
    "First, we need to find out the value of $x$ that corresponds to $f(5)$ using the given property of the function.",
    "[ES]",
    "$x$ [SEP]",
    "$3x-1=5$ [\ES]",
    "Results: $x = 2$",
    "Second, substituting $x=2$, we get: $f(3(2)-1)=2^2+2+1$",
    "[CALC]",
    "$2^2+2+1$ [\CALC]",
    "Results: $2^2+2+1 = 7$",
    "Finally, $f(5)=f(3(2)-1)=2^2+2+1=7$. Therefore, the answer is 7",
]


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
                max_tokens=512,
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


def chat_and_reasoning(data, max_hop=8):
    def get_response():
        nonlocal messages
        stop_word = ['[\\ES]', '[\\CALC]', '[\ES]', '[\CALC]']
        response = call_chat_completion(messages, stop_word=stop_word)
        messages = messages + response
        return response
    
    messages = ""
    messages = messages + ' '.join(prompt_1_cot) + '\n'
    messages = messages + ' '.join(prompt_2_cot) + '\n'
    messages = messages + ' '.join(prompt_3_cot) + '\n'
    messages = messages + ' '.join(prompt_4_cot) + '\n'
    messages = messages + ' '.join(prompt_5_cot) + '\n'
    messages = messages + "Problem: {}\nLet's think step by step to solve this problem".format(data['problem'])

    answer = None
    for i in range(max_hop):
        # Step 1: decompose sub-problem
        response = get_response()
        
        if ('the answer is' in response): 
            answer = response.split('the answer is')[-1].strip()
            break
        
        # Step 2: Use tool
        pos_calc = response.rfind('[CALC]')
        pos_es = response.rfind('[ES]')

        if (pos_calc > pos_es):
            raw_all_equations = response.split('[CALC]')[-1].strip()
            print(raw_all_equations)

            calculator_pattern = '\$.+?\$'
            all_equations = []
            for equ in re.findall(calculator_pattern, raw_all_equations):
                equ = equ.strip()
                if (len(equ) > 2 and equ[0] == '$' and equ[-1] == '$'):
                    equ = equ[1:-1]
                all_equations.append(equ)

            calc_result = []
            for equ in all_equations:
                if (len(equ) > 2 and equ[0] == '$' and equ[-1] == '$'):
                    equ = equ[1:-1]
                equ = equ.replace('\\%', '/ 100')
                print(equ)
                try:
                    sympy_equ = parse_latex(equ)
                    simplify_equ = simplify(sympy_equ)
                    number_result = sympy.N(simplify_equ)
                except:
                    continue
                print(number_result)
                try:
                    number_result = float(number_result)
                    if (math.isinf(number_result) == True):
                        calc_result.append('the result of {} is too large.'.format(equ))
                    else:
                        calc_result.append('${} = {} = {}$'.format(equ, simplify_equ, number_result))
                except:
                    calc_result.append('${} = {}$'.format(equ, simplify_equ))
            if (len(calc_result) > 0):
                results = 'Results: {}'.format(', '.join(calc_result))
                print(results)
                messages = messages + f'[\CALC] {results} '
            else:
                messages = messages + f'[\CALC]'

        elif (pos_es > pos_calc):
            es_corpus = response.split('[ES]')[-1].strip()
            print(es_corpus)

            raw_unknown_variable = es_corpus.split('[SEP]')[0].strip()
            unknown_variable = []
            for var in raw_unknown_variable.split(','):
                var = var.strip()
                if (len(var) > 2 and var[0] == '$' and var[-1] == '$'):
                    var = var[1:-1]
                unknown_variable.append(var)
            print(unknown_variable)

            raw_equation_system = raw_unknown_variable = es_corpus.split('[SEP]')[-1].strip()
            equation_system = [var.strip() for var in raw_equation_system.split(',')]
            print(equation_system)

            sympy_equ = []
            for equ in equation_system:
                try:
                    if (len(equ) > 2 and equ[0] == '$', equ[-1] == '$'):
                        equ = equ[1:-1]
                    if ('=' in equ):
                        splited_equ = equ.split('=')
                        equ = '{} - ({})'.format(splited_equ[0], splited_equ[-1])
                    equ = equ.replace('\\%', '/ 100')
                    equ = parse_latex(equ)
                    equ = simplify(equ)
                    sympy_equ.append(equ)
                except:
                    continue
            
            variable = []
            for var in unknown_variable:
                variable.append(Symbol(var))

            print(variable)
            print(sympy_equ)

            try:
                raw_results = solve(sympy_equ, variable)
                print(raw_results)
                print('In first try')
                equ_result = []
                for var, val in raw_results.items():
                    equ_result.append('${} = {}$'.format(var, val))
                
                results = 'Results: {}'.format(', '.join(equ_result))
                messages = messages + f'[\ES] {results} '
            except:
                try:
                    raw_results = list(solve(sympy_equ, variable))
                    print(raw_results)
                    print('In second try')
                    equ_result = []
                    for result in raw_results:
                        for var, val in zip(unknown_variable, result):
                            equ_result.append('${} = {}$'.format(var, val))
                    
                    results = 'Results: {}'.format(', '.join(equ_result)) 
                    print(results)
                    print('---------------------------------')
                    messages = messages + f'[\ES] {results} '
                except:
                    messages = messages + '[\ES] '
        
        else:
            break

    return messages, answer


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
        data['problem'] = clean_problem(data['problem'])
        conversation, pred = chat_and_reasoning(data)

        data['tool_reasoning'] = conversation
        if (pred is None or len(pred) == 0):
            pred = "no answer"

        if (len(pred) >= 1 and pred[-1] == '.'):
            pred = pred[:-1]
        if (len(pred) > 2 and pred[0] == '$' and pred[-1] == '$'):
            pred = pred[1:-1]
        if (pred.rfind('$') != -1):
            if (pred[pred.rfind('$') + 1:].isalpha() == True):
                pred = pred[:pred.rfind('$') + 1]
        
        real_label = get_answer_boxed(data['solution'])

        pred = clean(pred)
        real_label = clean(real_label)
        data['llm_answer'] = pred
        data['real_answer'] = real_label

        data['score'] = False
        if (pred == real_label):
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
