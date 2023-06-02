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

#########################    Demonstration    #########################

prompt_1 = "Problem: Ramanujan and Hardy played a game where they both picked a complex number. If the product of their numbers was $32-8i$, and Hardy picked $5+3i$, what number did Ramanujan pick?\n\
Solution: Let's think step by step. To answer this question, first we need to find out how to use Hardy's number and the product to find Ramanujan's number.\n\
To solve this sub-problem, which tool can we use?\n\
Tool (choice: calculator, equation solver, do not use tool): do not use tool\n\
Based on the context, We can start by setting Ramanujan's number as $a+bi$ and then use the fact that the product of their numbers is $32-8i$ to set up an equation. We have:\n\n$(5+3i)(a+bi) = 32-8i$\n\nExpanding the left side, we get:\n\n$(5a-3b) + (3a+5b)i = 32-8i$\n\nEquating the real and imaginary parts, we get the system of equations:\n\n$5a-3b = 32$\n$3a+5b = -8$\n\
To solve this sub-problem, which tool can we use?\n\
Tool (choice: calculator, equation solver, do not use tool): equation solver\
Unknown variable: a, b\n\
Equation: $5*a-3*b = 32$, $3*a+5*b = -8$\n\
Result: $a = 4$, $b = -4$\n\
Based on the context, we can conclude that Ramanujan picked the complex number $4-4i$. Therefore, the answer is 4-4i\n\
"

prompt_2 = "Problem: Simplify\n\\[\\frac{1}{\\log_{15} 2 + 1} + \\frac{1}{\\log_{10} 3 + 1} + \\frac{1}{\\log_{6} 5 + 1}.\\]\n\
Solution: Let's think step by step. To answer this question, first we need to find out the values of $\\frac{1}{\\log_{15} 2 + 1} + \\frac{1}{\\log_{10} 3 + 1} + \\frac{1}{\\log_6 5 + 1}$.\n\
To solve this sub-problem, which tool can we use?\n\
Tool (choice: calculator, equation solver, do not use tool): calculator\n\
Equation: $\\frac{1}{\\log_{15} 2 + 1} + \\frac{1}{\\log_{10} 3 + 1} + \\frac{1}{\\log_{6} 5 + 1}$\n\
Result: $\\frac{1}{\\log_{15} 2 + 1} + \\frac{1}{\\log_{10} 3 + 1} + \\frac{1}{\\log_{6} 5 + 1} = 2$\n\
Based on the context, we can simplify the expression to just 2. Therefore, the answer is 2\n\
"

prompt_3 = "Problem: A parabola with equation $y=x^2+bx+c$ passes through the points $(-1,-11)$ and $(3,17)$. What is $c$?\n\
Solution: Let's think step by step. To answer this question, first we need to find out the values of $c$ using the given points and the equation of the parabola.\n\
To solve this sub-problem, which tool can we use?\n\
Tool (choice: calculator, equation solver, do not use tool): equation solver\n\
Unknown variable: b, c\n\
Equation: $(-1)^2 - b + c = -11$, $(3)^2 + 3b + c = 17$\n\
Result: $b = 3$, $c = -7$\n\
Based on the context, we can conclude that $c=-7$. Therefore, the answer is -7.\n\
"

prompt_4 = "Problem: A sequence consists of $2010$ terms.  Each term after the first is 1 larger than the previous term.  The sum of the $2010$ terms is $5307$.  When every second term is added up, starting with the first term and ending with the second last term, what is the sum?\n\
Solution: Let's think step by step. To answer this question, first we need to find out the relation between each odd-numbered term with the following even-numbered term.\n\
To solve this sub-problem, which tool can we use?\n\
Tool (choice: calculator, equation solver, do not use tool): do not use tool\n\
Based on the context, we pair the first term with the second, the third term with the fourth, and so on, until we pair the 2009th term with the 2010th term. There are 1005 such pairs.\n\
To solve this sub-problem, which tool can we use?\n\
Tool (choice: calculator, equation solver, do not use tool): do not use tool\n\
Based on the context, we suppose that $S$ is the sum of the odd-numbered terms in the sequence. In each pair, the even-numbered term is one bigger than the odd-numbered term. That is, $x_2-x_1=1$, $x_4-x_3=1$, and so on.  Therefore, the sum of the even-numbered terms is 1005 greater than the sum of the odd-numbered terms. Thus, the sum of the even-numbered terms is $S+1005$.\n\nSince the sum of all of the terms equals the sum of the odd-numbered terms plus the sum of the even-numbered terms, then $S+(S+1005)=5307$.\n\
To solve this sub-problem, which tool can we use?\n\
Tool (choice: calculator, equation solver, do not use tool): equation solver\n\
Unknown variable: S\n\
Equation: $S+(S+1005)=5307$\n\
Result: $S = 2151$\n\
Based on the context, we know that $S$ is the sum of the odd-numbered terms in the sequence. Therefore, the answer is 2151\n\
"

prompt_5 = "Problem: A function $f$ has the property that $f(3x-1)=x^2+x+1$ for all real numbers $x$.  What is $f(5)$?\n\
Solution: Let's think step by step. To answer this question, first we need to find out the value of $x$ that corresponds to $f(5)$ using the given property of the function.\n\
To solve this sub-problem, which tool can we use?\n\
Tool (choice: calculator, equation solver, do not use tool): equation solver\n\
Unknown variable: x\n\
Equation: $3x-1=5$\n\
Result: $x=2$\n\
Based on the context, substituting $x=2$, we get:$f(3(2)-1)=2^2+2+1$\n\
To solve this sub-problem, which tool can we use?\n\
Tool (choice: calculator, equation solver, do not use tool): calculator\n\
Equation: $2^2+2+1$\n\
Result: $2^2+2+1 = 7$\n\
Based on the context, $f(5)=f(3(2)-1)=2^2+2+1=7$. Therefore, the answer is 7.\n\
"


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


def chat_and_reasoning(data, max_hop=5):
    def clean_response(response):
        stop_word_list = ['To solve this sub-problem,', 'Tool (choice: ', 'Based on the context,', 'Result: ', 'Equation: ', 'Unknown variable: ', '$\n\n', 'Solution: ']
        for sw in stop_word_list:
            if (sw in response):
                response = response.split(sw)[0].strip()
                if (sw.startswith('$')):
                    response = response + '$'
        return response
    
    def get_response(stop_word):
        nonlocal prompt, decompose_information
        response = call_chat_completion(prompt + decompose_information, stop_word=stop_word)
        response = clean_response(response)
        decompose_information = decompose_information + response
        decompose_information = decompose_information.strip() + '\n'
        return response
    
    stop_word = ['To solve this sub-problem,', 'Tool (choice: ', 'Based on the context,']
    # data['problem'] = "If $\\sqrt[3]{4x^2}=4$, find all possible values of $x$ and list them from least to greatest."

    prompt = prompt_1 + prompt_2 + prompt_3 + prompt_4 + prompt_5 + 'Problem: {}\n'.format(data['problem'])
    # prompt = prompt_1 + prompt_2 + 'Problem: {}\n'.format(data['problem'])
    decompose_information = ''
    use_tool = False
    answer = None
    for i in range(max_hop):
        # Step 1: decompose sub-problem
        if (i == 0):
            reason_template = "Solution: Let's think step by step. To answer this question, first we need to find out "
        else:
            reason_template = "Based on the context, "
        decompose_information = decompose_information + reason_template
        response = get_response(stop_word=stop_word)
        print([response], file=logger_file)
        
        if ('the answer is' in response): 
            decompose_information = decompose_information[:decompose_information.rfind(reason_template)]
            answer = response.split('the answer is')[-1].strip()
            break

        # Step 2: Use tool
        tool_template = "To solve this sub-problem, which tool can we use?\nTool (choice: calculator, equation solver, do not use tool): "
        decompose_information = decompose_information + tool_template
        response = get_response(stop_word=['\n'])
        print([response], file=logger_file)
        tool_response = tool_template + response
        tool_response = tool_response.strip() + '\n'
        
        success_use = False
        if ('calculator' in response):
            calculator_template = "Equation: "
            decompose_information = decompose_information + calculator_template
            response = get_response(stop_word=['\n'])
            print([response], file=logger_file)

            calculator_pattern = '\$.+?\$'
            all_equations = re.findall(calculator_pattern, response)
            calc_result = []
            for equ in all_equations:
                if (len(equ) > 2 and equ[0] == '$' and equ[-1] == '$'):
                    equ = equ[1:-1]
                equ = equ.replace('\\%', '/ 100')
                try:
                    sympy_equ = parse_latex(equ)
                    simplify_equ = simplify(sympy_equ)
                    number_result = sympy.N(simplify_equ)
                except:
                    continue
                try:
                    number_result = float(number_result)
                    calc_result.append('${} = {} = {}$'.format(equ, simplify_equ, str(number_result)))
                except:
                    calc_result.append('${} = {}$'.format(equ, simplify_equ))
            if (len(calc_result) > 0):
                decompose_information = decompose_information + 'Results: {}'.format(', '.join(calc_result)) + '\n'
                print('Result: {}'.format(', '.join(calc_result)), file=logger_file)
                use_tool = True
                success_use = True

        elif ('equation solver' in response):
            variable_template = "Unknown variable: "
            decompose_information = decompose_information + variable_template
            raw_unknown_variable = get_response(stop_word=['\n'])
            raw_unknown_variable = [x.strip() for x in raw_unknown_variable.split(',')]
            unknown_variable = []
            for var in raw_unknown_variable:
                if (len(var) > 2 and var[0] == '$' and var[-1] == '$'):
                    var = var[1:-1]
                if (var not in unknown_variable):
                    unknown_variable.append(var)
            print(unknown_variable, file=logger_file)

            def find_var(equ):
                var_pattern = '[a-zA-Z]+'
                unfind_vars = re.findall(var_pattern, equ)
                print('Unfind Vars: {}'.format(unfind_vars), file=logger_file)
                return unfind_vars

            equation_template = "Equation: "
            decompose_information = decompose_information + equation_template
            equation = get_response(stop_word=['\n'])
            equ_pattern = '\$.+?\$'
            equation = re.findall(equ_pattern, equation)
            print(equation, file=logger_file)
            sympy_equ = []
            func2var = None
            for equ in equation:
                try:
                    if (len(equ) > 2 and equ[0] == '$', equ[-1] == '$'):
                        equ = equ[1:-1]
                    if ('=' in equ):
                        splited_equ = equ.split('=')
                        equ = '{} - ({})'.format(splited_equ[0], splited_equ[-1])
                    equ = equ.replace('\\%', '/ 100')
                    if (len(unknown_variable) == 1 and f'f({unknown_variable[0]})' in equ):
                        if (func2var is not None):
                            equ = equ.replace(f'f({unknown_variable[0]})', func2var)
                        else:
                            for i in range(24, 26):
                                if (chr(97 + i) not in unknown_variable):
                                    equ = equ.replace(f'f({unknown_variable[0]})', chr(97 + i))
                                    func2var = chr(97 + i)
                                    break
                    unfind_vars = find_var(equ)
                    for uv in unfind_vars:
                        if (uv not in unknown_variable and equ.find('\\' + uv) == -1):
                            unknown_variable.append(uv)
                    equ = parse_latex(equ)
                    print(equ, file=logger_file)
                    sympy_equ.append(equ)
                except:
                    continue
            
            variable = []
            print(unknown_variable, file=logger_file)
            for var in unknown_variable:
                variable.append(Symbol(var))
            try:
                raw_results = solve(sympy_equ, variable)
                print(raw_results, file=logger_file)
                equ_result = []
                for var, val in raw_results.items():
                    equ_result.append('${} = {}$'.format(var, val))
                if (func2var is not None):
                    equ_result.append('${} = {}$'.format(func2var, f'f({unknown_variable[0]})'))
                decompose_information = decompose_information + 'Results: {}'.format(', '.join(equ_result)) + '\n'
                print('Result: {}'.format(', '.join(equ_result)), file=logger_file)
                use_tool = True
                success_use = True
            except:
                pass
        
        if (success_use == True):
            decompose_information = decompose_information.replace(tool_response, '')
        else:
            decompose_information = decompose_information[:decompose_information.rfind(tool_response)]

    decompose_information = decompose_information.strip() + '\n'
    decompose_information = decompose_information.replace('\n\n', '\n')
    # if (use_tool == False):
    #     print('\n', file=logger_file)
    #     print("No information because don't use tool", file=logger_file)
    #     decompose_information = "Solution: Let's think step by step. "
    # else:
    print('\n', file=logger_file)
    print(decompose_information, file=logger_file)
    print('----------------------------------------', file=logger_file)
    if (answer is None):
        answer = get_answer_boxed(decompose_information)
    return decompose_information, answer


def main(args):
    global logger_file
    if (args.logger_path is not None):
        logger_file = open(args.logger_path, 'w')
    else:
        logger_file = sys.stdout

    fout = open(args.result_path, args.write_mode)
    data_processer = DATA_PROCESSER[args.dataset_name](args.demo_path, args.num_examplar)
    
    with open('dataset/math/test_retrieval-all.json', 'r') as fin:
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
        # print(data['problem'], file=logger_file)
        if (args.use_decompose == True):
            decompose_information, llm_step = chat_and_reasoning(data)
        else:
            decompose_information, llm_step = '', None

        for i in range(args.num_examplar):
            prompt, real_label = data_processer.process_retrieval(data, i, data['retrieval_result'])
            if (len(prompt) < int(4096 * 1.5)):
                break
        prompt = prompt + decompose_information
        if (llm_step is None):
            llm_step = call_chat_completion(prompt)
        else:
            llm_step = 'The answer is ' + llm_step
        
        data['chat_and_reason'] = decompose_information
        data['llm_step'] = llm_step
        if ('The answer is' in llm_step):
            pred = llm_step.split('The answer is')[-1].strip()
        else:
            pred = get_answer_boxed(llm_step)
        if (pred is None or len(pred) == 0):
            prompt = prompt + decompose_information + llm_step + ' The answer is '
            pred = call_chat_completion(prompt)

        if (len(pred) >= 1 and pred[-1] == '.'):
            pred = pred[:-1]
        if (pred.rfind('$') != -1):
            if (pred[pred.rfind('$') + 1:].isalpha() == True):
                pred = pred[:pred.rfind('$') + 1]
        data['prompt'] = prompt
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
