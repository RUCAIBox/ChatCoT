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
from datasets import load_dataset, Dataset

import sympy
from sympy import simplify, Symbol, solve
from sympy.parsing.latex import parse_latex

from data_process import (
    DataProcessForMATH,
)

# api_key_list for multiprocess
api_key_list = [
    '',
    YOUR_API_KEY_1,
    YOUR_API_KEY_2,
    YOUR_API_KEY_3,
    YOUR_API_KEY_4,
    YOUR_API_KEY_5,
    YOUR_API_KEY_6,
]

DATA_PROCESSER = {
    'math': DataProcessForMATH,
}

bootstrap = {
    "role": "system",
    "content": "You are an expert in mathematical problem. You should follow the example and answer the last question.",
}

retrieval_demos = None

#########################    Demonstration    #########################

use_tool = {
    "role": "user",
    "content": "To solve this sub-problem, which tool can we use? You can choose calculator, equation solver or do not use tool"
}
continue_reasoning = {
    "role": "user",
    "content": "Continue reasoning"
}
final_get_answer = {
    "role": "user",
    "content": "Base on the context, what is the answer?"
}
EquationSolver_UnknownVariable = {
    "role": "user",
    "content": "Give me the unknown variable of the equation system in latex style and separated by comma"
}
EquationSolver_EquationSystem = {
    "role": "user",
    "content": "Give me the equation system in latex style and separated by comma"
}
Calculator_Equation = {
    "role": "user",
    "content": "Give me the equation to calculate in latex style and separated by comma"
}

prompt_1_chat = [
    {
        "role": "user", 
        "content": "Problem: Ramanujan and Hardy played a game where they both picked a complex number. If the product of their numbers was $32-8i$, and Hardy picked $5+3i$, what number did Ramanujan pick?\nLet's think step by step to solve this problem"
    },
    {
        "role": "assistant", 
        "content": "First, we need to find out how to use Hardy's number and the product to find Ramanujan's number"
    },
    use_tool,
    {
        "role": "assistant",
        "content": "Do not use tool"
    },
    continue_reasoning,
    {
        "role": "assistant",
        "content": "Second, we can start by setting Ramanujan's number as $a+bi$ and then use the fact that the product of their numbers is $32-8i$ to set up an equation. We have:\n\n$(5+3i)(a+bi) = 32-8i$\n\nExpanding the left side, we get:\n\n$(5a-3b) + (3a+5b)i = 32-8i$\n\nEquating the real and imaginary parts, we get the system of equations:\n\n$5a-3b = 32$\n$3a+5b = -8$"
    },
    use_tool,
    {
        "role": "assistant",
        "content": "Equation solver"
    },
    EquationSolver_UnknownVariable,
    {
        "role": "assistant",
        "content": "$a$, $b$"
    },
    EquationSolver_EquationSystem,
    {
        "role": "assistant",
        "content": "$5*a-3*b = 32$, $3*a+5*b = -8$"
    },
    {
        "role": "user",
        "content": "Results: $a = 4$, $b = -4$\nContinue reasoning"
    },
    {
        "role": "assistant",
        "content": "Finally, we can conclude that Ramanujan picked the complex number $4-4i$. Therefore, the answer is 4-4i"
    }
]

prompt_2_chat = [
    {
        "role": "user", 
        "content": "Problem: Simplify\n\\[\\frac{1}{\\log_{15} 2 + 1} + \\frac{1}{\\log_{10} 3 + 1} + \\frac{1}{\\log_{6} 5 + 1}.\\]\nLet's think step by step to solve this problem"
    },
    {
        "role": "assistant", 
        "content": "First, we need to find out the values of $\\frac{1}{\\log_{15} 2 + 1} + \\frac{1}{\\log_{10} 3 + 1} + \\frac{1}{\\log_6 5 + 1}$"
    },
    use_tool,
    {
        "role": "assistant",
        "content": "Calculator"
    },
    Calculator_Equation,
    {
        "role": "assistant",
        "content": "$\\frac{1}{\\log_{15} 2 + 1} + \\frac{1}{\\log_{10} 3 + 1} + \\frac{1}{\\log_{6} 5 + 1}$"
    },
    {
        "role": "user",
        "content": "Results: $\\frac{1}{\\log_{15} 2 + 1} + \\frac{1}{\\log_{10} 3 + 1} + \\frac{1}{\\log_{6} 5 + 1} = 2$\nContinue reasoning"
    },
    {
        "role": "assistant",
        "content": "Finally, we can simplify the expression to just 2. Therefore, the answer is 2"
    },
]

prompt_3_chat = [
    {
        "role": "user", 
        "content": "Problem: A parabola with equation $y=x^2+bx+c$ passes through the points $(-1,-11)$ and $(3,17)$. What is $c$?\nLet's think step by step and use knowledge in similar problem to solve this problem"
    },
    {
        "role": "assistant", 
        "content": "First, we need to find out the values of $c$ using the given points and the equation of the parabola"
    },
    use_tool,
    {
        "role": "assistant",
        "content": "Equation solver"
    },
    EquationSolver_UnknownVariable,
    {
        "role": "assistant",
        "content": "$b$, $c$"
    },
    EquationSolver_EquationSystem,
    {
        "role": "assistant",
        "content": "$(-1)^2 - b + c = -11$, $(3)^2 + 3b + c = 17$"
    },
    {
        "role": "user",
        "content": "Results: $b = 3$, $c = -7$\nContinue reasoning"
    },
    {
        "role": "assistant",
        "content": "Finally, we can conclude that $c=-7$. Therefore, the answer is \\boxed{-7}"
    },
]

prompt_4_chat = [
    {
        "role": "user", 
        "content": "Problem: A sequence consists of $2010$ terms.  Each term after the first is 1 larger than the previous term.  The sum of the $2010$ terms is $5307$.  When every second term is added up, starting with the first term and ending with the second last term, what is the sum?\nLet's think step by step and use knowledge in similar problem to solve this problem"
    },
    {
        "role": "assistant", 
        "content": "First, we need to find out the relation between each odd-numbered term with the following even-numbered term"
    },
    use_tool,
    {
        "role": "assistant",
        "content": "Do not use tool"
    },
    continue_reasoning,
    {
        "role": "assistant", 
        "content": "Second, we pair the first term with the second, the third term with the fourth, and so on, until we pair the 2009th term with the 2010th term. There are 1005 such pairs."
    },
    use_tool,
    {
        "role": "assistant",
        "content": "Do not use tool"
    },
    continue_reasoning,
    {
        "role": "assistant", 
        "content": "Third, we suppose that $S$ is the sum of the odd-numbered terms in the sequence. In each pair, the even-numbered term is one bigger than the odd-numbered term. That is, $x_2-x_1=1$, $x_4-x_3=1$, and so on.  Therefore, the sum of the even-numbered terms is 1005 greater than the sum of the odd-numbered terms. Thus, the sum of the even-numbered terms is $S+1005$.\n\nSince the sum of all of the terms equals the sum of the odd-numbered terms plus the sum of the even-numbered terms, then $S+(S+1005)=5307$."
    },
    use_tool,
    {
        "role": "assistant",
        "content": "Equation solver"
    },
    EquationSolver_UnknownVariable,
    {
        "role": "assistant",
        "content": "$S$"
    },
    EquationSolver_EquationSystem,
    {
        "role": "assistant",
        "content": "$S+(S+1005)=5307$"
    },
    {
        "role": "user",
        "content": "Results: $S = 2151$\nContinue reasoning"
    },
    {
        "role": "assistant",
        "content": "Finally, we know that $S$ is the sum of the odd-numbered terms in the sequence. Therefore, the answer is \\boxed{2151}"
    },
]

prompt_5_chat = [
    {
        "role": "user", 
        "content": "Problem: A function $f$ has the property that $f(3x-1)=x^2+x+1$ for all real numbers $x$.  What is $f(5)$?\nLet's think step by step and use knowledge in similar problem to solve this problem"
    },
    {
        "role": "assistant", 
        "content": "First, we need to find out the value of $x$ that corresponds to $f(5)$ using the given property of the function."
    },
    use_tool,
    {
        "role": "assistant",
        "content": "Equation solver"
    },
    EquationSolver_UnknownVariable,
    {
        "role": "assistant",
        "content": "$x$"
    },
    EquationSolver_EquationSystem,
    {
        "role": "assistant",
        "content": "$3x-1=5$"
    },
    {
        "role": "user",
        "content": "Results: $x = 2$\nContinue reasoning"
    },
    {
        "role": "assistant",
        "content": "Second, substituting $x=2$, we get: $f(3(2)-1)=2^2+2+1$"
    },
    use_tool,
    {
        "role": "assistant",
        "content": "Calculator"
    },
    Calculator_Equation,
    {
        "role": "assistant",
        "content": "$2^2+2+1$"
    },
    {
        "role": "user",
        "content": "Results: $2^2+2+1 = 7$\nContinue reasoning"
    },
    {
        "role": "assistant",
        "content": "Finally, $f(5)=f(3(2)-1)=2^2+2+1=7$. Therefore, the answer is \\boxed{7}"
    }
]

begin_prompt = {
    "user_begin": "You should solve the problem step by step and you should follow the react in the history. In each reasoning step, you can use tool (calculator or equation solver) to help you solve problem. Do you understand?",
    "assistant_begin": "Yes, I understand. I will follow my response in the conversation history and solve the problem step by step.",
    "user_retrieval": "I give you some similar problems.\n{}You can use the knowledge and thoery in these problem. Do you understand?",
    "assistant_retrieval": "Yes, I understand. I will solve the problem step by step and use tool to help me.",
    "user_intr_tool": "You can use tool to help you solve the problem and I give you the instruction of tools usage. Calculator can help you simplify the equation and calculate the value of equation, and equation solver can help you solve the value of unknown variable. Do you understand?",
    "assistant_intr_tool": "Yes, I understand. I will use tool to help me solve the problem.",
}

start_prompt = None


def call_chat_completion(messages, num_retri, stop_word='Problem: '):
    global retrieval_demos
    del_prompt = num_retri
    len_prompt = [len(prompt_1_chat), len(prompt_2_chat), len(prompt_3_chat), len(prompt_4_chat), len(prompt_5_chat)]
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
        except openai.error.InvalidRequestError:
            if (del_prompt < 5):
                messages = start_prompt + messages[len(start_prompt) + len_prompt[del_prompt]:]
                del_prompt += 1
            else:
                # if (len(messages) < 7):
                #     retrieval_demos = retrieval_demos.split('Problem: ')[:-1]
                #     retrieval_demos = [demo.strip() for demo in retrieval_demos]
                #     retrieval_demos = '\nProblem: '.join(retrieval_demos)
                #     retrieval_demos = retrieval_demos.strip()
                #     start_prompt[3] = {"role": "user", "content": begin_prompt["user_retrieval"].format(retrieval_demos)}
                #     messages = start_prompt + messages[len(start_prompt):]
                # else:
                messages = start_prompt + messages[7:]
                
        except openai.error.RateLimitError:
            time.sleep(20)
        except:
            time.sleep(10)

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


def my_solve(expr, var):
    result = solve(expr, var)
    return result

def solve_with_timeout(expr, var, timeout=60):
    pool = multiprocessing.Pool(processes=1)
    async_result = pool.apply_async(my_solve, (expr, var,))
    try:
        result = async_result.get(timeout)
    except multiprocessing.TimeoutError:
        # print('TimeoutError')
        result = None
    return result


def my_N(expr):
    result = sympy.N(expr)
    return result

def N_with_timeout(expr, timeout=60):
    pool = multiprocessing.Pool(processes=1)
    async_result = pool.apply_async(my_N, (expr,))
    try:
        result = async_result.get(timeout)
    except multiprocessing.TimeoutError:
        # print('TimeoutError')
        result = None
    return result


def my_simplify(expr):
    result = sympy.N(expr)
    return result

def simplify_with_timeout(expr, timeout=60):
    pool = multiprocessing.Pool(processes=1)
    async_result = pool.apply_async(my_simplify, (expr,))
    try:
        result = async_result.get(timeout)
    except multiprocessing.TimeoutError:
        # print('TimeoutError')
        result = None
    return result


def chat_and_reasoning(data, num_retri, max_hop=8):
    def clean_response(response):
        stop_word_list = ['To solve this sub-problem,', 'Tool (choice: ', 'Based on the context,', 'Result: ', 'Equation: ', 'Unknown variable: ', 'Solution: ']
        for sw in stop_word_list:
            if (sw in response):
                response = response.split(sw)[0].strip()
                if (sw.startswith('$')):
                    response = response + '$'
        return response
    
    def get_response(stop_word):
        nonlocal messages
        response = call_chat_completion(messages, num_retri, stop_word=stop_word)
        response = clean_response(response)
        # print("response: {}".format(response))
        messages.append({"role": "assistant", "content": response})
        return response
    
    def get_similar_problem(similar_problem):
        retrieval_demo = ""
        for i in range(num_retri):
            retrieval_demo = retrieval_demo + "Problem: {}\nSolution: Let's think step by step. {} The answer is {}\n".format(
                similar_problem[i]['problem'],
                similar_problem[i]['solution'],
                get_answer_boxed(similar_problem[i]['solution']),
            )
            if (len(retrieval_demo) > 4000): break
        return retrieval_demo
    
    stop_word = ['To solve this sub-problem,', 'Tool (choice: ', 'Based on the context,']

    global start_prompt, retrieval_demos
    retrieval_demos = get_similar_problem(data['retrieval_result'])
    start_prompt = [
        bootstrap,
        # {"role": "user", "content": begin_prompt["user_intr_tool"]},
        # {"role": "assistant", "content": begin_prompt["assistant_intr_tool"]},
        {"role": "user", "content": begin_prompt["user_retrieval"].format(retrieval_demos)},
        {"role": "assistant", "content": begin_prompt["assistant_retrieval"]},
    ]
    messages = start_prompt
    annotated_examplars = [prompt_1_chat, prompt_2_chat, prompt_3_chat, prompt_4_chat, prompt_5_chat]
    len_annoated = 0
    for i in range(num_retri, 5):
        messages = messages + annotated_examplars[i]
        len_annoated = len_annoated + len(annotated_examplars[i])
    messages = messages + [
        {"role": "user", "content": begin_prompt["user_begin"]},
        {"role": "assistant", "content": begin_prompt["assistant_begin"]},
        {"role": "user", "content": "Problem: {}\nLet's think step by step and use knowledge in similar problem to solve this problem".format(data['problem'])},
    ]

    answer = None
    for i in range(max_hop):
        # Step 1: decompose sub-problem
        response = get_response(stop_word=stop_word)
        
        if ('the answer is' in response): 
            answer = response.split('the answer is')[-1].strip()
            break
        
        if ('\\boxed' in response):
            answer = get_answer_boxed(response)
            break

        # Step 2: Use tool
        messages = messages + [use_tool]
        response = get_response(stop_word=['\n'])
        
        if ('calculator' in response.lower()):
            messages = messages + [Calculator_Equation]
            raw_all_equations = get_response(stop_word=['\n'])
            # print(raw_all_equations)

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
                # print(equ)
                try:
                    sympy_equ = parse_latex(equ)
                    simplify_equ = simplify_with_timeout(sympy_equ)
                    number_result = N_with_timeout(simplify_equ)
                except:
                    continue
                # print(number_result)
                try:
                    number_result = float(number_result)
                    if (math.isinf(number_result) == True):
                        calc_result.append('the result of {} is too large.'.format(equ))
                    else:
                        calc_result.append('${} = {} = {}$'.format(equ, simplify_equ, number_result))
                except:
                    calc_result.append('${} = {}$'.format(equ, simplify_equ))
            if (len(calc_result) > 0):
                results = 'Results: {}'.format(', '.join(calc_result)) + '\nContinue reasoning'
                # print(results)
                messages.append({"role": "user", "content": results})
            else:
                messages = messages[:-4] + [continue_reasoning]

        elif ('equation solver' in response.lower()):
            messages = messages + [EquationSolver_UnknownVariable]
            raw_unknown_variable = get_response(stop_word=['\n'])
            unknown_variable = []
            for var in raw_unknown_variable.split(','):
                var = var.strip()
                if (len(var) > 2 and var[0] == '$' and var[-1] == '$'):
                    var = var[1:-1]
                unknown_variable.append(var)
            # print(unknown_variable)

            messages = messages + [EquationSolver_EquationSystem]
            raw_equation_system = get_response(stop_word=['\n'])
            equation_system = [var.strip() for var in raw_equation_system.split(',')]
            # print(equation_system)

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
                    equ = simplify_with_timeout(equ)
                    sympy_equ.append(equ)
                except:
                    continue
            
            variable = []
            for var in unknown_variable:
                variable.append(Symbol(var))

            # print(variable)
            # print(sympy_equ)

            try:
                # print('test 1')
                raw_results = solve_with_timeout(sympy_equ, variable)
                # print(raw_results)
                # print('In first try')
                equ_result = []
                for var, val in raw_results.items():
                    equ_result.append('${} = {}$'.format(var, val))
                
                results = 'Results: {}'.format(', '.join(equ_result)) + '\nContinue reasoning'
                messages.append({"role": "user", "content": results})
            except:
                try:
                    # print('test 2')
                    raw_results = list(solve_with_timeout(sympy_equ, variable))
                    # print(raw_results)
                    # print('In second try')
                    equ_result = []
                    for result in raw_results:
                        for var, val in zip(unknown_variable, result):
                            equ_result.append('${} = {}$'.format(var, val))
                    
                    results = 'Results: {}'.format(', '.join(equ_result)) + '\nContinue reasoning'
                    # print(results)
                    # print('---------------------------------')
                    messages.append({"role": "user", "content": results})
                except:
                    messages = messages[:-6] + [continue_reasoning]
        
        else:
            messages = messages + [continue_reasoning]
        
        # print('end')

    if (answer is None):
        messages = messages[:-1] + [final_get_answer]
        answer = get_response(stop_word)
    return messages[len(start_prompt) + len_annoated:], answer


def main(args):
    openai.api_key = api_key_list[args.api_key_idx]

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
        if (step < args.start_prob or step > args.end_prob): continue
        data['problem'] = clean_problem(data['problem'])
        conversation, pred = chat_and_reasoning(data, args.num_retri)

        data['chat_and_reason'] = conversation
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

    fout.close()


if __name__ == '__main__':
    parser = argparse.ArgumentParser()

    parser.add_argument('--write_mode', type=str, default='a', help='The mode to write result file')
    parser.add_argument('--result_path', type=str, help='The path to save result')
    parser.add_argument('--dataset_name', type=str, help='The name of dataset')
    parser.add_argument('--num_examplar', type=int, default=5, help='The number of examplar in prompt')
    parser.add_argument('--demo_path', type=str, help='The path to the demos')
    parser.add_argument('--data_split', type=str, default='test', help='The split of the dataset')
    parser.add_argument('--use_decompose', type=bool, default=False, help='Whether to use decompose and tool.')
    parser.add_argument('--num_retri', type=int, default=None, help='The number of retrieval examplars.')
    parser.add_argument('--api_key_idx', type=int, default=None, help='The API key index of openai')
    parser.add_argument('--start_prob', type=int, default=None, help='Start problem index')
    parser.add_argument('--end_prob', type=int, default=None, help='End problem index')
    
    args = parser.parse_args()

    main(args)
