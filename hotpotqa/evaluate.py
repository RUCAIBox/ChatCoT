import re
import json
import argparse

import sympy
from sympy.parsing.latex import parse_latex

from tqdm import tqdm


def load_multi_line_json_v3(f):
    data = ''
    all_data = []
    raw_data =f.readlines()
    for line in raw_data:
        data = data + line
        if (line.startswith('}')):
            all_data.append(json.loads(data))
            data = ''
    return all_data


def main(args):
    with open(args.result_path, 'r') as fin:
        datas = load_multi_line_json_v3(fin)
    
    num_correct = 0
    total_problem = 0
    for data in tqdm(datas):
        if (data['real_answer'].lower() in data['llm_answer'].lower()):
        # if (data['real_answer'].lower() == data['llm_answer'].lower()):
            num_correct = num_correct + 1
        total_problem = total_problem + 1

        # if (total_problem >= 52): break

    print('Accuracy: {} ( {} / {} )'.format(round(num_correct / total_problem * 100, 2), num_correct, total_problem))

if __name__ == '__main__':
    parser = argparse.ArgumentParser()

    parser.add_argument('--result_path', type=str, help='The path to result')
    
    args = parser.parse_args()

    main(args)
