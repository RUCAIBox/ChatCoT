import re
import json
import numpy as np

class DataProcessForHotpotQA:
    def __init__(self, data_path, is_clean=False, num_examplar=5, is_ground_truth=False):
        self.num_examplar = num_examplar
        self.is_clean = is_clean
        self.data_path = data_path
        self.is_ground_truth = is_ground_truth
        with open(self.data_path, 'r') as fin:
            raw_examplars = json.load(fin)
        
        self.examplars = []
        for examplar in raw_examplars:
            self.examplars.append(self.process_single_data(examplar, is_ground_truth))

    def process(
            self, 
            data,
            num_del_examplar=0,
            pattern_demo="Problem: {}\nEvidence: {}\nSolution: Let's think step by step. {} The answer is {}\n",
            pattern_test="Problem: {}\nEvidence: {}\nSolution: Let's think step by step.",
    ):
        processed_data = self.process_single_data(data, self.is_ground_truth)
        
        prompt = ''
        num_example = self.num_examplar - num_del_examplar
        for i in range(num_example):
            problem = self.examplars[i]['problem']
            solution = self.examplars[i]['cot']
            answer = self.examplars[i]['answer']
            evidence = self.examplars[i]['evidence']
            prompt = prompt + pattern_demo.format(problem, evidence, solution, answer)
        prompt = prompt + pattern_test.format(processed_data['problem'], processed_data['evidence'])

        return prompt, processed_data['answer']

    def process_single_data(self, data, is_ground_truth):
        processed_data = {
            'problem': data['question'],
            'answer': data['answer'],
            'evidence': '\n'.join(data['evidence']),
        }
        if ('cot' in data):
            processed_data['cot'] = data['cot']

        if (is_ground_truth == True):
            processed_data['evidence'] = []
            titles = np.unique(data['supporting_facts']['title'])
            for t in titles:
                idx = data['context']['title'].index(t)
                processed_data['evidence'].append(''.join(data['context']['sentences'][idx]))

        return processed_data

if __name__ == '__main__':
    data_processer = DataProcessForMATH('demo/math.json', is_clean=True)
