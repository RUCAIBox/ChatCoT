import re
import json


class DataProcessForMATH:
    def __init__(self, data_path, is_clean=False, num_examplar=5):
        self.num_examplar = num_examplar
        self.data_path = data_path
        with open(self.data_path, 'r') as fin:
            raw_examplars = json.load(fin)

        self.examplars = []
        for examplar in raw_examplars:
            # if examplar['knowledge_point'] != 'algebra':
            #     continue
            self.examplars.append(self.process_single_data(examplar))

    def process(
            self,
            data,
            num_del_examplar=0,
            pattern_demo="Problem: {}\nSolution: Let's think step by step. {} The answer is {}\n",
            pattern_test="Problem: {}\nSolution: Let's think step by step.",
    ):
        processed_data = self.process_single_data(data)

        prompt = ''
        num_example = self.num_examplar - num_del_examplar
        for i in range(num_example):
            problem = self.examplars[i]['problem']
            solution = self.examplars[i]['cot']
            answer = self.examplars[i]['answer']
            prompt = prompt + pattern_demo.format(problem, solution, answer)
        prompt = prompt + pattern_test.format(processed_data['problem'])

        return prompt, processed_data['answer']
    
    def process_retrieval(
            self,
            data,
            num_del_examplar,
            ret_examplar,
            pattern_demo="Problem: {}\nSolution: Let's think step by step. {} The answer is {}\n",
            pattern_test="Problem: {}\nSolution: Let's think step by step.",
    ):
        processed_data = self.process_single_data(data)

        prompt = ''
        num_example = self.num_examplar - num_del_examplar
        for i in range(num_example):
            examplar = self.process_single_data(ret_examplar[i])
            problem = examplar['problem']
            solution = examplar['cot']
            answer = examplar['answer']
            prompt = prompt + pattern_demo.format(problem, solution, answer)
        prompt = prompt + pattern_test.format(processed_data['problem'])

        return prompt, processed_data['answer']

    def process_classifier(
            self,
            data,
            num_del_examplar=0,
            pattern_demo="Problem: {}\nProblem type: {}\n",
            pattern_test="Problem: {}\nProblem type: ",
    ):
        processed_data = self.process_single_data(data)

        prompt = 'You should classify the problem into counting_and_probability, geometry, intermediate_algebra, prealgebra, precalculus, algebra, number_theory\n'
        num_example = self.num_examplar - num_del_examplar
        for i in range(num_example):
            problem = self.examplars[i]['problem']
            prob_type = self.examplars[i]['knowledge_point']
            prompt = prompt + pattern_demo.format(problem, prob_type)
        prompt = prompt + pattern_test.format(processed_data['problem'])

        return prompt, processed_data['knowledge_point']

    def get_answer(self, content):
        pattern = '\\boxed'
        start_pos = content.rfind(pattern)
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


    def process_single_data(self, data):
        processed_data = {
                'problem': data['problem'],
                'cot': data['solution'],
                'answer': self.get_answer(data['solution']),
        }
        if ('knowledge_point' in data):
            processed_data['knowledge_point'] = data['knowledge_point']
        return processed_data


