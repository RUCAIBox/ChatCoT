import json

from simcse import SimCSE
from datasets import Dataset
from tqdm import tqdm


def get_similar_content(model, sentences_a, data):
    sentences_b = [' '.join(d) for d in data['context']['sentences']]
    similarities = model.similarity(sentences_a, sentences_b)
    score = []
    for i in range(len(sentences_b)):
        score.append((similarities[i], i))
    score = sorted(score, key=lambda x: x[0], reverse=True)
    print(score)
    return score, sentences_b

if __name__ == '__main__':
    with open('demo/hotpot_qa.json', 'r') as fin:
        test_data = json.load(fin)
    test_data = [test_data[4]]

    # model_path = '/mnt/chenzhipeng/llm_data/model/roberta-base-simcse-math'
    model_path = 'princeton-nlp/unsup-simcse-roberta-base'

    model = SimCSE(model_path)
    print(model)

    get_similar_content(model, test_data)