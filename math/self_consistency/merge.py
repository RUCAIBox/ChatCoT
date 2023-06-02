import os
import json
import argparse

def load_multi_line_json(f):
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
    full_data = []
    results = os.listdir(args.result_folder)
    for r in results:
        file_path = os.path.join(args.result_folder, r)
        with open(file_path, 'r') as fin:
            data = load_multi_line_json(fin)
        full_data = full_data + data
    with open(args.target_path, 'w') as fout:
        for data in full_data:
            fout.write(json.dumps(data, indent=4, ensure_ascii=False) + '\n')

if __name__ == '__main__':
    parser = argparse.ArgumentParser()

    parser.add_argument('--result_folder', type=str, default=None, help='The path to the result folder.')
    parser.add_argument('--target_path', type=str, default=None, help='The path to the target path.')
    
    args = parser.parse_args()

    main(args)