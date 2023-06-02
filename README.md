# ChatCoT: Tool-Augmented Chain-of-Thought Reasoning on Chat-based Large Language Models

This repo provides the source code & data of our paper: [ChatCoT: Tool-Augmented Chain-of-Thought Reasoning on Chat-based Large Language Models](https://arxiv.org/abs/2305.14323) (Arxiv 2023).

![./picture/framework.png]()

```
@InProceedings{Chen-ChatCot-2023,
      title = {ChatCoT: Tool-Augmented Chain-of-Thought Reasoning on Chat-based Large Language Models}, 
      author = {Zhipeng Chen and Kun Zhou and Beichen Zhang and Zheng Gong and Wayne Xin Zhao and Ji-Rong Wen},
      year = {2023},
      eprint = {2305.14323},
      archivePrefix = {arXiv},
      primaryClass = {cs.CL}
}
```

## Data & Code

+ `math/`: the code about ChatCoT on MATH dataset
    + `demo/`: the demos used in in-context learning on few-shot setting
    + `math/result/`: the result files of different methods
    + `math/scripts/`: the running scripts of different methods
    + `math/ablation`: the code of ablation study
    + `math/self_consistency`: the code to explore combining ChatCoT with CoT improvement strategies

The `hotpotqa/` folder is similar with `math/` folder

## Usage

### Prepare

You can use following scripts to install related python package through pip:
```
git clone https://github.com/RUCAIBox/ChatCoT.git
cd ChatCoT
pip install -r requirements.txt
```

### Inference

You can run ChatCot on the sub-task of MATH dataset by running `run_turbo_chatcot.sh`:
```
cd math
bash scripts/run_turbo_chatcot.sh
```

You have to replace `YOUR_API_KEY ` with you openai api key in the code. Specially, we run ChatCoT through multi-processing, and you should prepare a list of api key in order to run the code correctly.

### Evaluate

You can evaluate the results by running `eval.sh`:
```
cd math
bash scripts/eval.sh
```

## Results

### Main Results

| Methods      | Algebra   | CP    | PC    | PA    | Geometry  | IA    | NT    |
| :-----:      | :-:       | :-:   | :-:   |:-:    | :-:       | :-:   |:-:    |
| CoT          | 48.10     | 31.43 | 21.06 | 56.60 | 22.34     | 18.27 | 29.07 |
| CoT w/ Tool  | 35.89     | 22.57 |  9.34 | 40.53 | 13.57     |  9.41 | 19.44 |
| CoT w/ Retri | <u>52.74</u> | <u>32.70</u> | <u>18.86</u> | <u>58.44</u> | <u>29.23</u> | **19.93** | <u>31.67</u> |
| ChatCoT      | 56.11     | **34.18** | **23.81** | **59.24** | **29.85** | <u>19.49</u> | **32.59** |

| Methods               | HotpotQA     |
| :-----:               | :-:          |
| CoT                   | 37.99        |
| CoT w/ Tool           | 31.42        |
| ChatCoT w/o Feedback  | <u>53.79</u> |
| ChatCoT               | **59.16**    |

### Ablation Study

| Methods | PC  | Geo | NT  |
| :-----: | :-: | :-: | :-: |
| ChatCoT | **23.81** | **29.85** | **32.59** |
| ChatCoT w/o TK | 23.26 | 29.23 | 30.56 |
| ChatCoT w/o RATK | 19.96 | 27.35 | 30.93 |
| ChatCoT w/o MRF | 21.61 | 24.22 | 32.22 |

The results of ablation study. TK, RATK, and MRF denote if using tool knowledge, retrieval-augmented task knowledge, and multi-turn reasoning format at early turns of the conversation, respectively.


### Combining CoT Improvement Strategies

| Methods | CP  | NT  |
| :-----: | :-: | :-: |
| CoT | 31.43 | 29.07 |
| CoT + SC | 35.23 | 34.44 |
| ChatCoT | 34.18 | 32.59 |
| ChatCoT + SC | **40.08** | **38.33** |