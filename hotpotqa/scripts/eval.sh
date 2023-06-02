echo "HotpotQA (Distractor)"
echo "CoT"
python evaluate.py --result_path result/turbo-baseline-4shot.json
echo "CoT w/ Tool"
python evaluate.py --result_path result/turbo-tool-4shot.json
echo "ChatCoT w/o Feedback"
python evaluate.py --result_path result/turbo-chatcot_wo_feedback-4shot.json
echo "ChatCoT"
python evaluate.py --result_path result/turbo-chatcot-4shot.json
