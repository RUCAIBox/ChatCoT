echo "CP"
echo "CoT"
python evaluate.py --result_path result/math_cp/cot/turbo-cot-5shot.json
echo "CoT + SC"
python evaluate.py --result_path result/math_cp/cot_w_sc/turbo-w_sc-5shot.json
echo "ChatCoT"
python evaluate.py --result_path result/math_cp/chatcot/turbo-chatcot-5shot.json
echo "ChatCoT + SC"
python evaluate.py --result_path result/math_cp/chatcot_w_sc/turbo-w_sc-5shot.json

echo "NT"
echo "CoT"
python evaluate.py --result_path result/math_nt/cot/turbo-cot-5shot.json
echo "CoT + SC"
python evaluate.py --result_path result/math_nt/cot_w_sc/turbo-w_sc-5shot.json
echo "ChatCoT"
python evaluate.py --result_path result/math_nt/chatcot/turbo-chatcot-5shot.json
echo "ChatCoT + SC"
python evaluate.py --result_path result/math_nt/chatcot_w_sc/turbo-w_sc-5shot.json