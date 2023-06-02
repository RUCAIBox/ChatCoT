echo "PC"
echo "ChatCoT"
python evaluate.py --result_path result/math_pc/chatcot/turbo-chatcot-5shot.json
echo "ChatCoT w/o TK"
python evaluate.py --result_path result/math_pc/wo_tk/turbo-wo_tk-5shot.json
echo "ChatCoT w/o RATK"
python evaluate.py --result_path result/math_pc/wo_ratk/turbo-wo_ratk-5shot.json
echo "ChatCoT w/o MRF"
python evaluate.py --result_path result/math_pc/wo_mrf/turbo-wo_mrf-5shot.json

echo "Geometry"
echo "ChatCoT"
python evaluate.py --result_path result/math_geometry/chatcot/turbo-chatcot-5shot.json
echo "ChatCoT w/o TK"
python evaluate.py --result_path result/math_geometry/wo_tk/turbo-wo_tk-5shot.json
echo "ChatCoT w/o RATK"
python evaluate.py --result_path result/math_geometry/wo_ratk/turbo-wo_ratk-5shot.json
echo "ChatCoT w/o MRF"
python evaluate.py --result_path result/math_geometry/wo_mrf/turbo-wo_mrf-5shot.json

echo "NT"
echo "ChatCoT"
python evaluate.py --result_path result/math_nt/chatcot/turbo-chatcot-5shot.json
echo "ChatCoT w/o TK"
python evaluate.py --result_path result/math_nt/wo_tk/turbo-wo_tk-5shot.json
echo "ChatCoT w/o RATK"
python evaluate.py --result_path result/math_nt/wo_ratk/turbo-wo_ratk-5shot.json
echo "ChatCoT w/o MRF"
python evaluate.py --result_path result/math_nt/wo_mrf/turbo-wo_mrf-5shot.json