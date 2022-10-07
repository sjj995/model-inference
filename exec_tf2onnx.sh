#!/bin/bash
src_dir=$1
tensorflow_model_dir=$2
target_dir=$3
onnx_model_name=$4


display_error(){
	echo "Refer to Help and Enter it correctly"
}

display_usage(){
	echo -e "\ndiscription exec_tf2onnx.sh will run tensorflow-onnx tf2onnx convert module\n"
	echo -e "usage : 추후에 변경 <-  파이썬으로 한 번에 실행시킬 거임\n"
	echo -e "optional arguments : \n"
}

# src_dir -> source directory
# target_dir -> where onnx model downloaded

if [[ $1 == "--help" ]] || [[ $1 == "-h" ]];then
	display_usage
elif [ $# -le 3 ];then
	display_error
else
	python3 -m tf2onnx.convert --saved-model $tensorflow_model_dir --output $target_dir/$onnx_model_name 2>&1 | tee $src_dir/tf2onnx_temp.txt
fi

