import subprocess
import tempfile
import sys
import os


def process_tf2onnx(base_path, saved_model_path, onnx_model_path, onnx_model_name):
    command_path = base_path+'/exec_tf2onnx.sh'

    command = ['bash', command_path, base_path,
               saved_model_path, onnx_model_path, onnx_model_name]

    subprocess.run(command, stdout=subprocess.PIPE)

# 이거 수정해야될 수 있음
def get_tensorflow_files(root_path):
    tf_list = []
    for (root, dirs, files) in os.walk(root_path):
        for file_name in files:
            if file_name.endswith(".pb"):
                tf_list.append(root)
                break
    return tf_list



if __name__ == "__main__":
    pass
