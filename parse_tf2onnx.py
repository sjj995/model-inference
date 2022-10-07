import os


def process_parsing(base_path, tensorflow_model_path, onnx_model_path, onnx_file):
    ErrorCode = ['ValueError', 'RuntimeError',
                 'NotImplementedError', 'TypeError', 'FileNotFoundError']
    temp_file = base_path+'/tf2onnx_temp.txt'
    log_file = os.path.basename(tensorflow_model_path)+'_error.log'

    if os.path.isfile(onnx_model_path+'/'+onnx_file):
        print('\033[96m'+f'[COMPLETE] Onnx file has been created successfully'+'\033[0m')
    else:
        with open(temp_file, 'r', encoding='utf-8') as f:
            informations = f.readlines()
        exist_error = False
        for info in informations:
            for error in ErrorCode:
                if info.startswith(error):
                    print(f'\033[31m[tf2onnx Error] {info}\033[0m')
                    exist_error = True
        if exist_error == False:
            if os.path.exists(onnx_model_path+'/'+onnx_file):
                os.remove(temp_file)
            else:
                os.rename(temp_file, base_path+'/'+log_file)
