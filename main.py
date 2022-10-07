import argparse
import os
from evaluate_conversion import process_convert_models as onnx_convert
from evaluate_conversion import process_converts_model as onnx_converts
from exec_tf2onnx import process_tf2onnx as tf_convert
from exec_tf2onnx import get_tensorflow_files 
from parse_tf2onnx import process_parsing as tf_log_parsing

_HELP_TEXT = """
Usage Examples:

python prj/main.py --model_type saved-model --tf_model_path "saved model path" --onnx_model_path "onnx model directory" \
    --onnx_model "model.onnx" --trt_model_path "TensorRT save directory" --save-engine True \
    --use_existing_trt_model False --unsupported_report_path  ./
    
python prj/main.py --model_type saved-model --tf_model_path "saved model path" --onnx_model_path "onnx model directory" \
    --is_several True --trt_model_path "TensorRT save directory" --save-engine True \
    --use_existing_trt_model False --unsupported_report_path  ./

"""


def str2bool(v):
    if isinstance(v, bool):
        return v
    if v.lower() in ('yes', 'true', 'True', 't', 'y', '1'):
        return True
    elif v.lower() in ('no', 'false', 'False', 'f', 'n', '0'):
        return False
    else:
        raise argparse.ArgumentTypeError('Boolean value expected.')


def get_abspath(cpath):
    if cpath is not None:
        return os.path.abspath(cpath)


def get_args():
    """Parse commandline."""
    parser = argparse.ArgumentParser(description="Convert tensorflow graphs to TensorRT or ONNX graphs to TensorRT.",
                                     formatter_class=argparse.RawDescriptionHelpFormatter, epilog=_HELP_TEXT)
    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--model_type",
        type=str,
        help="""
        --model_type {saved-model,torch,onnx}
        The type of the input model: {'saved-model': TensorFlow saved model directory, 'torch': Pytorch model 'onnx': ONNX model}
        """
    )

    parser.add_argument(
        "--tf_model_path",
        type=str,
        default=None,
        help="Tensorflow Model Path",
    )

    parser.add_argument(
        "--onnx_model_path",
        type=str,
        default=None,
        help="Onnx Model Path(Directory)"
    )

    parser.add_argument(
        "--onnx_model",
        type=str,
        help="Onnx Model Name(Use only when running one at a time, e.g vgg16.onnx)",
        default=None
    )

    parser.add_argument(
        "--is_several",
        type=str2bool,
        help="Whether or not there are multiple onnx models(T/F)",
        default=None
    )

    parser.add_argument(
        "--trt_model_path",
        type=str,
        help="TensorRT model save path",
        default=None
    )

    parser.add_argument(
        "--save_engine",
        type=str2bool,
        default=False,
        help="decide save TensorRT model(T/F)"
    )

    parser.add_argument(
        "--use_existing_trt_model",
        type=str2bool,
        default=False,
        help="decide skip converting stage(onnx -> trt)"
    )

    parser.add_argument(
        "--h_input_shape",
        type=int,
        default=416,
        help="hyper parameter for input shape"
    )

    parser.add_argument(
        "--unsupported_report_path",
        type=str,
        default=None,
        help="saving path, unsupported op/layers for model"
    )

    args = parser.parse_args()

    args.onnx_model_path = get_abspath(args.onnx_model_path)
    args.tf_model_path = get_abspath(args.tf_model_path)
    args.trt_model_path = get_abspath(args.trt_model_path)

    #Tensorflow,Torch 단독 -> Onnx 모델로 변환할 때
    if args.is_several is False:
        if args.model_type == 'saved-model':
            if args.onnx_model is None:
                args.onnx_model = os.path.basename(args.tf_model_path)+'.onnx'
        elif args.model_type =='torch':
            #torch model path로 받아야됨.
            pass
            #args.onnx_model = os.path.basename(args.torch_model_path)+'.onnx'
        elif args.model_type =='onnx':
            if args.onnx_model is None:
                print('Please enter Onnx Model Name(e.g vgg.onnx)')

    if args.trt_model_path is None:
        print('Please enter TensorRT Model Path')
        return

    if args.unsupported_report_path is None:
        if args.onnx_model_path is not None:
            args.unsupported_report_path = args.onnx_model_path
    return args


def main():
    args = get_args()
    base_path = os.path.dirname(os.path.abspath(__file__))
    report_path = base_path+'/report.json'

    #Tensorflow Models(saved models)
    if args.model_type == 'saved-model':
        if args.is_several is True:
            #Get Tensorflow model list(more than One)
            tf_list = get_tensorflow_files(args.tf_model_path)
            for tf_model in tf_list:
                onnx_model = os.path.basename(tf_model)+'.onnx'
                #Try converting TF to Onnx
                tf_convert(base_path, tf_model,
                    args.onnx_model_path, onnx_model)
                #Check that the ONNX model has been created
                tf_log_parsing(base_path, tf_model,
                        args.onnx_model_path, onnx_model)
                #Onnx Model to TensorRT
                if os.path.exists(args.onnx_model_path+'/'+onnx_model):
                    onnx_converts(args.onnx_model_path+'/'+onnx_model, args.trt_model_path, args.save_engine, args.h_input_shape,
                              report_path, args.unsupported_report_path, args.use_existing_trt_model)
        else:
            #Do just one Tensorlfow model -> Onnx
            tf_convert(base_path, args.tf_model_path,
                   args.onnx_model_path, args.onnx_model)
            tf_log_parsing(base_path, args.tf_model_path,
                    args.onnx_model_path, args.onnx_model)
            if os.path.exists(args.onnx_model_path+'/'+args.onnx_model):
                onnx_converts(args.onnx_model_path+'/'+args.onnx_model, args.trt_model_path, args.save_engine, args.h_input_shape,
                    report_path, args.unsupported_report_path, args.use_existing_trt_model)
    
    elif args.model_type == 'torch':
        pass

    elif args.model_type == 'onnx':
        if args.is_several is True:
            #Get Onnx model list(more than One)
            onnx_convert(args.onnx_model_path, args.trt_model_path, args.save_engine, args.h_input_shape,
                         report_path, args.unsupported_report_path, args.use_existing_trt_model)
        else:
            onnx_converts(args.onnx_model_path+'/'+args.onnx_model, args.trt_model_path, args.save_engine, args.h_input_shape,
                          report_path, args.unsupported_report_path, args.use_existing_trt_model)
    else:
        print("""Please enter the correct model type\n
        --model_type {saved-model,onnx}
        The type of the input model: {'saved-model': TensorFlow saved model directory, 'onnx': ONNX model}
        """)


if __name__ == "__main__":
    main()
