# -*- coding: utf-8 -*-
import onnx
import onnxruntime
import numpy as np
import tensorrt as trt
import pycuda.driver as cuda
import pycuda.autoinit
import torch
import argparse
from pathlib import Path
import os
import datetime
import pandas as pd
import json
import re
from utils import MemRecord as memchk
import utils


def get_onnx_files(root_path):
    onnx_list = []
    for (root, dirs, files) in os.walk(root_path):
        for file_name in files:
            if file_name.endswith(".onnx"):
                file_path = os.path.join(root, file_name)
                onnx_list.append(file_path)
    return onnx_list


'''
CheckShape function : set shape if shape's datatype is not integer
MakeDataset function : input random value

It is applied differently depending on the data types,shape and dimensions.
eg) (batch size, 3, 244, 244)
- first dimension is batch size. so set batch size == 1
eg) (batch size, 3, unknown, unknown)
- if other dimension(except first dimension)'s datatype is not integer, set value 128 if it is related to sequence model 
or set value hyper parameter (h_input) if it is related to image or other models
- datatype is depending on onnx inputs,outputs, datatype
- random integer will be set 2 cases.
    1. (1,150) <- sequence model
    2. (400,700) <- vision model
'''


def MakeDataset(h_input, inputs, outputs):
    temp_data = []

    def CheckShape(datatype):
        for i in range(len(datatype['shape'])):
            for j in range(len(datatype['shape'][i])):
                if type(datatype['shape'][i][j]) != int:
                    if datatype['shape'][i][j] == 'batch_size' or datatype['shape'][i][j] == 'batch' or j == 0:
                        datatype['shape'][i][j] = 1
                    elif datatype['shape'][i][j] == 'sequence':
                        datatype['shape'][i][j] = 128
                    else:
                        datatype['shape'][i][j] = h_input

    CheckShape(inputs)
    CheckShape(outputs)

    device = torch.device(
        'cuda:0') if torch.cuda.is_available() else torch.device('cpu')

    for k in range(len(inputs['shape'])):
        if inputs['data_type'][k] == 'int64' or inputs['data_type'][k] == 'int':
            if 'sequence' in inputs['shape'][k]:
                temp_data.append(torch.randint(1, 150, tuple(
                    inputs['shape'][k]), dtype=torch.int64).to(device))
            else:
                temp_data.append(torch.randint(400, 700, tuple(
                    inputs['shape'][k]), dtype=torch.int64).to(device))
        elif inputs['data_type'][k] == 'int32':
            if 'sequence' in inputs['shape'][k]:
                temp_data.append(torch.randint(1, 150, tuple(
                    inputs['shape'][k]), dtype=torch.int32).to(device))
            else:
                temp_data.append(torch.randint(400, 700, tuple(
                    inputs['shape'][k]), dtype=torch.int32).to(device))
        elif inputs['data_type'][k] == 'uint8':
            temp_data.append(torch.randint(1, 127, tuple(
                inputs['shape'][k]), dtype=torch.uint8).to(device))

        elif inputs['data_type'][k] == 'float32' or inputs['data_type'][k] == 'flaot64' or inputs['data_type'][k] == 'float':
            temp_data.append(torch.randn(
                tuple(inputs['shape'][k]), dtype=torch.float32).to(device))

    inputs['data'] = temp_data

    return inputs, outputs


def to_numpy(tensor):
    return tensor.detach().cpu().numpy() if tensor.requires_grad else tensor.cpu().numpy()


def report_unsupport_operation(csv_path, model_name, operation_dict):
    unsupported_op_pd = pd.DataFrame(operation_dict).transpose()
    unsupported_op_pd.columns = ['Count', 'Name']
    print(unsupported_op_pd)
    unsupported_op_pd.to_csv(csv_path+'/'+model_name+'.csv')
    return


"""
Comparison class
- compare relative difference between onnx inference output and tensorRT inference output
- compare absolute difference between onnx inference output and tensorRT inference output
- record position and value about max difference between each onnx inference output and tensorRT inference output
- if there is unsupported operation for tensorRT, it is going to record in dictionary.
"""


class Comparison(object):
    model_dict = {}

    def __init__(self, mem_used_percent, mem_used_MB, onnx_ptime, trt_ptime, ort_throughput, trt_throughput, onnx_file_size, trt_file_size, fail_to_convert_trt_flag, unsupportedOp, inputs, outputs, model_name, onnx_output, trt_output):
        self.mem_used_percent = mem_used_percent
        self.mem_used_MB = mem_used_MB
        self.onnx_ptime = onnx_ptime
        self.trt_ptime = trt_ptime
        self.ort_throughput = ort_throughput
        self.trt_throughput = trt_throughput
        self.onnx_file_size = onnx_file_size
        self.trt_file_size = trt_file_size
        self.fail_to_convert_trt_flag = fail_to_convert_trt_flag
        self.unsupportedOp = unsupportedOp
        self.inputs = inputs
        self.outputs = outputs
        self.input_name = inputs['name']
        self.input_shape = inputs['shape']
        self.output_name = outputs['name']
        self.output_shape = outputs['shape']
        self.priority_dict = {}
        self.model_name = model_name
        self.onnx_output = onnx_output
        self.trt_output = trt_output
        self.atol = 1e-04
        self.rtol = 1e-05
        self.rel_diff_propotion = []
        self.abs_diff_propotion = []
        self.max_value = []
        self.max_pos = []
        if self.fail_to_convert_trt_flag == False:
            self.do_comparison()
        self.reporting()

    def do_comparison(self):

        # if, each output order is different, then reorder wrong order.
        for i in range(len(self.onnx_output)):
            for j in range(len(self.trt_output)):
                if self.onnx_output[i].shape == self.trt_output[j].shape:
                    self.priority_dict[i] = j

        for i in range(len(self.onnx_output)):
            if i in self.priority_dict:
                self.max_value.append(
                    abs(self.onnx_output[i]-self.trt_output[self.priority_dict[i]]).max())
                temp_pos = np.where(abs(
                    self.onnx_output[i]-self.trt_output[self.priority_dict[i]]) == self.max_value[i])

                rel_diff = np.isclose(
                    self.onnx_output[i], self.trt_output[self.priority_dict[i]], rtol=self.rtol)
                self.rel_diff_propotion.append(
                    (round((rel_diff == True).sum() / np.size(rel_diff), 6)))

                abs_diff = np.isclose(
                    self.onnx_output[i], self.trt_output[self.priority_dict[i]], atol=self.atol)
                self.abs_diff_propotion.append(
                    (round((abs_diff == True).sum() / np.size(abs_diff), 6)))
            else:
                if self.onnx_output[i].shape != self.trt_output[i].shape:
                    print('\033[31m'+str(self.output_name[i])+' shape is different! '+'onnx : '+str(
                        self.onnx_output[i].shape)+' trt : '+str(self.trt_output[i].shape)+'\033[0m')
                    self.max_value.append(None)
                    self.rel_diff_propotion.append(None)
                    self.abs_diff_propotion.append(None)
                    temp_pos = None
                else:
                    self.max_value.append(
                        abs(self.onnx_output[i]-self.trt_output[i]).max())
                    temp_pos = np.where(
                        abs(self.onnx_output[i]-self.trt_output[i]) == self.max_value[i])

                    rel_diff = np.isclose(
                        self.onnx_output[i], self.trt_output[i], rtol=self.rtol)
                    self.rel_diff_propotion.append(
                        (round((rel_diff == True).sum() / np.size(rel_diff), 6)))

                    abs_diff = np.isclose(
                        self.onnx_output[i], self.trt_output[i], atol=self.atol)
                    self.abs_diff_propotion.append(
                        (round((abs_diff == True).sum() / np.size(abs_diff), 6)))

            if temp_pos is not None:
                tuple_pos = tuple([temp_pos[j][0]
                                  for j in range(len(temp_pos))])
            else:
                tuple_pos = None
            self.max_pos.append(tuple_pos)

    def reporting(self):
        remove_parentheses = re.compile('[\[\]\']')
        print_input_name = re.sub(remove_parentheses, '', str(self.input_name))
        print_output_name = re.sub(
            remove_parentheses, '', str(self.output_name))
        print_unsupported_op_layers = re.sub(
            remove_parentheses, '', str(self.unsupportedOp))
        print_mem_used_percent = round(self.mem_used_percent, 2)
        print_mem_used_MB = self.mem_used_MB
        print_onnx_size = round(
            self.onnx_file_size/1000000, 3) if self.onnx_file_size is not None else None
        # if self.onnx_file_size is not None:
        #     print_onnx_size = round(self.onnx_file_size/1000000, 3)
        print_trt_size = round(self.trt_file_size/1000000,
                               3) if self.trt_file_size is not None else None
        # if self.trt_file_size is not None:
        #     print_trt_size = round(self.trt_file_size/1000000, 3)

        def is_not_None(calculate_value):
            if calculate_value is not None:
                calculate_value = re.sub(
                    remove_parentheses, '', str(calculate_value))
            return calculate_value

        if self.fail_to_convert_trt_flag == False:
            print_abs_pos = is_not_None(self.abs_diff_propotion)
            print_rel_pos = is_not_None(self.rel_diff_propotion)
            print_max_val = is_not_None(self.max_value)
            print_max_pos = is_not_None(self.max_pos)
            Comparison.model_dict[self.model_name] = [print_input_name, str(tuple(self.input_shape)), print_output_name, str(tuple(self.output_shape)), print_abs_pos,
                                                      print_rel_pos, print_max_val, print_max_pos, print_unsupported_op_layers, print_mem_used_percent, print_mem_used_MB, self.onnx_ptime, self.trt_ptime, self.ort_throughput, self.trt_throughput, print_onnx_size, print_trt_size]
        else:
            Comparison.model_dict[self.model_name] = [print_input_name, str(tuple(self.input_shape)), print_output_name, str(tuple(self.output_shape)), None,
                                                      None, None, None, print_unsupported_op_layers, print_mem_used_percent, print_mem_used_MB, self.onnx_ptime, self.trt_ptime, self.ort_throughput, self.trt_throughput, print_onnx_size, print_trt_size]


class EvaluateOnnx(object):
    def __init__(self, onnx_model, h_input_shape):
        self.onnx_model = onnx_model
        self.h_input_shape = h_input_shape
        self.inputs = {}
        self.input_name = []
        self.input_shape = []
        self.input_data = []
        self.input_data_type = []
        self.outputs = {}
        self.output_name = []
        self.output_shape = []
        self.ort_inputs = {}
        self.onnx_file_size = None

    def remove_initializer_from_input(self):

        model = onnx.load(self.onnx_model)
        self.onnx_file_size = utils.ModelSizeCheck.get_filesize(
            self.onnx_model)
        if model.ir_version < 4:
            print(
                "Model with ir_version below 4 requires to include initilizer in graph input")
        else:
            inputs = model.graph.input
            name_to_input = {}
            for input in inputs:
                name_to_input[input.name] = input
            for initializer in model.graph.initializer:
                if initializer.name in name_to_input:
                    inputs.remove(name_to_input[initializer.name])
        # python <- protobuf object has 2GB limit <- use c++ cf. https://github.com/onnx/onnx/pull/3012
        onnx_bytes = model.SerializeToString()
        return onnx_bytes

    def do_onnx_inference(self):
        onnx_bytes = self.remove_initializer_from_input()
        # providers=['CPUExecutionProvider','CUDAExecutionProvider','TensorrtExecutionProvider'])
        session = onnxruntime.InferenceSession(
            onnx_bytes, providers=['CPUExecutionProvider'])
        type_catcher = re.compile('\w+\((\w+?)\)')

        for i in range(len(session.get_inputs())):
            self.input_name.append(session.get_inputs()[i].name)
            self.input_shape.append(session.get_inputs()[i].shape)
            self.input_data_type.append(type_catcher.match(
                session.get_inputs()[i].type).group(1))

        self.inputs['name'] = self.input_name
        self.inputs['shape'] = self.input_shape
        self.inputs['data'] = self.input_data
        self.inputs['data_type'] = self.input_data_type

        for j in range(len(session.get_outputs())):
            self.output_name.append(session.get_outputs()[j].name)
            self.output_shape.append(session.get_outputs()[j].shape)

        self.outputs['name'] = self.output_name
        self.outputs['shape'] = self.output_shape

        self.inputs, self.outputs = MakeDataset(
            self.h_input_shape, self.inputs, self.outputs)

        for k in range(len(self.inputs['name'])):
            self.ort_inputs[self.inputs['name'][k]
                            ] = to_numpy(self.inputs['data'][k])

        ort_outs = session.run(self.output_name, self.ort_inputs)
        return self.onnx_file_size, self.inputs, self.outputs, ort_outs


class OnnxToTensorRT(object):

    EXPLICIT_BATCH = 1 << (int)(
        trt.NetworkDefinitionCreationFlag.EXPLICIT_BATCH)
    # You can set the logger severity higher to suppress messages (or lower to display more messages).
    #TRT_LOGGER = trt.Logger(trt.Logger.WARNING)
    TRT_LOGGER = trt.Logger(trt.Logger.ERROR)

    def __init__(self, inputs, onnx_model, trt_model_save_path, tensorrt_file_name, trt_save_flag, fail_to_convert_trt_flag):
        self.inputs = inputs
        self.onnx_model = onnx_model
        self.trt_model_save_path = trt_model_save_path
        self.tensorrt_file_name = tensorrt_file_name
        self.trt_save_flag = trt_save_flag
        self.fail_to_convert_trt_flag = fail_to_convert_trt_flag
        self.unsupportedOperationName = []
        self.trt_support_node = []
        self.unsupported_op = {}

    # The Onnx path is used for Onnx models.
    def build_engine_onnx(self,):
        builder = trt.Builder(OnnxToTensorRT.TRT_LOGGER)
        profile = builder.create_optimization_profile()
        for i in range(len(self.inputs['name'])):
            profile.set_shape(self.inputs['name'][i], tuple(self.inputs['shape'][i]), tuple(
                self.inputs['shape'][i]), tuple(self.inputs['shape'][i]))
        network = builder.create_network(OnnxToTensorRT.EXPLICIT_BATCH)
        config = builder.create_builder_config()
        config.add_optimization_profile(profile)
        # Due to tensorrt 8.4 DeprecationWarning
        config.set_memory_pool_limit(trt.MemoryPoolType.WORKSPACE, 8 << 30)
        parser = trt.OnnxParser(network, OnnxToTensorRT.TRT_LOGGER)

        # Load the Onnx model and parse it in order to populate the TensorRT network.
        with open(self.onnx_model, "rb") as model:
            chk_support = parser.supports_model(model.read())
            if parser.num_errors != 0:
                print("\033[31m" +
                      "ERROR: Failed to parse the ONNX file."+"\033[0m")
                if parser.get_error(0).node() == -1:
                    # 이 부분을 단순히 Unsupported Node로 받아오기에는 무리가 있음 Assertion failed 이유를 봐야됨
                    print("\033[31m"+"ERROR: Unsupported Node(i.e Check " +
                          str(self.inputs['data_type'])+"\033[0m")
                    self.unsupported_op[re.sub('[\[\]\']', '', str(self.inputs['data_type']))] = [
                        1, ['invaild data type']]
                    self.fail_to_convert_trt_flag = True
                for subgraph in chk_support[1]:
                    for idx in subgraph[0]:
                        self.trt_support_node.append(idx)
                onnx_model_detailed = onnx.load(self.onnx_model)
                for node_idx, node_info in enumerate(onnx_model_detailed.graph.node):
                    if node_idx not in self.trt_support_node:
                        if node_info.op_type not in self.unsupported_op:
                            self.unsupported_op[node_info.op_type] = [
                                1, [node_info.name]]
                        else:
                            self.unsupported_op[node_info.op_type][0] += 1
                            self.unsupported_op[node_info.op_type][1].append(
                                node_info.name)
                self.unsupportedOperationName = list(
                    self.unsupported_op.keys())
                self.fail_to_convert_trt_flag = True
                return self.unsupported_op
            else:
                return builder.build_engine(network, config=config)
        # return builder.build_serialized_network(network,config=config) # Due to tensorrt 8.4 DeprecationWarning

    def export_engine(self):
        trt_engine = self.build_engine_onnx()
        if self.fail_to_convert_trt_flag == True:
            print('\033[31m'+'ERROR: Cannot convert to TensorRT due to ' +
                  re.sub('[\[\]\']', '', str(self.unsupportedOperationName))+'\033[0m')
            return self.unsupported_op
        if self.trt_save_flag == False:
            return trt_engine
        else:
            buf = trt_engine.serialize()
            with open(self.trt_model_save_path+'/'+self.tensorrt_file_name, 'wb') as f:
                f.write(buf)


class HostDeviceMem(object):
    def __init__(self, host_mem, device_mem):
        self.host = host_mem
        self.device = device_mem

    def __str__(self):
        return "Host:\n" + str(self.host) + "\nDevice:\n" + str(self.device)

    def __repr__(self):
        return self.__str__()


class EvaluateTRT(object):

    # due to cudnn version <- have to upgrade 8.4.1? now 8.4.0
    EXPLICIT_BATCH = 1 << (int)(
        trt.NetworkDefinitionCreationFlag.EXPLICIT_BATCH)
    # create builder <- optimization profile, Takes a network in TensorRT and generates an engine that is optimized for the target platform.
    TRT_LOGGER = trt.Logger(trt.Logger.ERROR)
    trt_runtime = trt.Runtime(TRT_LOGGER)
    trt.init_libnvinfer_plugins(None, '')

    def __init__(self, input_data, outputs, trt_model_save_path, tensorrt_file_name, trt_save_flag, skip_convert_stage_flag, engine):
        self.input_data = input_data['data']
        self.output_data = outputs['name']
        self.tensorrt_model = trt_model_save_path+'/'+tensorrt_file_name
        self.trt_save_flag = trt_save_flag
        self.skip_convert_stage_flag = skip_convert_stage_flag
        self.trt_engine = engine
        self.context = None
        self.inputs = []
        self.outputs = []
        self.bindings = []
        self.output_shape = []
        self.stream = cuda.Stream()

    def deserialize_engine(self):
        if self.trt_save_flag == True or self.skip_convert_stage_flag == True:
            with open(self.tensorrt_model, 'rb') as f:
                engine_data = f.read()
                engine = EvaluateTRT.trt_runtime.deserialize_cuda_engine(
                    engine_data)
        else:
            engine = self.trt_engine

        self.context = engine.create_execution_context()

        for binding in engine:
            if binding in self.output_data:
                self.output_shape.append(engine.get_binding_shape(binding))
            size = trt.volume(engine.get_binding_shape(
                binding)) * EvaluateTRT.EXPLICIT_BATCH
            dtype = trt.nptype(engine.get_binding_dtype(binding))
            host_mem = cuda.pagelocked_empty(size, dtype)
            device_mem = cuda.mem_alloc(host_mem.nbytes)
            self.bindings.append(int(device_mem))
            # Append to the appropriate list.
            if engine.binding_is_input(binding):
                self.inputs.append(HostDeviceMem(host_mem, device_mem))
            else:
                self.outputs.append(HostDeviceMem(host_mem, device_mem))

        for i in range(len(self.inputs)):
            temp_data = to_numpy(self.input_data[i])
            np.copyto(self.inputs[i].host, np.ravel(temp_data))

        for j in range(len(self.output_data)):
            self.outputs[j].host.resize(tuple(self.output_shape[j]))

    # This function is generalized for multiple inputs/outputs for full dimension networks.
    # inputs and outputs are expected to be lists of HostDeviceMem objects.
    def do_inference_v2(self):
        self.deserialize_engine()
        # Transfer input data to the GPU.
        [cuda.memcpy_htod_async(inp.device, inp.host, self.stream)
         for inp in self.inputs]
        # Run inference.
        self.context.execute_async_v2(
            bindings=self.bindings, stream_handle=self.stream.handle)
        # Transfer predictions back from the GPU.
        [cuda.memcpy_dtoh_async(out.host, out.device, self.stream)
         for out in self.outputs]

        # Synchronize the stream
        self.stream.synchronize()

        trt_outputs = [out.host for out in self.outputs]
        return trt_outputs


def process_converts_model(onnx_model, trt_model_save_path, trt_save_flag, h_input_shape, report_path, unsupported_report_path, skip_convert_stage_flag):
    throughput_chk = utils.ThroughtputCheck()
    final_report = {}
    fail_to_convert_trt_flag = False
    onnx_model_name = os.path.basename(onnx_model)
    before_mem_percent, before_mem_mb = memchk.initial_mem_usage()
    print('\033[93m'+f'Working Model : {onnx_model_name}'+'\033[0m')
    tensorrt_file_name = onnx_model_name.replace('.onnx','.trt')
    trt_file_size = None
    trt_ptime = None
    trt_throughput = None
    unsupportedOp = None
    trt_output = {}

    try:
        throughput_chk.start_working()
        EvalOnnx = EvaluateOnnx(onnx_model, h_input_shape)
        onnx_file_size, inputs, outputs, onnx_output = EvalOnnx.do_onnx_inference()
        throughput_chk.end_working()
        onnx_ptime = throughput_chk.show_onnx_time_record()
        ort_throughput = throughput_chk.cal_throughput(onnx_ptime)
    except Exception as e:
        print('\033[31m'+f'ERROR: Failed to parse the ONNX file.\n{e}'+'\033[0m')
        return 

    if skip_convert_stage_flag == False:
        throughput_chk.start_working()
        OnnxToTrt = OnnxToTensorRT(
            inputs, onnx_model, trt_model_save_path, tensorrt_file_name, trt_save_flag, fail_to_convert_trt_flag)
        engine = OnnxToTrt.export_engine()
        if OnnxToTrt.fail_to_convert_trt_flag == False:
            throughput_chk.end_working()
            throughput_chk.show_onnx_to_trt_time_record()
    else:
        print(
            '\033[91m'+'Skip Convert Stage(Onnx model -> TensorRT model)'+'\033[0m')

    if OnnxToTrt.fail_to_convert_trt_flag == False:
        try:
            throughput_chk.start_working()
            EvalTrt = EvaluateTRT(inputs, outputs, trt_model_save_path,
                              tensorrt_file_name, trt_save_flag, skip_convert_stage_flag, engine)
            trt_output = EvalTrt.do_inference_v2()
            throughput_chk.end_working()
            trt_ptime = throughput_chk.show_trt_time_record()
            trt_throughput = throughput_chk.cal_throughput(trt_ptime)
        except Exception as e:
            print(f'')
         
    else:
        unsupportedOp = OnnxToTrt.unsupportedOperationName
        report_unsupport_operation(unsupported_report_path, Path(
            onnx_model).stem, OnnxToTrt.unsupported_op)

    if trt_save_flag is True and  OnnxToTrt.fail_to_convert_trt_flag == False:
        trt_file_size = utils.ModelSizeCheck.get_filesize(
            trt_model_save_path+'/'+tensorrt_file_name)
    after_mem_percent, after_mem_mb = memchk.after_mem_usage()
    mem_used_percent = after_mem_percent - before_mem_percent
    mem_used_MB = round(after_mem_mb - before_mem_mb, 3)
    res = Comparison(mem_used_percent, mem_used_MB, onnx_ptime, trt_ptime, ort_throughput, trt_throughput, onnx_file_size, trt_file_size,
                     OnnxToTrt.fail_to_convert_trt_flag, unsupportedOp, inputs, outputs, onnx_model, onnx_output, trt_output)
    now = datetime.datetime.now()

    print('\033[32m'+'[Final Report] : ['+Path(onnx_model).stem +
          ' - '+(now.strftime("%c"))+']'+'\033[0m')
    print(res.model_dict)
    final_report = res.model_dict
    with open(report_path, 'w', encoding='utf-8') as f:
        json.dump(final_report, f, indent=4)

    return


def process_convert_models(onnx_root_dir, trt_model_save_path, trt_save_flag, h_input_shape, report_path, unsupported_report_path, skip_convert_stage_flag):
    #warnings.filterwarnings("ignore", category=DeprecationWarning)
    throughput_chk = utils.ThroughtputCheck()
    final_report = {}
    onnx_models = get_onnx_files(onnx_root_dir)
    fail_to_convert_trt_flag = False

    for onnx_model in onnx_models:
        before_mem_percent, before_mem_mb = memchk.initial_mem_usage()
        model_name = Path(onnx_model).stem
        print('\033[93m'+f'Working Model : {model_name}'+'\033[0m')
        tensorrt_file_name = model_name+'.trt'
        trt_file_size = None
        trt_ptime = None
        trt_throughput = None
        unsupportedOp = None
        trt_output = {}
        
        try:
            throughput_chk.start_working()
            EvalOnnx = EvaluateOnnx(onnx_model, h_input_shape)
            onnx_file_size, inputs, outputs, onnx_output = EvalOnnx.do_onnx_inference()
            throughput_chk.end_working()
            onnx_ptime = throughput_chk.show_onnx_time_record()
            ort_throughput = throughput_chk.cal_throughput(onnx_ptime)
        except Exception as e:
            print('\033[31m'+f'ERROR: Failed to parse the ONNX file.\n{e}'+'\033[0m')
            continue

        if skip_convert_stage_flag == False:
            throughput_chk.start_working()
            OnnxToTrt = OnnxToTensorRT(
                inputs, onnx_model, trt_model_save_path, tensorrt_file_name, trt_save_flag, fail_to_convert_trt_flag)
            engine = OnnxToTrt.export_engine()
            if OnnxToTrt.fail_to_convert_trt_flag == False:
                throughput_chk.end_working()
                throughput_chk.show_onnx_to_trt_time_record()
        else:
            print(
                '\033[91m'+'Skip Convert Stage(Onnx model -> TensorRT model)'+'\033[0m')

        if OnnxToTrt.fail_to_convert_trt_flag == False:
            throughput_chk.start_working()
            EvalTrt = EvaluateTRT(inputs, outputs, trt_model_save_path,
                                  tensorrt_file_name, trt_save_flag, skip_convert_stage_flag, engine)
            trt_output = EvalTrt.do_inference_v2()
            throughput_chk.end_working()
            trt_ptime = throughput_chk.show_trt_time_record()
            trt_throughput = throughput_chk.cal_throughput(trt_ptime)
        else:
            unsupportedOp = OnnxToTrt.unsupportedOperationName
            report_unsupport_operation(unsupported_report_path, Path(
                onnx_model).stem, OnnxToTrt.unsupported_op)

        if trt_save_flag is True:
            trt_file_size = utils.ModelSizeCheck.get_filesize(
                trt_model_save_path+'/'+tensorrt_file_name)
        after_mem_percent, after_mem_mb = memchk.after_mem_usage()
        mem_used_percent = after_mem_percent - before_mem_percent
        mem_used_MB = round(after_mem_mb - before_mem_mb, 3)
        res = Comparison(mem_used_percent, mem_used_MB, onnx_ptime, trt_ptime, ort_throughput, trt_throughput, onnx_file_size, trt_file_size,
                         OnnxToTrt.fail_to_convert_trt_flag, unsupportedOp, inputs, outputs, model_name, onnx_output, trt_output)
        now = datetime.datetime.now()
        print('\033[32m'+'[Final Report] : ['+model_name +
              ' - '+(now.strftime("%c"))+']'+'\033[0m')
        print(res.model_dict)
        final_report = res.model_dict

    with open(report_path, 'w', encoding='utf-8') as f:
        json.dump(final_report, f, indent=4)

    return


if __name__ == "__main__":
    pass
