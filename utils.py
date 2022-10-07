import psutil
import os
import pandas as pd
import json
import time


class MemRecord(object):

    def __init__(self):
        pass

    def initial_mem_usage():
        mem_usage_dict = dict(psutil.virtual_memory()._asdict())
        mem_usage_percent = mem_usage_dict['percent']
        pid = os.getpid()
        current_process = psutil.Process(pid)
        mem_usage_as_MB = current_process.memory_info()[0] / 2.**20
        print(
            '\033[94m'+f'BEFORE RUNNING CODE : memory usage(%) : {mem_usage_percent}%'+'\033[0m')
        print(
            '\033[94m'+f'BEFORE RUNNING CODE : memory usage(MB) : {round(mem_usage_as_MB,3)}MB'+'\033[0m')
        return mem_usage_percent, mem_usage_as_MB

    def after_mem_usage():
        # recording by Mega Bytes
        mem_usage_dict = dict(psutil.virtual_memory()._asdict())
        mem_usage_percent = mem_usage_dict['percent']
        pid = os.getpid()
        current_process = psutil.Process(pid)
        mem_usage_as_MB = current_process.memory_info()[0] / 2.**20
        print(
            '\033[94m'+f'AFTER RUNNING CODE : memory usage(%) : {mem_usage_percent}%'+'\033[0m')
        print(
            '\033[94m'+f'AFTER RUNNING CODE : memory usage(MB) : {round(mem_usage_as_MB,3)}MB'+'\033[0m')
        return mem_usage_percent, mem_usage_as_MB


class ThroughtputCheck(object):

    def __init__(self):
        self.start_time = None
        self.end_time = None
        pass

    def start_working(self):
        self.start_time = time.process_time()
        return self.start_time

    def end_working(self):
        self.end_time = time.process_time()
        return self.end_time

    def show_onnx_time_record(self):
        print(
            '\033[96m'+f'[COMPLETE] ONNX Model Inference : {round(self.end_time - self.start_time,6)}(s)'+'\033[0m')
        return round(self.end_time - self.start_time, 6)

    def show_trt_time_record(self):
        print(
            '\033[96m'+f'[COMPLETE] TensorRT Model Inference : {round(self.end_time - self.start_time,6)}(s)'+'\033[0m')
        return round(self.end_time - self.start_time, 6)

    def show_onnx_to_trt_time_record(self):
        print(
            '\033[96m'+f'[COMPLETE] ONNX Model to TensorRT Model : {round(self.end_time - self.start_time,6)}(s)'+'\033[0m')
        return round(self.end_time - self.start_time, 6)

    def cal_throughput(self, ptime):
        return round(1 / ptime, 6)


class ModelSizeCheck(object):

    def __init__(self):
        pass

    def get_filesize(model):
        return os.path.getsize(model)
