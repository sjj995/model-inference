import torch
import warnings
import logging
import inspect
from typing import Union, Tuple, Any, Callable, Iterator, Set, Optional, overload, TypeVar, Mapping, Dict, List
import sys
sys.path.append('/workspace/')
from custom import test_forward_if_script as custom_script


inference_input = torch.randn(1,3,256,256)
condition_testing = False
branch_testing = False

model = custom_script.CustomModel(condition_testing,branch_testing)

# print(model(inference_input))
# print(model.forward)

# def _forward_unimplemented(self, *input: Any) -> None:
#     r"""Defines the computation performed at every call.

#     Should be overridden by all subclasses.

#     .. note::
#         Although the recipe for forward pass needs to be defined within
#         this function, one should call the :class:`Module` instance afterwards
#         instead of this since the former takes care of running the
#         registered hooks while the latter silently ignores them.
#     """
#     raise NotImplementedError


logger = logging.getLogger()
logger.setLevel(logging.INFO)	

_DtypeWarning = 'The layer(op) associated with (Parameter) may not be converted normally. Please check'



class Module():
    def __init__(self,model) -> None:
        # forward: Callable[..., Any] = _forward_unimplemented
        self.model = model
        self.forward = model.forward
        self.caution_list = []
        self.caution_dict = []
        self.argument_list = []
    
    def __len__(self,*args):
        return len(args)

    # forward 함수만 적용
    def get_argument(self,):
        self.argument_list = self.forward.__code__.co_varnames

    def alert_argument(self,):
        for caution in self.caution_list:
            param = self.argument_list[caution]
            warnings.warn('('+param+')'+' '+_DtypeWarning,UserWarning)
    
    def _call_impl(self, *input, **kwargs):
        result = self.forward(*input, **kwargs)
        
        for i in range(self.__len__(*input)):
            if type(input[i]) != torch.Tensor:
                self.caution_list.append(i)
    
        if self.__len__(*kwargs) != 0:
            for k,v in kwargs.items():
                if type(v) != torch.Tensor:
                    self.caution_dict.append(k)
                        
        self.get_argument()
        self.alert_argument()


    __call__ : Callable[..., Any] = _call_impl


module = Module(model)

#forward input이 1개일 경우
#module(inference_input)

#forward input이 여러개일 경우
module(inference_input,inference_input,1.2,3.3)

print(help(model))
