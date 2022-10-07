import torch
import torchvision
import os
#from efficient_model.efficientdet.model import EfficientNet as efficient

dummy_input = torch.randn(32, 3, 3, 3, device="cuda")

path = os.getcwd()
#model2 = torch.load(path+'/efficient_model/efficientdet-d2.pth')
#model2 = efficient(1).cuda()
# print(type(model2))
# torch.save(model2, path+'/efficient.pt')

# yolo v7
model3 = torch.load(path+'/yolov7-tiny.pt')
print(model3)

#model = torchvision.models.alexnet(pretrained=True).cuda()

# print('===================================================')
# print(type(model))

# Providing input and output names sets the display names for values
# within the model's graph. Setting these does not change the semantics
# of the graph; it is only for readability.
#
# The inputs to the network consist of the flat list of inputs (i.e.
# the values you would pass to the forward() method) followed by the
# flat list of parameters. You can partially specify names, i.e. provide
# a list here shorter than the number of inputs to the model, and we will
# only set that subset of names, starting from the beginning.

# print('===================================================')
# input_names = ["actual_input_1"] + ["learned_%d" % i for i in range(16)]
# print(f'chk! {input_names}')
# output_names = ["output1"]

# torch.onnx.export(model2, dummy_input, path+"/effi_test.onnx", verbose=False,
#                   opset_version=11)


# torch.onnx.export(model, dummy_input, path+"/alexnet.onnx", verbose=True,
#                   input_names=input_names, output_names=output_names, opset_version=13)
