import torch
from thop import profile,clever_format
from ptflops import get_model_complexity_info
# import  ml_

from Net.hyperseg.utils.obj_factory import obj_factory
from Net.efficientv2_new import efficient_v2
from Net.viewAL.deeplab import DeepLab
from Net.ocnet.network.resnet101_pyramid_oc import get_resnet50_pyramid_oc_dsn as ocnet

device = torch.device('cuda:0'if torch.cuda.is_available() else 'cpu')

efficient = efficient_v2(2,pretrained=True)
viewAL = DeepLab(num_classes=2,output_stride=16)
Ocnet = ocnet(num_classes=2)
hyper = obj_factory('hyperseg.models.hyperseg_v0_1.hyperseg_efficientnet')   #需要在hyperseg_v0_1.py中修改num_class

model = hyper.to(device)
inputx = torch.rand(1,3,512,512).to(device)

flops,params = profile(model,inputs=(inputx,))

print("thop")
print(flops,params)
flops,params = clever_format([flops,params],"%.4f")
print(flops,params)

# flops,params = get_model_complexity_info(model,(3,512,512),as_strings=True,print_per_layer_stat=True)
# print(flops,params)



