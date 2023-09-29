import torch
import torchvision
import timm


m = timm.create_model('mobilenetv3_large_100')
m.eval()
print(m)