import torch
import torchvision
from torchvision import transforms
from urllib.request import urlopen
from PIL import Image
import timm


img = torch.randn(10, 4, 224, 224)

backbone = timm.create_model('vit_small_patch16_224', 
                      pretrained=True,
                      num_classes=0,
                      in_chans=4)   # remove classifier nn.Linear
backbone = backbone.eval()
data_config = timm.data.resolve_model_data_config(backbone)
new_transforms = transforms.Compose([
    transforms.Resize(data_config['input_size'][-2:]),
    transforms.CenterCrop(data_config['input_size'][-2:]),
    transforms.Normalize(mean=torch.tensor([0.5000, 0.5000, 0.5000, 0.]), 
                            std=torch.tensor([0.5000, 0.5000, 0.5000, 1.]))
])

output = backbone(new_transforms(torch.concatenate([img, img], dim=0)))
print(output.shape)