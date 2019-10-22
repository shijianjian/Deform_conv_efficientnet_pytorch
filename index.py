from deform_efficientnet import EfficientNet as DeformEfficientNet
import numpy as np
import torch

if __name__ == "__main__":
    device = "cuda:1"
    model = DeformEfficientNet.from_name("efficientnet-b0").to(device)
    res = model(torch.randn((2, 3, 224, 224)).to(device))
    print(res)
