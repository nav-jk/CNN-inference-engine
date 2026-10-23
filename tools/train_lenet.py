"""Train a LeNet on MNIST with PyTorch and export it for the C++ engine.

Run from the repo root:   python3 tools/train_lenet.py
Needs: torch, torchvision, numpy.

Writes (in the current directory):
    lenet.bin             weights, format described in include/utils/load.h
    test_image.bin        one MNIST test image, float32, 1x28x28
    expected_logits.bin   PyTorch's output for that image (to check the engine)

Then:  make run
"""
import struct
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision import datasets, transforms


class LeNet(nn.Module):
    def __init__(self):
        super().__init__()
        self.c1 = nn.Conv2d(1, 6, 5)
        self.c2 = nn.Conv2d(6, 16, 5)
        self.f1 = nn.Linear(256, 120)
        self.f2 = nn.Linear(120, 84)
        self.f3 = nn.Linear(84, 10)

    def forward(self, x):
        x = F.max_pool2d(F.relu(self.c1(x)), 2)
        x = F.max_pool2d(F.relu(self.c2(x)), 2)
        x = x.flatten(1)
        return self.f3(F.relu(self.f2(F.relu(self.f1(x)))))


def export(model, path):
    layers = [model.c1, model.c2, model.f1, model.f2, model.f3]  # weighted layers, in order
    with open(path, "wb") as f:
        f.write(b"CNNW")
        f.write(struct.pack("<I", len(layers)))
        for l in layers:
            l.weight.detach().numpy().astype("<f4").tofile(f)  # conv: OIHW, linear: [out, in]
            l.bias.detach().numpy().astype("<f4").tofile(f)


if __name__ == "__main__":
    tf = transforms.ToTensor()
    train = torch.utils.data.DataLoader(
        datasets.MNIST("data", train=True, download=True, transform=tf),
        batch_size=128, shuffle=True)
    test = datasets.MNIST("data", train=False, download=True, transform=tf)

    model = LeNet()
    opt = torch.optim.Adam(model.parameters(), 1e-3)
    for epoch in range(3):
        for x, y in train:
            opt.zero_grad()
            F.cross_entropy(model(x), y).backward()
            opt.step()
        print("epoch", epoch + 1, "done")

    model.eval()
    export(model, "lenet.bin")
    img, label = test[0]
    img.numpy().astype("<f4").tofile("test_image.bin")
    with torch.no_grad():
        model(img[None])[0].numpy().astype("<f4").tofile("expected_logits.bin")
    print("exported; test image label:", label)
