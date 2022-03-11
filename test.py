from foolbox import PyTorchModel
from foolbox.attacks import LinfPGD
import numpy as np
import torch

model = resnet50()
checkpoint = torch.load('./models/trained/resnet50.pth.tar')
model.load_state_dict(checkpoint['state_dict'])
model.eval()

preprocessing = dict(mean=[0.4914, 0.4822, 0.4465], std=[0.2023, 0.1994, 0.2010], axis=-3)
fmodel = PyTorchModel(model, bounds=(0, 1), preprocessing=preprocessing)

images = np.load('images.npy')
labels = np.load('labels.npy')

attack = LinfPGD()
_, _, success = attack(fmodel, images, labels, epsilons=[8/255])

robust_accuracy = 1 - success.astype(float).mean(axis=-1)
print(robust_accuracy)

