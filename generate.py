import torch
import torch.nn as nn
import torch.optim as optim
from model import VGGNet, VGG_CONFS


MODEL_PATH = 'models/checkpoint_e49.pkl'
SAMPLE_IMG_PATH = ''
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

cpt = torch.load(MODEL_PATH, map_location=device)
epoch = cpt['epoch']
seed = cpt['seed']
total_steps = cpt['total_steps']
# make even this the same...
vgg16 = nn.parallel.DataParallel(VGGNet(VGG_CONFS['vgg16'], dim=32, num_classes=10))
vgg16.load_state_dict(cpt['model'])
print(vgg16)

# test loading optimizer
optimizer = optim.SGD(vgg16.parameters(), lr=0.0001, weight_decay=0.00005, momentum=0.9).load_state_dict(cpt['optimizer'])
