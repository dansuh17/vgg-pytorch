import os
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from torch.utils import data
import torchvision.datasets as datasets
import torchvision.transforms as transforms
from tensorboardX import SummaryWriter
from model import VGGNet, VGG_CONFS

# define pytorch device - useful for device-agnostic execution
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

DATASET = 'imagenet'  # or 'cifar10' or 'cifar100'
MODEL_TYPE = 'vgg16'
# define model parameters based on original paper
NUM_EPOCHS = 74
BATCH_SIZE = 256
MOMENTUM = 0.9
LR_INIT = 0.0001
DEVICE_IDS = [0, 1, 2, 3]  # GPUs to use
# modify this to point to your data directory
INPUT_ROOT_DIR = 'vggnet_data_in'
TRAIN_IMG_DIR = os.path.join(INPUT_ROOT_DIR, DATASET)
OUTPUT_DIR = 'vggnet_data_out'
LOG_DIR = OUTPUT_DIR + '/tblogs'  # tensorboard logs
CPT_DIR = OUTPUT_DIR + '/checkpoints'  # checkpoint directory

normalize = transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])

# create dataset
dataset = None
NUM_CLASSES = None
IMAGE_DIM = None
if DATASET == 'imagenet':
    NUM_CLASSES = 1000  # 1000 classes for imagenet challenge 2012
    IMAGE_DIM = 224  # pixels
    # create dataset and data loader
    dataset = datasets.ImageFolder(TRAIN_IMG_DIR, transforms.Compose([
        transforms.RandomResizedCrop(IMAGE_DIM, scale=(0.9, 1.0), ratio=(0.9, 1.1)),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        normalize,
    ]))
elif DATASET == 'cifar10':
    NUM_CLASSES = 10
    IMAGE_DIM = 32
    dataset = datasets.CIFAR10(root=TRAIN_IMG_DIR, train=True, download=True,
                               transform=transforms.Compose([
                                   transforms.RandomHorizontalFlip(),
                                   transforms.RandomResizedCrop(32, scale=(0.8, 1.0)),
                                   transforms.ToTensor(),
                                   normalize,
                               ]))
elif DATASET == 'cifar100':
    NUM_CLASSES = 100
    IMAGE_DIM = 32
    dataset = datasets.CIFAR100(
        root=TRAIN_IMG_DIR, train=True, download=True,
        transform=transforms.Compose([
            transforms.RandomHorizontalFlip(),
            transforms.RandomResizedCrop(32, scale=(0.8, 1.0)),
            transforms.ToTensor(),
            normalize,
        ]))
print('Dataset created - size: {}'.format(len(dataset)))

seed = torch.initial_seed()
print('Using seed : {}'.format(seed))

# create model
vggnet = VGGNet(VGG_CONFS[MODEL_TYPE], dim=IMAGE_DIM, num_classes=NUM_CLASSES).to(device)
# train on multiple GPUs
vggnet = torch.nn.parallel.DataParallel(vggnet, device_ids=DEVICE_IDS)
print(vggnet)
print('VGGNet created')

dataloader = data.DataLoader(
    dataset,
    shuffle=True,
    pin_memory=True,
    drop_last=True,
    num_workers=4,
    batch_size=BATCH_SIZE)
print('Dataloader created')

# create optimizer
optimizer = optim.SGD(
    params=vggnet.parameters(),
    lr=LR_INIT,
    momentum=MOMENTUM)
print('Optimizer created')

# multiply LR by 1 / 10 after every 20 epochs
lr_scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=20, gamma=0.1)
print('LR Scheduler created')

tbwriter = SummaryWriter(log_dir=LOG_DIR)
print('TensorboardX summary writer created')

# criterion defined
criterion = nn.CrossEntropyLoss()
print('Criterion defined')

# start training!!
print('Starting training...')
total_steps = 1
for epoch in range(NUM_EPOCHS):
    lr_scheduler.step()
    for imgs, classes in dataloader:
        imgs, classes = imgs.to(device), classes.to(device)
        optimizer.zero_grad()

        # calculate the loss
        output = vggnet(imgs)
        loss = F.cross_entropy(output, classes)

        # update the parameters
        loss.backward()
        optimizer.step()

        # log the information and add to tensorboard
        if total_steps % 10 == 0:
            with torch.no_grad():
                _, preds = torch.max(output, 1)
                accuracy = torch.sum(preds == classes)

                print('Epoch: {} \tStep: {} \tLoss: {:.4f} \tAcc: {}'
                      .format(epoch + 1, total_steps, loss.item(), accuracy.item()))
                tbwriter.add_scalar('loss', loss.item(), total_steps)

        if total_steps % 100 == 0:
            with torch.no_grad():
                for name, parameter in vggnet.named_parameters():
                    if parameter.grad is not None:
                        avg_grad = torch.mean(parameter.grad)
                        print('\tavg_grad for {} = {:.4f}'.format(name, avg_grad))
                        tbwriter.add_scalar('avg_grad/{}'.format(name), avg_grad.item(), total_steps)
                        tbwriter.add_histogram('grad/{}'.format(name), parameter.grad.cpu().numpy(), total_steps)
                    if parameter.data is not None:
                        avg_weight = torch.mean(parameter.data)
                        print('\tavg_weight for {} = {:.4f}'.format(name, avg_weight))
                        tbwriter.add_scalar('avg_weight/{}'.format(name), avg_weight.item())
                        tbwriter.add_histogram('weight/{}'.format(name), parameter.data.cpu().numpy(), total_steps)

        total_steps += 1

    # save checkpoint after epoch
    cpt_dir = os.path.join(CPT_DIR, 'checkpoint_e{}.pkl'.format(epoch + 1))
    torch.save({
        'epocoh': epoch,
        'model': vggnet.state_dict(),
        'optimizer': optimizer.state_dict(),
        'seed': seed,
        'total_steps': total_steps,
    }, cpt_dir)
tbwriter.close()
