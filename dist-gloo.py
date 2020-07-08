import os
from datetime import datetime
import argparse
import torch.multiprocessing as mp
import torchvision
import torchvision.datasets as dset
import torchvision.transforms as transforms
import torch
import torch.nn as nn
import torch.distributed as dist


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('-n', '--nodes', default=1, type=int, metavar='N',
                        help='number of data loading workers (default: 4)')
    parser.add_argument('-g', '--gpus', default=1, type=int,
                        help='number of gpus per node')
    parser.add_argument('-nr', '--nr', default=0, type=int,
                        help='ranking within the nodes')
    parser.add_argument('--epochs', default=2, type=int, metavar='N',
                        help='number of total epochs to run')
    
    args = parser.parse_args()
    args.world_size = args.gpus * args.nodes
    
    # master address and port will be automatically set by Singularity. Run printenv | grep "MASTER" to validate. 
    # os.environ['MASTER_ADDR'] = '10.0.66.7' 
    # os.environ['MASTER_PORT'] = '8883'
    
    mp.spawn(train, nprocs=args.gpus, args=(args,))


class ConvNet(nn.Module):
    def __init__(self, num_classes=10):
        super(ConvNet, self).__init__()
        self.layer1 = nn.Sequential(
            nn.Conv2d(1, 16, kernel_size=5, stride=1, padding=2),
            nn.BatchNorm2d(16),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2))
        self.layer2 = nn.Sequential(
            nn.Conv2d(16, 32, kernel_size=5, stride=1, padding=2),
            nn.BatchNorm2d(32),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2))
        self.fc = nn.Linear(7*7*32, num_classes)

    def forward(self, x):
        out = self.layer1(x)
        out = self.layer2(out)
        out = out.reshape(out.size(0), -1)
        out = self.fc(out)
        return out


def train(gpu, args):
    print('Starting training..')
    rank = args.nr * args.gpus + gpu
    print('Rank {}..'.format(rank), end = '')
    
    print('initializing process group, world_size={}, rank={}'.format(args.world_size, rank), end = '')
    dist.init_process_group(backend='gloo', init_method='env://', world_size=args.world_size, rank=rank)
    print('Done. ', end = '')
    
    torch.manual_seed(0)
    model = ConvNet()
    torch.cuda.set_device(gpu)
    model.cuda(gpu)
    batch_size = 100
    
    # define loss function (criterion) and optimizer
    criterion = nn.CrossEntropyLoss().cuda(gpu)
    optimizer = torch.optim.SGD(model.parameters(), 1e-4)
    
    # Wrap the model
    model = nn.parallel.DistributedDataParallel(model, device_ids=[gpu])

    print('Device_id [{}]..'.format(gpu))

    # Data loading code
    
    root_dir = './data_' + str(rank)
    if not os.path.exists(root_dir):
        os.mkdir(root_dir)
    
    train_dataset = torchvision.datasets.MNIST(root=root_dir,
                                               train=True,
                                               transform=transforms.ToTensor(),
                                               download=True)
    
    train_sampler = torch.utils.data.distributed.DistributedSampler(train_dataset,
                                                                    num_replicas=args.world_size,
                                                                    rank=rank)
    train_loader = torch.utils.data.DataLoader(dataset=train_dataset,
                                               batch_size=batch_size,
                                               shuffle=False,
                                               num_workers=0,
                                               pin_memory=True,
                                               sampler=train_sampler)

    start = datetime.now()
    total_step = len(train_loader)
    count_epoch = 1
    for epoch in range(args.epochs):
        print('Epoch [{}]..'.format(count_epoch))
        count_iteration = 1
        for i, (images, labels) in enumerate(train_loader):
            print('Epoch [{}] Iteration [{}] started. '.format(count_epoch, count_iteration), end = '')
            images = images.cuda(non_blocking=True)
            labels = labels.cuda(non_blocking=True)
            # Forward pass
            print('Forward Pass - Done. ', end = '')
            outputs = model(images)
            loss = criterion(outputs, labels)

            # Backward and optimize
            print('Backward Pass - Done. ', end = '')
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            print('Iteration [{}] is done.'.format(count_iteration))
            count_iteration += 1

            if (i + 1) % 100 == 0 and gpu == 0:
                print('Epoch [{}/{}], Step [{}/{}], Loss: {:.4f}'.format(epoch + 1, args.epochs, i + 1, total_step,
                                                                         loss.item()))
        count_epoch += 1
    if gpu == 0:
        print("Training complete in: " + str(datetime.now() - start))
        print("Loss: " + str(loss.item()))


if __name__ == '__main__':
    main()
