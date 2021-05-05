import torch
import matplotlib.pyplot as plt
from torch import nn

from dataset.dataloader import Dataloader
from torch import optim

from models.hrnet_18.cls_hrnet import get_cls_net
from models.naive_classifiers import Naive_CNN, Naive_RNN

import argparse
from models.hrnet_18.hrnet_18_config.default import update_config
from models.hrnet_18.hrnet_18_config.default import _C as config

import torch.backends.cudnn as cudnn


def parse_args():
    parser = argparse.ArgumentParser(description='Train keypoints network')

    parser.add_argument('--cfg',
                        help='experiment configure file name',
                        required=True,
                        type=str)

    parser.add_argument('--modelDir',
                        help='model directory',
                        type=str,
                        default='')
    parser.add_argument('--logDir',
                        help='log directory',
                        type=str,
                        default='')
    parser.add_argument('--dataDir',
                        help='data directory',
                        type=str,
                        default='')
    parser.add_argument('--testModel',
                        help='testModel',
                        type=str,
                        default='')

    args = parser.parse_args(['--cfg', 'models/hrnet_18/hrnet_18_config/cls_hrnet_w18_small_v1_sgd_lr5e-2_wd1e-4_bs32_x100.yaml',])
    update_config(config, args)

    return args


def get_parameter_number(net):
    total_num = sum(p.numel() for p in net.parameters())
    trainable_num = sum(p.numel() for p in net.parameters() if p.requires_grad)
    print('Total: ', total_num, 'Trainable: ', trainable_num)


def train_naive_net(train_data_path, batchsize, epoch, device, lr, model, pth_path=None):
    train_dataloader = Dataloader(train_data_path, batchsize)

    model = model.to(device)
    get_parameter_number(model)
    model.train()

    optimizer = torch.optim.SGD(params=model.parameters(), lr=lr, momentum=0.2)

    loss_hist = list()
    criteon = nn.CrossEntropyLoss().to(device)
    for ep in range(epoch):
        for data, label in train_dataloader:
            data = data.to(dtype=torch.float).to(device)
            label = label.to(device)

            logits = model(data)
            loss = criteon(logits, label)
            loss_hist.append(loss)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

    if pth_path:
        torch.save(model.state_dict(), pth_path)
    return loss_hist


def get_optimizer(cfg, model):
    optimizer = None

    # ======================================================
    excluded_group = [model.classifier.parameters(), model.reclassifier.parameters()]
    ignored_ids = list()
    for excluded_terms in excluded_group:
        ignored_ids += list(map(id, excluded_terms))
    exclude_fun = lambda p: id(p) not in ignored_ids

    base_params = filter(exclude_fun, model.parameters())
    # ======================================================

    if cfg.TRAIN.OPTIMIZER == 'sgd':
        optimizer = torch.optim.SGD(
            [{'params': base_params},
             {'params': model.classifier.parameters(), 'lr': cfg.TRAIN.LR},
             {'params': model.reclassifier.parameters(), 'lr': cfg.TRAIN.LR}],
                                    lr=0.1*cfg.TRAIN.LR,
                                    momentum=cfg.TRAIN.MOMENTUM,
                                    weight_decay=cfg.TRAIN.WD,
                                    nesterov=cfg.TRAIN.NESTEROV
                                    )
    elif cfg.TRAIN.OPTIMIZER == 'adam':
        optimizer = optim.Adam(
            filter(lambda p: p.requires_grad, model.parameters()),
            lr=cfg.TRAIN.LR
        )
    elif cfg.TRAIN.OPTIMIZER == 'rmsprop':
        optimizer = optim.RMSprop(
            filter(lambda p: p.requires_grad, model.parameters()),
            lr=cfg.TRAIN.LR,
            momentum=cfg.TRAIN.MOMENTUM,
            weight_decay=cfg.TRAIN.WD,
            alpha=cfg.TRAIN.RMSPROP_ALPHA,
            centered=cfg.TRAIN.RMSPROP_CENTERED
        )
    return optimizer


def fine_tune_hrnet18(train_data_path, batchsize, epoch, device, model, pth_path=None):
    train_dataloader = Dataloader(train_data_path, batchsize)

    model = model.to(device)
    get_parameter_number(model)
    model.train()

    optimizer = get_optimizer(config, model)

    loss_hist = list()
    criteon = nn.CrossEntropyLoss().to(device)
    for ep in range(epoch):
        for data, label in train_dataloader:
            data = data.to(dtype=torch.float).to(device)
            label = label.to(device)

            logits = model(data)
            loss = criteon(logits, label)
            loss_hist.append(loss)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

    if pth_path:
        torch.save(model.state_dict(), pth_path)
    return loss_hist


if __name__ == '__main__':
    option_list = ['Naive_CNN', 'Naive_RNN', 'HRNet']
    train_data_root = 'dataset/train'

    choosen_model = ''
    choice = option_list.index(choosen_model)

    if choice == 0:
        model = Naive_CNN()
        loss_hist = train_naive_net(train_data_path=train_data_root,
              pth_path=None,  # figure out a path to save the model, or it will not be saved.
              batchsize=16,
              epoch=5,
              model=model,
              device='cuda',
              lr=1e-1)

    elif choice == 1:
        model = Naive_RNN()
        loss_hist = train_naive_net(train_data_path=train_data_root,
                                    pth_path=None,  # figure out a path to save the model, or it will not be saved.
                                    batchsize=128,
                                    epoch=200,
                                    model=model,
                                    device='cuda',
                                    lr=1e-2)
    else:
        args = parse_args()
        model = get_cls_net(config)

        # cudnn related setting
        cudnn.benchmark = config.CUDNN.BENCHMARK
        torch.backends.cudnn.deterministic = config.CUDNN.DETERMINISTIC
        torch.backends.cudnn.enabled = config.CUDNN.ENABLED

        loss_hist = fine_tune_hrnet18(train_data_path=train_data_root,
                                    pth_path=None,  # figure out a path to save the model, or it will not be saved.
                                    batchsize=16,
                                    epoch=3,
                                    model=model,
                                    device='cuda')

    plt.figure(1)
    plt.plot(loss_hist)
    plt.show()