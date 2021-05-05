import torch

from dataset.dataloader import Dataloader

from models.naive_classifiers import Naive_CNN, Naive_RNN
from models.hrnet_18.cls_hrnet import get_cls_net

import argparse
from models.hrnet_18.hrnet_18_config.default import update_config
from models.hrnet_18.hrnet_18_config.default import _C as config

import tqdm


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


def evaluate(eval_data_path, batchsize, device, pth_path, model):
    eval_dataloader = Dataloader(eval_data_path, batchsize)
    model = model.to(device)
    model.eval()
    if pth_path:
        model.load_state_dict(torch.load(pth_path))

    cm = torch.zeros((4, 4))  # gt at row

    pbar = tqdm.tqdm(total=len(eval_dataloader))
    for data, label in eval_dataloader:
        data = data.to(dtype=torch.float).to(device)
        label = label.to(device)

        pred = torch.argmax(model(data), dim=1)
        for l, p in zip(label, pred):
            cm[l, p] += 1
        pbar.update(label.shape[0])
    pbar.close()
    return cm


if __name__ == '__main__':
    option_list = ['Naive_CNN', 'Naive_RNN', 'HRNet']
    eval_data_root = 'dataset/test'

    choosen_model = option_list[0]
    choice = option_list.index(choosen_model)

    if choice == 0:
        model = Naive_CNN()
        pth_path = 'ckpts/model_cnn.pth'
    elif choice == 1:
        model = Naive_RNN()
        pth_path = 'ckpts/model_rnn.pth'
    else:
        args = parse_args()
        model = get_cls_net(config)
        pth_path = 'ckpts/fine_tuning_hrnet.pth'

    cm = evaluate(eval_data_path=eval_data_root,
                  pth_path=pth_path,
                  batchsize=16,
                  device='cuda',
                  model=model)

    acc = torch.diag(cm).sum() / cm.sum()
    print(acc)

