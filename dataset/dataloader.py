import os
import cv2
import torch


class Dataloader():
    def __init__(self, data_dir, batch_size):
        self.type_table = ['left', 'right', 'straight', 'up']
        self.batch_size = batch_size
        self.data_path = data_dir
        self.get_imgs_labels()

    def get_imgs_labels(self):
        self.img_list = list()
        self.label_list = list()
        img_n_list = os.listdir(self.data_path)
        for img_n in img_n_list:
            real_img_n = os.path.join(self.data_path, img_n)
            img = cv2.imread(real_img_n)
            img = torch.tensor(img)
            img = img.permute(2, 0, 1)

            name_split = img_n.split('_')
            label = self.type_table.index(name_split[1])

            self.img_list.append(img)
            self.label_list.append(label)

    def __getitem__(self, index):
        batch_data, batch_label = list(), list()
        num = len(self.img_list)
        batch, last = divmod(num, self.batch_size)
        for k in range(batch):
            batch_data.append(torch.stack(self.img_list[k*self.batch_size:(k+1)*self.batch_size], dim=0))
            batch_label.append(torch.tensor(self.label_list[k*self.batch_size:(k+1)*self.batch_size], dtype=torch.long))
        if last:
            batch_data.append(torch.stack(self.img_list[(k+1)*self.batch_size:], dim=0))
            batch_label.append(torch.tensor(self.label_list[(k+1)*self.batch_size:], dtype=torch.long))
        return batch_data[index], batch_label[index]

    def __len__(self):
        return len(self.img_list)