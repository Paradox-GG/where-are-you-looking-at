import os
import shutil
import random


def split_into_4types(dataset_root, go_to):
    dir_list = os.listdir(dataset_root)
    for dir in dir_list:
        real_dir = os.path.join(dataset_root, dir)
        for img_n in os.listdir(real_dir):
            print(img_n)
            name_split = img_n.split('_')

            belong_to_path = os.path.join(go_to, name_split[1])
            if not os.path.exists(belong_to_path):
                os.makedirs(belong_to_path)

            load_from = os.path.join(real_dir, img_n)
            save_to = os.path.join(belong_to_path, img_n)
            shutil.copyfile(load_from, save_to)


# # ==============================================data split======================================================
def split_into_train_eval(root_dir, to_dir_1, to_dir_2):
    dir_list = os.listdir(root_dir)
    random.seed(66)
    for dir in dir_list:
        real_dir = os.path.join(root_dir, dir)
        current_img_list = os.listdir(real_dir)
        random.shuffle(current_img_list)
        test_ids = current_img_list[:len(current_img_list)//5]
        train_ids = current_img_list[len(current_img_list)//5:]

        for img_id in test_ids:    # 先选一部分到2号
            real_img_id = os.path.join(real_dir, img_id)
            # print(real_img_id, os.path.join(to_dir_2, img_id))
            shutil.move(real_img_id, os.path.join(to_dir_2, img_id))

        for img_id in train_ids:         # 移走剩下的到1号
            real_img_id = os.path.join(real_dir, img_id)
            # print(real_img_id, os.path.join(to_dir_2, img_id))
            shutil.move(real_img_id, os.path.join(to_dir_1, img_id))


def main():
    dataset_root = ''  # todo: download from: https://archive.ics.uci.edu/ml/machine-learning-databases/faces-mld/faces_4.tar.gz
    go_to = ''  # create a temp folder you like
    split_into_4types(dataset_root, go_to)

    train_dir = ''
    eval_dir = ''
    split_into_train_eval(go_to, train_dir, eval_dir)


if __name__ == '__main__':
    main()




