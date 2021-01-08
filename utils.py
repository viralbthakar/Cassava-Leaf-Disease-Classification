import os
import random
import numpy as np
import matplotlib.pyplot as plt

def split_data(dataframe, label_col_id, image_col_id, images_path, split_index=0.8):
    train_dict = {image_col_id:[], label_col_id:[]}
    val_dict = {image_col_id:[], label_col_id:[]}
    class_list = sorted(list(np.unique(dataframe[label_col_id])))
    grouped = dataframe.groupby(label_col_id)
    for i, class_id in enumerate(class_list):
        print("Splitting Data for {} Class : Assigning Label {}".format(class_id, i))
        all_images = list(grouped.get_group(class_id).sample(frac = 1)[image_col_id])
        random.shuffle(all_images)
        train_images = [os.path.join(images_path, l) for l in all_images[:int(len(all_images)*split_index)]]
        val_images = [os.path.join(images_path, l) for l in all_images[int(len(all_images)*split_index):]]
        train_dict[image_col_id].extend(train_images)
        val_dict[image_col_id].extend(val_images)
        train_dict[label_col_id].extend([i for l in range(len(train_images))])
        val_dict[label_col_id].extend([i for l in range(len(val_images))])
        print("Found Total {} Images under {} Class".format(len(all_images), class_id))
        print("Splitted {} Images for Training and {} Images for Validation".format(len(train_images), len(val_images)))
    return train_dict, val_dict

def show_batch(image_batch, label_batch, plt_title, rows=2, cols=2):
    num_images_to_show = rows * cols
    plt.figure(figsize=(8,8))
    for n in range(num_images_to_show):
        ax = plt.subplot(rows, cols, n+1)
        plt.imshow(image_batch.numpy()[n])
        plt.title(str(label_batch.numpy()[n]))
        plt.axis('off')
    plt.suptitle(plt_title, fontsize=14)
    plt.show()






