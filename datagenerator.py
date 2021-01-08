import os
import tensorflow as tf
import pandas as pd
import numpy as np
from utils import split_data

class DataGenerator(object):
    def __init__(self, data_dir, images_dir, csv_file, batch_size, input_shape, img_ext, augment=False, label_col_id="label", image_col_id="image_id"):
        self.data_dir = data_dir
        self.images_dir = os.path.join(self.data_dir, images_dir)
        self.csv_file = csv_file
        self.batch_size = batch_size
        self.input_shape = input_shape
        self.img_ext = img_ext
        self.augment = augment
        self.label_col_id = label_col_id
        self.image_col_id = image_col_id

        self.dataframe = pd.read_csv(os.path.join(self.data_dir, self.csv_file))
        self.class_list = sorted(list(np.unique(self.dataframe[self.label_col_id])))
        self.num_classes = len(self.class_list)
        self.train_data_dict, self.val_data_dict = split_data(self.dataframe, self.label_col_id, self.image_col_id, self.images_dir)
        self.total_train_images = len(self.train_data_dict[image_col_id])
        self.total_val_images = len(self.val_data_dict[image_col_id])

        print('+'*15, 'Data Summary', '+'*15)
        print("Class List : {}".format(self.class_list))
        print("Total {} Images and {} Labels Found".format(len(self.dataframe[self.image_col_id]), len(self.dataframe[self.label_col_id])))
        print("Found total {} Images and {} Labes for Training".format(len(self.train_data_dict[self.image_col_id]), len(self.train_data_dict[self.label_col_id])))
        print("Found total {} Images and {} Labes for Validation".format(len(self.val_data_dict[self.image_col_id]), len(self.val_data_dict[self.label_col_id])))

    def augment(self, image, label):
        image = tf.image.random_flip_up_down(image, seed=None)
        image = tf.image.random_flip_left_right(image, seed=None)
        image = tf.image.random_brightness(image, max_delta=0.5)
        image = tf.image.rot90(image, k=tf.random.uniform(shape=[], minval=0, maxval=3, dtype=tf.int32))
        image = tf.image.random_crop(image, size=self.input_shape)
        image = tf.clip_by_value(image, 0, 1)
        return image, label

    def get_img_file(self, img_path):
        image = tf.io.read_file(img_path)
        if self.img_ext == 'png':
            image = tf.image.decode_png(image, channels=3)
        elif self.img_ext == 'jpg':
            image = tf.image.decode_jpeg(image, channels=3)
        image = tf.image.resize(image, [self.input_shape[0], self.input_shape[1]], antialias=True)
        image = tf.cast(image, tf.float32)/255.0
        return image

    def get_label(self, cl):
        if self.num_classes > 2:
            one_hot = tf.one_hot(cl, self.num_classes)
        else:
            one_hot = cl
        return one_hot

    def parse_function(self, ip_dict):
        label = self.get_label(cl=ip_dict[self.label_col_id])
        image = self.get_img_file(img_path=ip_dict[self.image_col_id])
        return image, label

    def get_dataset(self, mode='train'):
        with tf.device('/cpu:0'):
            if mode=='train':
                dataset = tf.data.Dataset.from_tensor_slices(self.train_data_dict)
                dataset = dataset.shuffle(self.total_train_images)
            elif mode=='val':
                dataset = tf.data.Dataset.from_tensor_slices(self.val_data_dict)
                dataset = dataset.shuffle(self.total_val_images)
            else:
                raise Exception("Enter Valid Mode : 'train' or 'val'")
            dataset = dataset.map(self.parse_function, num_parallel_calls=tf.data.experimental.AUTOTUNE)
            if self.augment:
                dataset = dataset.map(self.augment, num_parallel_calls=tf.data.experimental.AUTOTUNE)
            dataset = dataset.batch(self.batch_size)
            dataset = dataset.prefetch(buffer_size=1)
        return dataset
