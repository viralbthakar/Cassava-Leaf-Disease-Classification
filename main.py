import numpy as np
from datagenerator import DataGenerator
from utils import show_batch


DATA_DIR = "./data"
IMG_DIR = "train_images"
DATA_CSV = "train.csv"
INPUT_SHAPE = (256, 256, 3)
BATCH_SIZE = 16
EXT = 'jpg'

datagen = DataGenerator(data_dir=DATA_DIR,
    images_dir=IMG_DIR,
    csv_file=DATA_CSV,
    batch_size=BATCH_SIZE,
    input_shape=INPUT_SHAPE,
    img_ext=EXT)

train_datagen = datagen.get_dataset(mode='train')
val_datagen = datagen.get_dataset(mode='val')

#Show One Batch
image_batch, label_batch = next(iter(train_datagen))
print("Input Batch Shape : {}".format(image_batch.numpy().shape))
print("Label Batch Shape : {}".format(label_batch.numpy().shape))
show_batch(image_batch, label_batch, plt_title=datagen.class_list, rows=4, cols=4)
print(np.amax(image_batch.numpy()))
print(np.amin(image_batch.numpy()))
print(label_batch.numpy())


