from PIL import Image
import os, sys

#ресайзер для изображений, сверточным слоям инспетрона необходимо разрешение изображений минимум 75х75, поэтому если вдруг то вот
path_load = ('E:/LEARN/Keras_IncepnionV3_binary_classification/data/img_train_non_resize/Bird/')
path_save = ('E:/LEARN/Keras_IncepnionV3_binary_classification/data/img_train/Bird/')
dirs = os.listdir(path_load)

def resize():
    for item in dirs:
        if os.path.isfile(path_load + item):
            print(item)
            im = Image.open(path_load + item)
            f, e = os.path.splitext(item)
            imResize = im.resize((150,150), Image.LANCZOS)
            imResize.save(path_save + item, 'PNG', quality=90)

if __name__ == "__main__":
    resize()