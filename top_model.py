from keras.models import Sequential, Model
from keras.layers import Activation, Dropout, Flatten, Dense

from keras import backend as K
K.set_image_dim_ordering('th')

import os
import numpy as np
import h5py

#загружаем наши датасеты в виде нампи массивов
#при загрузке изображений в генератор для аугментации мы не использовали shuffle -> они не перемешались -> у нас в
# train_data загрузятся из 2 папок в img_train последовательно сначала 100 изображений 1 класса (т.е. птицы) и мы можем смело указать им 0 в качестве меток
# и 100 изображений класса 2 (т.е. самолеты) и мы укажем им метку 1
train_data = np.load(open('bottleneck_features/bn_features_train.npy', 'rb'))
train_labels = np.array([0] * 1000 + [1] * 1000)

validation_data = np.load(open('bottleneck_features/bn_features_validation.npy', 'rb'))
validation_labels = np.array([0] * 1000 + [1] * 1000)

#создаем верхний слой модели(?), в оригинальном пейпаере слои с 256 нейронами, но используем тут 2 слова по 64
#для того чтобы мой суперкомпьютер не напрегался
#и дропаут в 0.5 для борьбы с переобучением в условиях ограниченной выборки (но может уже и не такой ограниченной, аугментация топ)
def fc_model():
    fc_model = Sequential()
    fc_model.add(Flatten(input_shape=train_data.shape[1:]))
    fc_model.add(Dense(64, activation='relu', name='dense_one'))
    fc_model.add(Dropout(0.5, name='dropout_one'))
    fc_model.add(Dense(64, activation='relu', name='dense_two'))
    fc_model.add(Dropout(0.5, name='dropout_two'))
    fc_model.add(Dense(1, activation='sigmoid', name='output'))

    return fc_model

print(' ')
print('-'*50)
print('''
  ___                  _   _       __   ______  ___ _
 |_ _|_ _  __ ___ _ __| |_(_)___ _ \ \ / /__ / | _ |_)_ _  __ _ _ _ _  _
  | || ' \/ _/ -_) '_ \  _| / _ \ ' \ V / |_ \ | _ \ | ' \/ _` | '_| || |
 |___|_||_\__\___| .__/\__|_\___/_||_\_/ |___/ |___/_|_||_\__,_|_|  \_, |
                 |_|                                                |__/
        ''')
print('Step_2')
print('Training FFN model with npy arrays saved in last step')
print('-'*50)
epochs=int(input('How much epochs we need?:'))
print('-'*50)
print('Fitting train and val .npy arrays to top_model...')

model=fc_model()
model.compile(optimizer='rmsprop', loss='binary_crossentropy', metrics=['accuracy'])
model.fit(train_data, train_labels, nb_epoch=epochs, batch_size=32, validation_data=(validation_data, validation_labels))

print('Training the model has been completed')
print('-'*50)

if not os.path.exists('weights/top_model_weights/'):
    os.makedirs('weights/top_model_weights/')
    print('Directory "weights/top_model_weights/" has been created')

print('Saving weights to weights/top_model_weights/fc_inception_cats_dogs_250.hdf5')
model.save_weights('weights/top_model_weights/fc_inception_cats_dogs_250.hdf5')

print('Finished')
print('-'*50)
loss, accuracy = model.evaluate(validation_data, validation_labels)
print("\nLoss: %.2f, Accuracy: %.2f%%" % (loss, accuracy*100))
