from keras.preprocessing.image import ImageDataGenerator
from keras.models import Model
from keras.applications.inception_v3 import InceptionV3
from keras.layers import Dropout, Flatten, Dense
from keras.callbacks import ModelCheckpoint
from keras.optimizers import SGD

from keras import backend as K
K.set_image_dim_ordering = 'th'

import os
import numpy as np
import h5py

weights_filename='weights/complete_model_checkpoint_weights/weights-improvement-11-0.92.hdf5'

def complete_model(weights_path=weights_filename):
#инсепшен - нижний сверточный слой:
#include_top - вкл/выкл верхной части сети инсепшена с 1000 выходов - нам не надо, у нас будет классификация
#weights - загрузка весов обученных на ImageNet, нам не надо(или надо, без них обучиться походу будет нереально)
#input_shape - настраиваем под наши изображения
    inc_model=InceptionV3(include_top=False,
                          weights='imagenet',
                          input_shape=((150, 150, 3)))
#верхний слой, ну тут как бы все понятно, в конце top_model - указываем 2 выхода в последнем, чтоб аутпут в виде 2 классов
    x = Flatten()(inc_model.output)
    x = Dense(64, activation='relu', name='dense_one')(x)
    x = Dropout(0.5, name='dropout_one')(x)
    x = Dense(64, activation='relu', name='dense_two')(x)
    x = Dropout(0.5, name='dropout_two')(x)
    top_model=Dense(1, activation='sigmoid', name='output')(x)
    model = Model(inputs=inc_model.input, outputs=top_model)
#здесь подгрузим ранее обученные веса полученные при 1000 изображениях для дообечения  с аугментированными данными, пока закоментим
    model.load_weights(weights_filename, by_name=True)

    for layer in inc_model.layers[:205]:
         layer.trainable = False


    return model
if __name__ == "__main__":
    print(' ')
    print('-'*50)
    print('''
Мне нужна твоя одежда, человек
    ''')
    print('Step_3')
    print('Training complete model with images')
    print('-'*50)

#во входных параметрах генератора указываем свойства аугментации, мне нужно зеркально по горизонтале
#рескейл нужен для масштабирования данных, чтоб все пиксели к единообразую пон?
# добавим horizontal_flip=True для горизонт. аугмент.
    train_datagen = ImageDataGenerator(
            rescale=1./255,
            horizontal_flip=True)

    test_datagen = ImageDataGenerator(rescale=1./255)

#загружаем в генератор пикчи, аугментируем, получаем кучу аугментированных пиктч с метками(получаем их легко и просто
#блягодаря class_mode='binary'
    train_generator = train_datagen.flow_from_directory(
            'data/img_train/',
            target_size=(150, 150),
            batch_size=32,
            class_mode='binary')

    validation_generator = test_datagen.flow_from_directory(
            'data/img_val/',
            target_size=(150, 150),
            batch_size=32,
            class_mode='binary')

    pred_generator=test_datagen.flow_from_directory('data/img_val/',
                                                         target_size=(150, 150),
                                                         batch_size=100,
                                                         class_mode='binary')

    epochs=int(input('How much epochs we need?:'))

    filepath="weights/complete_model_checkpoint_weights/weights-improvement-{epoch:02d}-{val_accuracy:.2f}.hdf5"

    if not os.path.exists('weights/complete_model_checkpoint_weights/'):
        os.makedirs('weights/complete_model_checkpoint_weights/')
        print('Directory "weights/complete_model_checkpoint_weights/" has been created')

    #сохраняем веса с наибольшей точностью на тестовой выборке
    checkpoint = ModelCheckpoint(filepath, monitor='val_accuracy', verbose=1, save_best_only=True, mode='max')
    callbacks_list = [checkpoint]

    model=complete_model()
    print('-'*50)
    print (
    ''' Compiling model with:
    - stochastic gradient descend optimizer;
    - learning rate=0.0001;
    - momentum=0.9 ''')

    model.compile(loss='binary_crossentropy',
              optimizer=SGD(learning_rate=1e-4, momentum=0.9),
                #optimizer='rmsprop',
              metrics=['accuracy'])
    print(' Compiling has been finished')
    print('-'*50)
    print ('Training the model...')
    print ('A half of Cristmas tree is coming :)')

    model.fit(
        train_generator,
        epochs=epochs,
        validation_data=validation_generator,
        callbacks=callbacks_list,
        verbose=1)

    print('-'*50)
    print('Training the model has been completed')

#здесь указываем валидациолнную выборку для оценки качества (указываем с аугментацией и без, позже)
    loss, accuracy = model.evaluate(validation_generator)
    print("\nLoss: %.2f, Accuracy: %.2f%%" % (loss, accuracy*100))
