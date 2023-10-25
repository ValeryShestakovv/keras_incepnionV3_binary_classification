# keras_incepnionV3_binary_classification

Алгоритм получается такой:
1) Подготавливаем изображение 2 классов (самолетики, птички). 
Если берешь из CIFAR-10, то они там 32х32 пикселя, а для инсептион слоя надо 75х75, поэтому используй "маленький ресайзер"
2) Сначала обучаем на 200 изображениях (100 птиц 100 самолетов), закоменти аугментацию и подгрузку весов в модель во время обучения(приходится после)
3) Потом обучай на 2000 изображениях (аналогично)
4) Потому обучай на 2000 + добавляйпше  аугментацию нужную тебе + запускай загрузку весов в модель
6) Визуализируй в визуализейтион.пи
7) Профит

Датасет СИФАР-10 чтобы жестко скачать руками:
https://www.kaggle.com/code/fanbyprinciple/cifar10-explanation-with-pytorch/input
Последние актуальные веса для самолетиков и птичек:
[https://dropmefiles.com/gX7Nk](https://drive.google.com/file/d/1rhc0H5IrAsCRhag6UQj5cLIMHvhaRYlX/view?usp=sharing)
Заресайзеные пикчи:
[https://dropmefiles.com/XiwEj
](https://drive.google.com/file/d/1jGEcZSnmTruygm20Yz9fIIGwdahSFHM3/view?usp=sharing)
Итоговые результаты:
| 200img/50epochs  | 2000img/50epochs | 2000img+augment/50epochs |
| -----------------| -----------------| -------------------------|
|    Loss: 0.39    |   Loss: 0.25     |         Loss: 0.15       |
| Accuracy: 84.00% | Accuracy: 0.91   |       Accuracy: 0.96     |


![200 изображений, 50 эпох](https://github.com/ValeryShestakovv/keras_incepnionV3_binary_classification/assets/57315830/d8d0c2bd-3a65-45f4-a072-b75844ebe5bd)
