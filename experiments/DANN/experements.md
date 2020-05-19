Обучал сплитя датасеты 0.8/0.1/0.1. В таблице метрика посчитанная по всему таргет датасету.

Model | A → W | D→ W | W→ D	| A→ D | D → A | W→ A 
--- | --- | --- | --- | --- | --- | --- |
DANN Alexnet Статья | 0.73 |	0.964 | 0.992||||
DANN Resnet Обзор | 0.826 | 0.978 | 1 | 0.833 | 0.668	| 0.661 |
ResNet 141 freeze with domain loss | 0.71484| - | - | 0.62708 | 0.63246 | 0.67543 |
ResNet 141 freeze without domain loss | 0.77995 |	0.96875 | 0.99583 | 0.79583 | 0.63565 | 0.63175 |
ResNet 129 freeze without domain loss | 0.7849 | - | - | - | - | - |
ResNet 72 freeze without domain loss | 0.7348 | - | - | - | - | - |
ResNet 0 freeze without domain loss | 0.5443 | - | - | - | - | - |


Тут max value за обучение

Model Max Value | A → W | D→ W | W→ D	| A→ D | D → A | W→ A 
--- | --- | --- | --- | --- | --- | --- |
DANN Alexnet Статья | 0.73 |	0.964 | 0.992||||
DANN Resnet Обзор | 0.826 | 0.978 | 1 | 0.833 | 0.668	| 0.661 |
ResNet 141 freeze with domain loss | 0.8354 | - | - | 0.7551 | 0.6583 | 0.6903 |
ResNet 141 freeze without domain loss | 0.8101 | 0.9620 | 1 | 0.7959 | 0.6583 | 0.6085 |
ResNet 129 freeze without domain loss |0.8354| - | - | - | - | - |
ResNet 72 freeze without domain loss |0.8101| - | - | - | - | - |
ResNet 0 freeze without domain loss |0.6075| - | - | - | - | - |


Описание обучения и модели(Experement 1):
Resnet50
classificator vanila_dann
141 layer freeze
200 epoch
10 steps per epoch
64 bs

Model | A → W | D→ W | W→ D	| A→ D | D → A | W→ A 
--- | --- | --- | --- | --- | --- | --- |
DANN Alexnet Статья | 0.73 |	0.964 | 0.992||||
DANN Resnet Обзор | 0.826 | 0.978 | 1 | 0.833 | 0.668	| 0.661 |
ResNet with domain loss    | 0.8451 | 0.9349 | 0.9754 | 0.7969 | 0.6719 | 0.6893 |
ResNet without domain loss | 0.7604 | 0.9674 | 0.9911 | 0.7924 | 0.6403 | 0.6317 |


Описание обучения и модели(Experement 2):
Resnet50
classificator vanila_dann
141 layer freeze
200 epoch
10 steps per epoch
32 bs

Model | A → W | D→ W | W→ D	| A→ D | D → A | W→ A 
--- | --- | --- | --- | --- | --- | --- |
DANN Alexnet Статья | 0.73 |	0.964 | 0.992||||
DANN Resnet Обзор | 0.826 | 0.978 | 1 | 0.833 | 0.668	| 0.661 |
ResNet with domain loss    | 0.8216 | 0.9076 | 0.9792 | 0.7146 | 0.6552 | 0.6882 |
ResNet without domain loss | 0.7578 | 0.9648 | 0.9979 | 0.7833 | 0.6342 | 0.6374 |


Описание обучения и модели(Experement 3):
Resnet50
classificator simple
141 layer freeze
200 epoch
20 steps per epoch
32 bs
max_value

Model | A → W | D→ W | W→ D	| A→ D | D → A | W→ A 
--- | --- | --- | --- | --- | --- | --- |
DANN Alexnet Статья | 0.73 |	0.964 | 0.992||||
DANN Resnet Обзор | 0.826 | 0.978 | 1 | 0.833 | 0.668	| 0.661 |
ResNet with domain loss    | 0.8354 | - | - | 0.7551 | 0.6583 | 0.6903 |
ResNet without domain loss | 0.8101 | 0.9620 | 1 | 0.7959 | 0.6583 | 0.67543 |


Описание обучения и модели(Experement 4):
Resnet50
classificator simple
141 layer freeze
200 epoch
20 steps per epoch
32 bs

Model | A → W | D→ W | W→ D	| A→ D | D → A | W→ A 
--- | --- | --- | --- | --- | --- | --- |
DANN Alexnet Статья | 0.73 |	0.964 | 0.992||||
DANN Resnet Обзор | 0.826 | 0.978 | 1 | 0.833 | 0.668	| 0.661 |
ResNet with domain loss | 0.71484| - | - | 0.62708 | 0.63246 | 0.67543 |
ResNet without domain loss    | 0.77995 |	0.96875 | 0.99583 | 0.79583 | 0.63565 | 0.63175 |


Описание обучения и модели(Experement 5):
Resnet50
classificator test5
141 layer freeze
200 epoch
20 steps per epoch
32 bs

Model | A → W | D→ W | W→ D	| A→ D | D → A | W→ A 
--- | --- | --- | --- | --- | --- | --- |
DANN Alexnet Статья | 0.73 |	0.964 | 0.992||||
DANN Resnet Обзор | 0.826 | 0.978 | 1 | 0.833 | 0.668	| 0.661 |
ResNet with domain loss | 0.8503 | 0.9154 | 0.9263 | 0.7902 | 0.6708 | 0.7109 |