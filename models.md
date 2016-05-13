# Models
(Including algorithm name, parameter setting of each model)

## Simple Convolutional Models (LeNet-5-like convolutional)

- input   	1@ 28x28
- filter  	32@ 5x5
  - 1) conv   	32@ 24x24
  - 2) pooling  32@ 12x12
- filter  	64@ 5x5
  - 3) conv   	64@ 8x8
  - 4) pooling  64@ 4x4
  - 5) fc1 		64  -> 512
- add a 50% dropout (during training only.) Dropout also scales
  - 6) fc2 		512 -> 10

Bias and rectified linear non-linearity
