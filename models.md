# Models
(Including algorithm name, parameter setting of each model)

## Simple Convolutional Models (LeNet-5-like convolutional)

- input   	1@ 28x28
- filter  	32@ 5x5
  - :one: conv   	32@ 24x24
  - :two: pooling  32@ 12x12
- filter  	64@ 5x5
  - :three: conv   	64@ 8x8
  - :four: pooling  64@ 4x4
  - :five: fc1 		64  -> 512
- add a 50% dropout (during training only.) Dropout also scales
  - :six: fc2 		512 -> 10

Bias and rectified linear non-linearity
