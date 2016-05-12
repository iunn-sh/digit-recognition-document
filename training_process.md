# Training Process
(CV? Feature importance? Anything worth to mention during this work.)

- Training computation: logits + cross-entropy loss.
- Use simple momentum for the optimization of minimize loss.
- softmax regression

1. each traing batch size 50
2. training steps = 40 * 10000 / 50
3. evaluate error rate and learning rate with batch size 64 when every 100 training