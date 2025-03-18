# PottsMGNet

This is the Pytorch code for "PottsMGNet: A mathematical explanation of encoder-decoder based neural networks" by Xue-Cheng Tai, Hao Liu and Raymond Chan.

If this code is useful, pleae cite our paper
https://epubs.siam.org/doi/abs/10.1137/23M1586355

Copyright (c) Hao Liu (haoliu AT hkbu.edu.hk)
Department of Mathematics,
Hong Kong Baptist University,
Kowloon, Hong Kong
https://www.math.hkbu.edu.hk/~haoliu/

Permission is hereby granted, free of charge, to any person obtaining a copy of this software and associated documentation files (the "Software"), to deal in the Software without restriction, subject to the following conditions:

The above copyright notice and this permission notice shall be included in all copies or substantial portions of the Software. The Software is provided "as is", without warranty of any kind.

Datasets used in our paper:<br />
DSB2018: https://www.kaggle.com/competitions/data-science-bowl-2018/overview <br />
MSRA10K: https://mmcheng.net/msra10k/

**Demo**:<br />
train_new.py implements the progressive training.<br />
The dataloader in this file is for the MSRA10K dataset.

After changing the data dictionaries in train_new.py to your own ones, you can run train_new.py to train the model.



