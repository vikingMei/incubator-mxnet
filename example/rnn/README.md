RNN Example
===========
This folder contains RNN examples using high level mxnet.rnn interface.

Examples using low level symbol interface have been deprecated and moved to old/

## Data
Run `get_ptb_data.sh` to download PenTreeBank data.

## Python

- [lstm_bucketing.py](lstm_bucketing.py) PennTreeBank language model by using LSTM

Performance Note:
More ```MXNET_GPU_WORKER_NTHREADS``` may lead to better performance. For setting ```MXNET_GPU_WORKER_NTHREADS```, please refer to [Environment Variables](https://mxnet.readthedocs.org/en/latest/how_to/env_var.html).


## multi-gpu
DataDesc.layout: NCHW
   get_batch_axis: 获取batch对应的维度 
然后，根据gpu的数目，在batch这个维度上进行切分

问题:
    1. 数据输入格式: TN
    2. DataDecs.layout: NCHW, 会按照第一个维度切分, 与实际不符
    具体代码见: mxnet/module/executor_group.py, line 300左右
