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

## how to view each node

```json
    # add grad for internal layer
    #outnames = pred.list_outputs()
    #argnames = pred.list_arguments()

    #intsyms = pred.get_internals()
    #sym_group = [pred]
    #for item in intsyms.list_outputs():
    #    if item not in outnames and item not in argnames:
    #        sym_group.append(mx.symbol.BlockGrad(intsyms[item], name=item))
    #pred = mx.symbol.Group(sym_group)
```

1. define custom op 
    mx.symbol.Custom(data=pred, label=label_weight, name='final_logistic', op_type='MyLogistic')

../../python/mxnet/operator.py:614

## BUG 
1. bucketingModule.install_monitor 
    always install on the default bucket module, not current module

1. bucketingModule 
    rescale_grad/batch_size: batch_size always be the default bucket key

## TODO 
### 验证
1. test验证
1. negative sample 是否重复的影响
1. bucketing 中batch_size的影响

### 代码调整
1. lnz的影响, 作为参数
1. bucketing 自动生成
1. learing rate 可调(查阅文档)
1. nan

### 加速优化
1. NTC调整为TNC     
1. 使用C++实现NceOutput 
