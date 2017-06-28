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


../../python/mxnet/operator.py:614

## how to run lstm on cpu 
    if not args.stack_rnn:
        stack = mx.rnn.FusedRNNCell(args.num_hidden, num_layers=args.num_layers,
                mode='lstm', bidirectional=args.bidirectional).unfuse()
    else:
        stack = mx.rnn.SequentialRNNCell()
        for i in range(args.num_layers):
            cell = mx.rnn.LSTMCell(num_hidden=args.num_hidden, prefix='lstm_l%d_'%i)
            stack.add(cell)

## BUG 
1. bucketingModule.install_monitor 
    always install on the default bucket module, not current module

1. bucketingModule 
    rescale_grad/batch_size: batch_size always be the default bucket key

## TODO 
### 验证
1. test验证 *DONE*
1. dropout的影响 
1. momentom的作用
1. negative sample 是否重复的影响
1. bucketing 中batch_size的影响
1. 使用自动生成的bucket影响性能
    - 下一个batch的NCE比上一个batch最后的值要差很多, 使用[10..80..10]没有这么显著
    - 学习效果变差
    - 会出现nan
    - 限制最小长度为10， 问题依旧
    - 影响到了梯度!!!!

    有一组句子s1,s2, ..., s_n, l1, lw, ..., l_n, batch_size为batch, 从l1 .. l_n中选一组bucket，使得
    丢弃的句子最少，同时padding尽可能少
    长度为: [l1, ln]
    cnt:    [c1, cn]
    batch_size

    假设l>idx的已经确定, 接下来从l<idx中选择一个
    1. 作为bucket:   
    2. 不作为

    a[i][j]: j是bucket，i,j之间的值都不是bucket，i是bucket, j<i
    a[i][i]: i是bucket， j<i都不是bucket

    sumbuf[i][j] = cnt[i] + cnt[i+1] + ... + cnt[j-1] 
    sumbuf[0][0] = cnt[0] 
    sumbuf[0][1] = cnt[0]+cnt[1] = sumbuf[0][0] + cnt[1]
    sumbuf[0][2] = cnt[0]+cnt[1]+cnt[2] = sumbuf[0][1]+cnt[2]
    sumbuf[1][1] = cnt[1]
    sumbuf[1][2] = cnt[1]+cnt[2] = sumbuf[1][1] + cnt[2]

    throw:

    a[i][j] = a[j][*] + cnt[i:j]

    loss: 
    

    

### 代码调整
1. lnz的影响, 作为参数  *DONE*
1. bucketing 自动生成   *DONE*
1. learing rate 可调(查阅文档) *DONE*
    效果不是很好

1. 修改alllab的输入方式
1. vocab中不要引入padding, 导致vocab_size增大
1. save vocab to model 
1. 使用ppl作为valid_metric

### 加速优化
1. NTC调整为TNC     
1. 使用C++实现NceOutput 
