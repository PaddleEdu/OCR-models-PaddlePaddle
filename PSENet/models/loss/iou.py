import paddle
from paddle.fluid.layers import where

EPS = 1e-6

def iou_single(a, b, mask, n_class):
    valid = mask == 1
    # a = a[valid]
    # b = b[valid]


    valid_flatten = paddle.reshape(valid,(-1,))
    valid_flatten = paddle.cast(valid_flatten,dtype="int32")
    index = where(valid_flatten == 1)
    if index.shape[0] == 0:
        return paddle.zeros((1,))
    
    index = paddle.reshape(index,(1,-1))
    a_flatten = paddle.reshape(a,(1,-1))
    a = paddle.index_sample(a_flatten, index)
    a = paddle.reshape(a,(-1,))

    b_flatten = paddle.reshape(b,(1,-1))
    b = paddle.index_sample(b_flatten, index)
    b = paddle.reshape(b,(-1,))

    miou = []
    for i in range(n_class):
        inter = paddle.logical_and(a == i,b == i)
        inter = paddle.cast(inter, dtype='float32')
        union = paddle.logical_or(a == i,b == i)
        union = paddle.cast(union, dtype='float32')

        miou.append(paddle.sum(inter) / (paddle.sum(union) + EPS))
    miou = sum(miou) / len(miou)
    return miou

def iou(a, b, mask, n_class=2, reduce=True):
    batch_size = a.shape[0]

    a = paddle.reshape(a, (batch_size, -1))
    b = paddle.reshape(b, (batch_size, -1))
    mask = paddle.reshape(mask, (batch_size, -1))
    # a = a.view(batch_size, -1)
    # b = b.view(batch_size, -1)
    # mask = mask.view(batch_size, -1)

    # iou = a.new_zeros((batch_size,), dtype=torch.float32)
    iou = paddle.zeros((batch_size,), dtype='float32')
    for i in range(batch_size):
        iou[i] = iou_single(a[i], b[i], mask[i], n_class)

    if reduce:
        iou = paddle.mean(iou)
    return iou