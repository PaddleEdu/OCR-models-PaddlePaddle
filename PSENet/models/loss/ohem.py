import paddle
import numpy as np
from paddle.fluid.layers import where

def ohem_single(score, gt_text, training_mask):
    gt_part = paddle.cast(gt_text > 0.5, dtype='float32')
    gt_tr_part = paddle.cast(paddle.logical_and(gt_text > 0.5, training_mask <= 0.5), dtype='float32')
    pos_num = int(paddle.sum(gt_part)) - int(paddle.sum(gt_tr_part))
    #pos_num = int(np.sum(gt_text.numpy() > 0.5)) - int(np.sum((gt_text.numpy() > 0.5) & (training_mask.numpy() <= 0.5)))
    #pos_num = int(paddle.sum(gt_text > 0.5)) - int(paddle.sum((gt_text > 0.5) & (training_mask <= 0.5)))
    if pos_num == 0:
        # selected_mask = gt_text.copy() * 0 # may be not good
        selected_mask = training_mask
        selected_mask = paddle.reshape(selected_mask,(1, selected_mask.shape[0], selected_mask.shape[1]))
        selected_mask = paddle.cast(selected_mask, dtype='float32')
        return selected_mask

    neg_num = int(np.sum(gt_text.numpy() <= 0.5))
    neg_num = int(min(pos_num * 3, neg_num))

    if neg_num == 0:
        selected_mask = training_mask
        # selected_mask = selected_mask.view(1, selected_mask.shape[0], selected_mask.shape[1]).float()
        selected_mask = paddle.reshape(selected_mask, (1, selected_mask.shape[0], selected_mask.shape[1]))
        selected_mask = paddle.cast(selected_mask, dtype='float32')
        return selected_mask

    #TODO: neg_score = score[gt_text <= 0.5]
    # neg_score_list = []
    # print(where(gt_text > 0.99).shape)
    # for i in where(gt_text <= 0.5).numpy():
    #     neg_score_list.append(score[int(i[0])][int(i[1])])
    # neg_score = paddle.concat(neg_score_list, axis=0, name=None)
    #print(neg_score)

    gt_text_flatten = paddle.reshape(gt_text,(-1,))
    index = where(gt_text_flatten <= 0.5)
    index = paddle.reshape(index,(1,-1))
    score_flatten = paddle.reshape(score,(1,-1))
    neg_score = paddle.index_sample(score_flatten, index)
    neg_score = paddle.reshape(neg_score,(-1,))
    

    neg_score_sorted = paddle.sort(-neg_score)
    threshold = -neg_score_sorted[neg_num - 1]

    # TODO: selected_mask = ((score >= threshold) | (gt_text > 0.5)) & (training_mask > 0.5)
    item1 = paddle.logical_or(score >= threshold, gt_text > 0.5)
    selected_mask = paddle.logical_and(item1, training_mask > 0.5)
    # selected_mask = selected_mask.reshape(1, selected_mask.shape[0], selected_mask.shape[1]).float()
    selected_mask = paddle.reshape(selected_mask, (1, selected_mask.shape[0], selected_mask.shape[1]))
    #selected_mask = selected_mask.reshape(1, selected_mask.shape[0], selected_mask.shape[1])
    selected_mask = paddle.cast(selected_mask, dtype='float32')
    return selected_mask

def ohem_batch(scores, gt_texts, training_masks):
    selected_masks = []
    for i in range(scores.shape[0]):
        selected_masks.append(ohem_single(scores[i, :, :], gt_texts[i, :, :], training_masks[i, :, :]))

    selected_masks = paddle.concat(selected_masks, axis=0, name=None)#.float()
    selected_masks = paddle.cast(selected_masks, dtype='float32')
    return selected_masks
