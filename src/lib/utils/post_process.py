from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import numpy as np
from .image import transform_preds


def ctdet_post_process(dets, c, s, h, w, num_classes):
  # dets: batch x max_dets x dim
  # return 1-based class det dict
  ret = []
  for i in range(dets.shape[0]):
    top_preds = {}
    dets[i, :, :2] = transform_preds(
          dets[i, :, 0:2], c[i], s[i], (w, h))
    dets[i, :, 2:4] = transform_preds(
          dets[i, :, 2:4], c[i], s[i], (w, h))
    classes = dets[i, :, -1]
    for j in range(num_classes):
      inds = (classes == j)
      top_preds[j + 1] = np.concatenate([
        dets[i, inds, :4].astype(np.float32),
        dets[i, inds, 4:5].astype(np.float32)], axis=1).tolist()
    ret.append(top_preds)
  return ret

def reback_ctdet_post_process(dets, c, s, h, w, h_feature, w_feature):
  # dets, 跟踪的tlbr
  # return 特征图中的inds
  dets = dets[None,:]
  dets[:,0:2] = transform_preds(dets[:,0:2], c[0], s[0], (w, h))
  dets[:,2:4] = transform_preds(dets[:,2:4], c[0], s[0], (w, h))

  center_w = (dets[:,0] + dets[:,2])//2
  center_h = (dets[:,1] + dets[:,2])//2
  if 0<= center_h<= h_feature and 0<= center_w <= w_feature:
    ind = center_w + center_h*w_feature
  elif center_h<0 or center_h>h_feature:
    ind = center_w
  elif center_w<0 or center_w>w_feature:
    ind = center_h
  ind = np.round(ind,0)
  
  return ind


    

