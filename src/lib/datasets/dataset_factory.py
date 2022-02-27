from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

from .dataset.jde import JointDataset
from .dataset.multiframe_jde import MultiFrameJointDataset


def get_dataset(dataset, task):
  if task == 'mot':
    return JointDataset
  if task == 'MultiFrameMot':
    return MultiFrameJointDataset
  else:
    return None
  
