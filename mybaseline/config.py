#coding:utf8
import warnings
import os
#os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'
#os.environ['CUDA_VISIBLE_DEVICES'] = '1'
class Param(object):
  model_name = 'Model'
  rng_seed = 1701
  max_sampling_in_batch = 32
  batch_size = 64
  learning_rate = 0.001
  weight_decay = 0.0
  max_iter = 5000
  display_freq = 1
  save_freq = 5000
  epsilon = 1e-8
  pair_topk = 20
  seg_topk = 200
  phase = 'train'
  object_num = 35
  predicate_num = 132
  feature_dim = 11070
  triplet_num = 2961
  model_dump_file = ''
  prediction_file = ""
  use_gt = False
  use_m = False
  vtranse = False
  env = 'szh'

def parse(self,kwargs):
	for k,v in kwargs.items():
		if not hasattr(self,k):
			warnings.warn("your config doesn't have that key(%s)"%(k))
		else:
			setattr(self,k,v)
	print("user config:")
	print("**************************************")
	for k in dir(self):
		if not k.startswith("_") and k!='parse' and k!="state_dict":
			print k,getattr(self,k)
	print("**************************************")
	return self

def state_dict(self):
	return {k:getattr(self,k) for k in dir(self) if not k.startswith('_') and k!="parse" and k!="state_dict"}


Param.parse = parse
Param.state_dict = state_dict
param = Param()
