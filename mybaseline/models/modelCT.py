#coding:utf8
import numpy as np
import torch as t
from torch import nn
from torch.autograd import Variable
from torch.nn import functional as F
from .basicmodule import BasicModule

class Model_CT(BasicModule):
	def __init__(self,param):
		super(Model_CT,self).__init__()
		self.param = param
		self.feature_dim = param.feature_dim
		self.object_num = param.object_num
		self.predicate_num = param.predicate_num
		self.batch_size = param.batch_size

		self.fc = nn.Linear(70+8000,self.predicate_num,bias=True)
		t.nn.init.xavier_uniform_(self.fc.weight, gain=1)
		t.nn.init.constant_(self.fc.bias, 0.0)

	def forward(self,f,keys):
		prob_s =Variable(t.from_numpy(f[:,:35])).cuda()
		prob_o = Variable(t.from_numpy(f[:,35:70])).cuda()
		inp_f = Variable(t.from_numpy(f[:,:8070])).cuda()
		p = self.fc(inp_f)
		if self.param.phase=='train':
			# p = F.softmax(p,dim=1)#!!!!
			sel_inds = np.asarray(keys,dtype='int32').T#3xnum of instances
			s = t.gather(prob_s,1,t.from_numpy(np.tile(sel_inds[0],(self.batch_size,1))).long().cuda())#batchx35->batchx2961
			p = t.gather(p,1,t.from_numpy(np.tile(sel_inds[1],(self.batch_size,1))).long().cuda())#batchx132->batchx2961
			o = t.gather(prob_o,1,t.from_numpy(np.tile(sel_inds[2],(self.batch_size,1))).long().cuda())#batchx35->batchx2961
			r = s*p*o
		#prob = F.softmax(r,dim=1)#batchx2961
			return r
		else:
			return p

