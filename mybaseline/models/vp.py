#coding:utf8
import numpy as np
import torch as t
from torch import nn
from torch.autograd import Variable
from torch.nn import functional as F
from .basicmodule import BasicModule

class VP(BasicModule):
	def __init__(self,param):
		super(VP,self).__init__()
		self.param = param
		self.feature_dim = param.feature_dim
		self.object_num = param.object_num
		self.predicate_num = param.predicate_num
		self.batch_size = param.batch_size

		self.fc = nn.Linear(8070,2961,bias=True)
		t.nn.init.xavier_uniform_(self.fc.weight, gain=1)
		t.nn.init.constant_(self.fc.bias, 0.0)

	# def forward(self,inp_f,prob_s,prob_o,keys):
	def forward(self,f,keys):
		if self.param.phase=='test':
			t.set_grad_enabled(False)
		inp_f = Variable(t.from_numpy(f[:,:8070])).cuda()
		r = self.fc(inp_f)
		if self.param.phase=='train':
			return r
		else:
			return r

