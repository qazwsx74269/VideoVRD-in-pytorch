#coding:utf8
import numpy as np
import torch as t
from torch import nn
from torch.autograd import Variable
from torch.nn import functional as F
from .basicmodule import BasicModule

class VtransE(BasicModule):
	def __init__(self,param):
		super(VtransE,self).__init__()
		self.param = param
		self.feature_dim = param.feature_dim
		self.object_num = param.object_num
		self.predicate_num = param.predicate_num
		self.batch_size = param.batch_size

		self.fc1 = nn.Linear(7035,500,bias=False)
		self.fc2 = nn.Linear(7035,500,bias=False)
		self.fc3 = nn.Linear(500,self.predicate_num,bias=True)
		t.nn.init.xavier_uniform_(self.fc1.weight, gain=1)
		t.nn.init.xavier_uniform_(self.fc2.weight, gain=1)
		t.nn.init.xavier_uniform_(self.fc3.weight, gain=1)
		t.nn.init.constant_(self.fc3.bias, 0.0)

	# def forward(self,inp_f,prob_s,prob_o,keys):
	def forward(self,f,keys):
		if self.param.phase=='test':
			t.set_grad_enabled(False)
		prob_s =Variable(t.from_numpy(f[:,:35])).cuda()
		prob_o = Variable(t.from_numpy(f[:,35:70])).cuda()
		# inp_f = Variable(t.from_numpy(f)).cuda()
		sf = Variable(t.from_numpy(np.concatenate((f[:,:35],f[:,70:4070],f[:,8070:11070]),axis=1))).cuda()
		of = Variable(t.from_numpy(np.concatenate((f[:,35:70],f[:,4070:8070],f[:,8070:11070]),axis=1))).cuda()
		xs = F.relu(self.fc1(sf))
		xo = F.relu(self.fc2(of))
		xp = xo - xs
		p = self.fc3(xp)
		if self.param.phase=='train':
			return p
		else:
			return p

