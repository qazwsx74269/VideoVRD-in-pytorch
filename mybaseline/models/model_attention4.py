#coding:utf8
import numpy as np
import torch as t
from torch import nn
from torch.autograd import Variable
from torch.nn import functional as F
from .basicmodule import BasicModule

nb_head = 10
size_per_head = 1107

class Model_attention4(BasicModule):
	def __init__(self,param):
		super(Model_attention4,self).__init__()
		self.param = param
		self.feature_dim = param.feature_dim
		self.object_num = param.object_num
		self.predicate_num = param.predicate_num
		self.batch_size = param.batch_size

		self.fc = nn.Linear(nb_head * size_per_head,self.predicate_num,bias=True)
		self.qfc = nn.Linear(self.feature_dim,nb_head * size_per_head,bias=False)
		self.kfc = nn.Linear(self.feature_dim,nb_head * size_per_head,bias=False)
		self.vfc = nn.Linear(self.feature_dim,nb_head * size_per_head,bias=False)
		t.nn.init.xavier_uniform_(self.fc.weight, gain=1)
		t.nn.init.constant_(self.fc.bias, 0.0)

	# def forward(self,inp_f,prob_s,prob_o,keys):
	def forward(self,f,keys):
		if self.param.phase=='test':
			t.set_grad_enabled(False)
		prob_s =Variable(t.from_numpy(f[:,:35])).cuda()
		prob_o = Variable(t.from_numpy(f[:,35:70])).cuda()
		inp_f = Variable(t.from_numpy(f)).cuda()
		Q = self.qfc(inp_f)
		
		Q = Q.view(-1, size_per_head, 1)
		K = self.kfc(inp_f)
		
		K = K.view(-1, 1, size_per_head)
		V = self.vfc(inp_f)
		
		V = V.view(-1, size_per_head, 1)
		A = Q.bmm(K)/t.sqrt(t.FloatTensor([size_per_head]).cuda())
		if self.param.phase=='test':
			del Q
			del K
		A = F.softmax(A,dim=2)
		O = A.bmm(V)
		if self.param.phase=='test':
			del V
			del A
		O = O.view(-1,nb_head*size_per_head)
		p = self.fc(O)
		if self.param.phase=='test':
			del O
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

