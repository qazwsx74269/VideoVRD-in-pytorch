#coding:utf8
import numpy as np
import torch as t
from torch import nn
from torch.autograd import Variable
from torch.nn import functional as F
from .basicmodule import BasicModule
import cPickle
from dataset import Dataset
from config import param

dataset= Dataset("/home/szh/AAAI/VidVRD-dataset",param)
num2vec = cPickle.load(open('num2vec.pkl','rb'))

class Model_LusV(BasicModule):
	def __init__(self,param):
		super(Model_LusV,self).__init__()
		self.param = param
		self.feature_dim = param.feature_dim
		self.object_num = param.object_num
		self.predicate_num = param.predicate_num
		self.batch_size = param.batch_size

		self.fc = nn.Linear(self.feature_dim,self.predicate_num,bias=True)
		self.language_module = nn.Linear(600,132,bias=True)
		t.nn.init.xavier_uniform_(self.fc.weight, gain=1)
		t.nn.init.constant_(self.fc.bias, 0.0)

	# def forward(self,inp_f,prob_s,prob_o,keys):
	def forward(self,f,keys):
		prob_s =Variable(t.from_numpy(f[:,:35])).cuda()
		prob_o = Variable(t.from_numpy(f[:,35:70])).cuda()
		inp_f = Variable(t.from_numpy(f)).cuda()
		p = self.fc(inp_f)
		prob_p = F.softmax(p,dim=1)
		max_s,index_s = t.topk(prob_s,1,dim=1)
		max_o,index_o = t.topk(prob_o,1,dim=1)
		max_p,index_p = t.topk(prob_p,1,dim=1)
		visualmodule = max_s*max_p*max_o
		Vs = visualmodule.view(-1,1)
		
		if self.param.phase=='train':
			sel_inds = np.asarray(keys,dtype='int32').T#3xnum of instances
			s = t.gather(prob_s,1,t.from_numpy(np.tile(sel_inds[0],(self.batch_size,1))).long().cuda())#batchx35->batchx2961
			p = t.gather(prob_p,1,t.from_numpy(np.tile(sel_inds[1],(self.batch_size,1))).long().cuda())#batchx132->batchx2961
			o = t.gather(prob_o,1,t.from_numpy(np.tile(sel_inds[2],(self.batch_size,1))).long().cuda())#batchx35->batchx2961
			r = s*p*o#visual module
			
		#prob = F.softmax(r,dim=1)#batchx2961
			return r
		else:
			return p

