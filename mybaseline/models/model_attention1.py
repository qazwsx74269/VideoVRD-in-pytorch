#coding:utf8
import numpy as np
import torch as t
from torch import nn
from torch.autograd import Variable
from torch.nn import functional as F
from .basicmodule import BasicModule

# def Attention(Q, K, V, nb_head, size_per_head, Q_len=None, V_len=None):
#     #对Q、K、V分别作线性映射
#     Q = Dense(Q, nb_head * size_per_head, False)
#     Q = tf.reshape(Q, (-1, tf.shape(Q)[1], nb_head, size_per_head))
#     Q = tf.transpose(Q, [0, 2, 1, 3])
#     K = Dense(K, nb_head * size_per_head, False)
#     K = tf.reshape(K, (-1, tf.shape(K)[1], nb_head, size_per_head))
#     K = tf.transpose(K, [0, 2, 1, 3])
#     V = Dense(V, nb_head * size_per_head, False)
#     V = tf.reshape(V, (-1, tf.shape(V)[1], nb_head, size_per_head))
#     V = tf.transpose(V, [0, 2, 1, 3])
#     #计算内积，然后mask，然后softmax
#     A = tf.matmul(Q, K, transpose_b=True) / tf.sqrt(float(size_per_head))
#     A = tf.transpose(A, [0, 3, 2, 1])
#     A = Mask(A, V_len, mode='add')
#     A = tf.transpose(A, [0, 3, 2, 1])
#     A = tf.nn.softmax(A)
#     #输出并mask
#     O = tf.matmul(A, V)
#     O = tf.transpose(O, [0, 2, 1, 3])
#     O = tf.reshape(O, (-1, tf.shape(O)[1], nb_head * size_per_head))
#     O = Mask(O, Q_len, 'mul')
#     return O

nb_head = 10
size_per_head = 1107

class Model_attention1(BasicModule):
	def __init__(self,param):
		super(Model_attention1,self).__init__()
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
		prob_s =Variable(t.from_numpy(f[:,:35])).cuda()
		prob_o = Variable(t.from_numpy(f[:,35:70])).cuda()
		inp_f = Variable(t.from_numpy(f)).cuda()
		Q = self.qfc(inp_f)
		Q = Q.view(-1, nb_head, size_per_head)
		K = self.kfc(inp_f)
		K = K.view(-1, nb_head, size_per_head)
		V = self.vfc(inp_f)
		V = V.view(-1, nb_head, size_per_head)
		A = Q*K/t.sqrt(t.FloatTensor([size_per_head]).cuda())
		A = F.softmax(A,dim=2)
		O = A*V
		O = O.view(-1,nb_head*size_per_head)
		p = self.fc(O)
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

