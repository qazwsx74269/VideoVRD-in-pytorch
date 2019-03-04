#coding:utf8
import torch as t
import time

class BasicModule(t.nn.Module):
	def __init__(self):
		super(BasicModule,self).__init__()

	def load(self,path):
		self.load_state_dict(t.load(path))

	def save(self,name=''):
		#if name is None:
		prefix = 'checkpoints/'+name+'_'
		fullname = time.strftime(prefix+'%m%d_%H%M%S.pth')
		t.save(self.state_dict(),fullname)