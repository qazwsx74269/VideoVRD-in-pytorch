#coding:utf8
import models
from config import param
import torch as t
from torch import nn
from torch.autograd import Variable
from utils import *
from dataset import Dataset
import numpy as np
from time import strftime
import json
from tqdm import tqdm
from collections import defaultdict,deque
import association
from metric import *
import time
import sys
from visualizer import Visualizer
from torchnet.meter import AverageValueMeter
# from gpu_profile import gpu_profile

def one_hot(ids,out_tensor):
		"""
		ids: (list, ndarray) shape:[batch_size]
		out_tensor:FloatTensor shape:[batch_size, depth]
		"""
		if not isinstance(ids, (list, np.ndarray)):
				raise ValueError("ids must be 1-D list or array")
		ids = t.LongTensor(ids).view(-1,1)
		out_tensor.zero_()
		out_tensor.scatter_(dim=1, index=ids, value=1)
		# out_tensor.scatter_(1, ids, 1.0)
 
def train(**kwargs):
	param.parse(kwargs)
	vis = Visualizer(param.env)
	dataset= Dataset("/home/szh/AAAI/VidVRD-dataset",param)
	keys = dataset._train_triplet_id.keys()
	# model = Model(param).cuda()
	model = getattr(models,param.model_name)(param).cuda()
	model.train()
	optimizer = t.optim.Adam(model.parameters(),lr=param.learning_rate)
	criterion = nn.CrossEntropyLoss().cuda()
	loss_meter = AverageValueMeter()
	for i in range(param.max_iter):
		try:
			f,r = dataset.get_prefetch_data()
			
			# prob_s =Variable(t.from_numpy(f[:,:35])).cuda()
			# prob_o = Variable(t.from_numpy(f[:,35:70])).cuda()
			# inp_f = Variable(t.from_numpy(f)).cuda()
			#y = t.FloatTensor(param.batch_size,param.triplet_num)
			#one_hot(r,y)
			y = Variable(t.from_numpy(r).long()).cuda()
			optimizer.zero_grad()
			p_ids = list()
			s_ids = list()
			o_ids = list()
			for j in range(len(r)):
				(s_id,p_id,o_id) = dataset._train_triplet_id_r[r[j]]
				p_ids.append(p_id)
				s_ids.append(s_id)
				o_ids.append(o_id)
			y1 = Variable(t.from_numpy(np.asarray(s_ids)).long()).cuda()
			y2 = Variable(t.from_numpy(np.asarray(p_ids)).long()).cuda()
			y3 = Variable(t.from_numpy(np.asarray(o_ids)).long()).cuda()
			# output  = model(inp_f,prob_s,prob_o,keys)
			output = model(f,keys)
			if param.use_m or param.vtranse:
				loss = criterion(output,y2)
				# loss1.backward()
				# loss3.backward()

				# loss = criterion(output2,y2)
			else:
				loss = criterion(output,y)
				
			loss_meter.add(loss.data.item())
			loss.backward()
			optimizer.step()
			if i % param.display_freq == 0:
					print('{} -{}- iter: {}/{} - loss: {:.4f}'.format(
							strftime('%Y-%m-%d %H:%M:%S'), param.model_name,i, param.max_iter, float(loss.data)))
			# if i % param.save_freq == 0 and i > 0:
			# 		param.model_dump_file = '{}_weights_iter_{}'.format(param.model_name, i)
			# 		model.save(name = param.model_dump_file)
			if i % 20 == 19:
					vis.plot('loss',loss_meter.value()[0])
		except KeyboardInterrupt:
			print('Early Stop.')
			break
	else:
		# save model
		param.model_dump_file = '{}_weights_iter_{}'.format(param.model_name, param.max_iter)
		model.save(name=param.model_dump_file)
		# save settings
		# with open(os.path.join(get_model_path(), '{}_setting.json'.format(param.model_name)), 'w') as fout:
		# 	json.dump(param.state_dict(), fout, indent=4)
	param.phase = 'test'
	dataset= Dataset("/home/szh/AAAI/VidVRD-dataset",param)
	eval(dataset,model)


def eval_file(**kwargs):
	param.phase = 'test'
	param.parse(kwargs)
	dataset= Dataset("/home/szh/AAAI/VidVRD-dataset",param)
	# model = Model(param).cuda()
	model = getattr(models,param.model_name)(param).cuda()
	model.load(os.path.join(get_model_path(), param.model_dump_file))
	eval(dataset,model)

def eval(dataset,model):
	model.eval()		
	keys = dataset._train_triplet_id.keys()
	print('predicting short-term visual relation...')
	pbar = tqdm(total=len(dataset.index))
	short_term_relations = dict()
	data = dataset.get_data()

	# ss = sys.stdout
	while data:
		index,pairs,feats,iou,trackid = data
		# prob_s =Variable(t.from_numpy(feats[:,:35])).cuda()
		# prob_o = Variable(t.from_numpy(feats[:,35:70])).cuda()
		# inp_f = Variable(t.from_numpy(feats)).cuda()
		# p = model(inp_f,prob_s,prob_o,keys)
		n = feats.shape[0]
		r = t.empty(0,2961)
		s = 0
		while n>0:
			batch = min(n,dataset.param.batch_size)
			n -= batch
			temp = model(feats[s:s+batch,:],keys)
			r = t.cat((r,temp.cpu()),0)
			s += batch
			t.cuda.empty_cache()
		# p = model(feats,keys)
		r = r.detach().numpy()
		predictions = []
		for i in range(len(pairs)):
			top_r_ind = np.argsort(r[i])[-param.pair_topk:]
			predictions.extend((
				r[i][top_r_ind[j]],
				(dataset._train_triplet_id_r[top_r_ind[j]]),
				tuple(pairs[i])) for j in range(param.pair_topk))
		predictions = sorted(predictions,key=lambda x: x[0],reverse=True)[:param.seg_topk]
		short_term_relations[index] = (predictions,iou,trackid)
		# with open('/home/szh/AAAI/mybaseline/debug.txt','a+') as file:
			# sys.stdout = file
			# print(index)
		data = dataset.get_data()
		pbar.update(1)
	pbar.close()
	# stdout = ss
	#group short term relations by video
	video_vid_short_relations = defaultdict(list)
	for index,relation in short_term_relations.items():
		vid = index[0]#vid
		video_vid_short_relations[vid].append((index,relation))
	# video-level visual relation detection by relational association
	
	print('greedy relational association ...')
	video_relations = dict()
	for vid in tqdm(video_vid_short_relations.keys()):#predict video-level relations
		video_relations[vid] = association.greedy_relational_association(
			dataset,video_vid_short_relations[vid],max_traj_num_in_clip=200)
	# save detection result
	prefix_ = 'output/{}_baseline_video_relations_'.format(model.param.model_name)
	fullname_ = time.strftime(prefix_+'%m%d_%H%M%S.json')
	with open(fullname_, 'w') as fout:
		json.dump(video_relations, fout)
	assess(dataset,video_relations)
	

def assess_file(**kwargs):
	param.parse(kwargs)
	dataset= Dataset("/home/szh/AAAI/VidVRD-dataset",param)
	with open(param.prediction_file, 'r') as fin:
		prediction_json = json.load(fin)
	# savedStdout = sys.stdout
	# with open('debug.txt','a+') as file:
	# 	sys.stdout = file
	# 	print(prediction_json.keys())
	# 	sys.stdout = savedStdout

	assess(dataset,prediction_json)

def assess(dataset,prediction_json):
	groundtruth = dict()
	for vid in dataset.get_index('test'):
		groundtruth[vid] = dataset.video_relations[vid]
	mAP, rec_at_n, mprec_at_n = eval_visual_relation(groundtruth, prediction_json)
	file_name = "experiment_results.xls"
	Method = dataset.param.model_name+'_'+str(dataset.param.max_iter)+time.strftime('_%m%d_%H%M%S')
	print(Method)
	print('detection mAP: {}'.format(mAP))
	print('detection recall@50: {}'.format(rec_at_n[50]))
	print('detection recall@100: {}'.format(rec_at_n[100]))
	print('tagging precision@1: {}'.format(mprec_at_n[1]))
	print('tagging precision@5: {}'.format(mprec_at_n[5]))
	print('tagging precision@10: {}'.format(mprec_at_n[10]))
	values1 = list()
	values1.extend([Method,rec_at_n[50].item(),rec_at_n[100].item(),mAP.item(),mprec_at_n[1].item(),mprec_at_n[5].item(),mprec_at_n[10].item()])
	# evaluate in zero-shot setting
	print('-----zero-shot------')
	zeroshot_triplets = dataset.get_triplets('test').difference(
		  dataset.get_triplets('train'))
	zeroshot_groundtruth = dict()
	for vid in dataset.get_index('test'):
		gt_relations = dataset.video_relations[vid]
		zs_gt_relations = []
		for r in gt_relations:
			if tuple(r['triplet']) in zeroshot_triplets:
				zs_gt_relations.append(r)
		if len(zs_gt_relations) > 0:
			zeroshot_groundtruth[vid] = zs_gt_relations
	mAP, rec_at_n, mprec_at_n = eval_visual_relation(
		  zeroshot_groundtruth, prediction_json)
	print('detection mAP: {}'.format(mAP))
	print('detection recall@50: {}'.format(rec_at_n[50]))
	print('detection recall@100: {}'.format(rec_at_n[100]))
	print('tagging precision@1: {}'.format(mprec_at_n[1]))
	print('tagging precision@5: {}'.format(mprec_at_n[5]))
	print('tagging precision@10: {}'.format(mprec_at_n[10]))
	values2 = list()
	Method = Method + '_z'
	values2.extend([Method,rec_at_n[50].item(),rec_at_n[100].item(),mAP.item(),mprec_at_n[1].item(),mprec_at_n[5].item(),mprec_at_n[10].item()])
	write_to_excel(file_name,values1,values2)

if __name__=="__main__":
	# sys.settrace(gpu_profile)
	import fire
	fire.Fire()
	