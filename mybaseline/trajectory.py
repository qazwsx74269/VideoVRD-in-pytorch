#coding:utf8
import numpy as np
from dlib import drectangle
import os
import json
from utils import *
from collections import deque

#bbox sequence in every video segment(30 frames)
class Trajectory(object):
	def __init__(self,pstart,pend,rois,score,category,classeme,vsig=None,gt_trackid=-1):
		assert len(rois)==pend-pstart
		self.pstart = pstart
		self.pend = pend
		self.rois = deque(drectangle(*roi) for roi in rois)#wrap bbox sequence nx4
		self.score = score
		self.category = category
		self.classeme = classeme
		self.vsig = vsig
		self.gt_trackid = gt_trackid

	def __lt__(self,other):
		return self.score<other.score

	def head(self):
		return self.rois[0]

	def tail(self):
		return self.rois[-1]

	def at(self,i):
		'''index via relative location of the video segment'''
		return self.rois[i]

	def roi_at(self,p):
		'''index via location of the whole video'''
		return self.rois[p-self.pstart]

	def bbox_at(self,p):
		"""return bbox's cv2 format"""
		roi = self.rois[p-self.pstart]
		return (roi.left(),roi.top(),roi.width(),roi.height())

	def length(self):
		return self.pend-self.pstart

	def predict(self,roi,reverse=False):
		if reverse:
			self.rois.appendleft(roi)
			self.pstart -= 1
		else:
			self.rois.append(roi)
			self.pend += 1
		return roi#it is detected by Faster-RCNN

	def serialize(self):
		obj = dict()
		obj['pstart'] = int(self.pstart)
		obj['pend'] = int(self.pend)
		obj['rois'] = [(bbox.left(),bbox.top(),bbox.right(),bbox.bottom()) for bbox in self.rois]
		obj['score'] = float(self.score)
		obj['category'] = int(self.category)
		obj['classeme'] = [float(x) for x in self.classeme]
		obj['vsig'] = self.vsig
		obj['gt_trackid'] = self.gt_trackid
		return obj

def _intersect(bboxes1,bboxes2):
	'''compute the intersection of two trajectories'''
	assert bboxes1.shape[0] == bboxes2.shape[0] #judge the amount of frames
	t = bboxes1.shape[0]
	inters = np.zeros((bboxes1.shape[1],bboxes2.shape[1]),dtype=np.float32)
	_min = np.empty((bboxes1.shape[1],bboxes2.shape[1]),dtype=np.float32)
	_max = np.empty((bboxes1.shape[1],bboxes2.shape[1]),dtype=np.float32)
	w = np.empty((bboxes1.shape[1],bboxes2.shape[1]),dtype=np.float32)
	h = np.empty((bboxes1.shape[1],bboxes2.shape[1]),dtype=np.float32)
	for i in range(t):
		np.maximum.outer(bboxes1[i,:,0],bboxes2[i,:,0],out=_min)#xmin
		np.minimum.outer(bboxes1[i,:,2],bboxes2[i,:,2],out=_max)#xmax
		np.subtract(_max+1,_min,out=w)
		w.clip(min=0,out=w)#ignore those which pair has no intersection
		np.maximum.outer(bboxes1[i,:,1],bboxes2[i,:,1],out=_min)#ymin!!!!!!!!!!!!!!
		np.minimum.outer(bboxes1[i,:,3],bboxes2[i,:,3],out=_max)#ymax
		np.subtract(_max+1,_min,out=h)
		h.clip(min=0,out=h)#ignore those which pair has no intersection
		np.multiply(w,h,out=w)
		inters += w
	return inters#n1xn2

def _union(bboxes1,bboxes2):
	'''compute the union of the two trajectories'''
	if id(bboxes1) == id(bboxes2):
		w = bboxes1[:,:,2] - bboxes1[:,:,0] + 1
		h = bboxes1[:,:,3] - bboxes1[:,:,1] + 1
		area = np.sum(w*h,axis=0)#n dimensional
		unions = np.add.outer(area,area)
	else:
		w = bboxes1[:,:,2] - bboxes1[:,:,0] + 1
		h = bboxes1[:,:,3] - bboxes1[:,:,1] + 1
		area1 = np.sum(w*h,axis=0)#n dimensional
		w = bboxes2[:,:,2] - bboxes2[:,:,0] + 1
		h = bboxes2[:,:,3] - bboxes2[:,:,1] + 1
		area2 = np.sum(w*h,axis=0)#n dimensional
		unions = np.add.outer(area1,area2)
	return unions#n1xn2

def cubic_iou(bboxes1,bboxes2):
	#inst["sub_traj"]. nxtx4 format But we need txnx4
	if id(bboxes1) == id(bboxes2):
		bboxes1 = bboxes1.transpose((1,0,2))
		bboxes2 = bboxes1
	else:
		bboxes1 = bboxes1.transpose((1,0,2))
		bboxes2 = bboxes2.transpose((1,0,2))

	iou = _intersect(bboxes1,bboxes2)
	union = _union(bboxes1,bboxes2)
	np.subtract(union,iou,out=union)
	np.divide(iou,union,out=iou)
	return iou

def traj_iou(trajs1,trajs2):
	bboxes1 = np.asarray([[[roi.left(),roi.top(),roi.right(),roi.bottom()] 
		for roi in traj.rois] for traj in trajs1])#boxesxframesx4
	if id(trajs1) == id(trajs2):
		bboxes2 = bboxes1
	else:
		bboxes2 = np.asarray([[[roi.left(),roi.top(),roi.right(),roi.bottom()] 
		for roi in traj.rois] for traj in trajs2])
	iou = cubic_iou(bboxes1,bboxes2)
	return iou

def object_trajectory_proposal(vid,fstart,fend,gt=False,verbose=False):
	'''load generated features'''
	vsig = get_segment_signature(vid,fstart,fend)
	name = 'traj_cls_gt' if gt else 'traj_cls'
	path = get_feature_path(name,vid)
	path = os.path.join(path,'{}-{}.json'.format(vsig,name))
	if os.path.exists(path):
		if verbose:
			print('loading object {} proposal for video segment {}'.format(name,vsig))
		with open(path,'r') as fin:
			trajs = json.load(fin)
		trajs = [Trajectory(**traj) for traj in trajs]
	else:
		if verbose:
			print('no object {} proposal for video segment {}'.format(name,vsig))
		trajs = []
	return trajs#boxesxframesx1

if __name__ == '__main__':
	trajs1 = object_trajectory_proposal('ILSVRC2015_train_00005003',0,30,True,True)
	trajs2 = object_trajectory_proposal('ILSVRC2015_train_00005003',15,45,True,True)
	iou = traj_iou(trajs1,trajs2)
	print(iou)