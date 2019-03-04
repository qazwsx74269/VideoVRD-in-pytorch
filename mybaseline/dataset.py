#coding:utf8
import glob
import json
import os
from utils import *
import numpy as np
from collections import defaultdict,OrderedDict
from config import param
from itertools import cycle,product
from multiprocessing import Process,Queue,sharedctypes
import atexit,signal

class SharedArray(object):#share the memory
	def __init__(self,shape,dtype=np.float32):
		size = np.prod(shape)
		if dtype == np.float32:
			typecode = 'f'
		elif dtype == np.float64:
			typecode = 'd'
		else:
			assert False,'Unknown dtype.'
		self.data = sharedctypes.RawArray(typecode,size)
		self.shape = shape
		self.dtype = dtype

	def set_value(self,value):
		nparr = np.ctypeslib.as_array(self.data)
		'''
		numpy.ctypeslib.as_array(obj, shape=None)
Create a numpy array from a ctypes array or POINTER.
The numpy array shares the memory with the ctypes object.
The shape parameter must be given if converting from a ctypes POINTER. The shape parameter is ignored if converting from a ctypes array
		'''
		nparr.shape = self.shape
		nparr[...] = value.astype(self.dtype,copy=False)

	def get_value(self,copy=True):
		nparr = np.ctypeslib.as_array(self.data)
		nparr.shape = self.shape
		if copy:
			return np.array(nparr)
		else:
			return nparr

class Dataset(Process):#add multiprocessing
	'''Process and index the annotation'''
	def __init__(self,data_path,param,prefetch_count=2):
	#***********************************************************************************************first part
		super(Dataset,self).__init__()
		self.prefetch_count = prefetch_count
		self.param = param
		self.data_path = data_path
		self.video_path = os.path.join(self.data_path,'videos')
		self.train_json_path = os.path.join(self.data_path,'train')
		self.test_json_path = os.path.join(self.data_path,'test')

		self.annotations = dict()
		self.video_relations = dict()#map video id to all relationship instances belonging to it
		self.train_vids = []
		self.test_vids = []

		self.object_categories = set() #text
		self.predicate_categories = set() #text

		self.object_categories2number = dict()
		self.predicate_categories2number = dict()
		self.object_categories2name = dict()
		self.predicate_categories2name = dict()

		#read the information
		json_paths = [self.train_json_path,self.test_json_path]
		for i,json_path in enumerate(json_paths):
			for json_file in glob.glob(os.path.join(json_path,'*.json')):
				with open(json_file,'r') as f:
					annotation = json.load(f)   
					#get related information via key
					video_id = annotation["video_id"]
					frame_count = annotation["frame_count"]
					fps = annotation["fps"]
					width = annotation["width"]
					height = annotation["height"]
					subject_objects = annotation["subject/objects"]
					trajectories = annotation["trajectories"]
					relation_instances = annotation["relation_instances"]
					#get what you want
					self.annotations[video_id] = annotation

					tid2category = dict()
					for o in subject_objects:
						tid2category[o["tid"]] = o["category"]
						self.object_categories.add(o["category"])
					for r in relation_instances:
						self.predicate_categories.add(r["predicate"])
					if i==0:#train set
						self.train_vids.append(video_id)
					else:
						self.test_vids.append(video_id)

					frames_boxes = []
					for frame in trajectories:
						tid2box = dict()
						for box in frame:
							tid2box[box["tid"]] = (
								box["bbox"]["xmin"],
								box["bbox"]["ymin"],
								box["bbox"]["xmax"],
								box["bbox"]["ymax"])
						frames_boxes.append(tid2box)#store mapping in every frame
					
					relationship_instances  = []#relationship in every video
					for relation_instance in relation_instances:
						instance = dict()
						stid = relation_instance["subject_tid"]
						otid = relation_instance["object_tid"]
						predicate = relation_instance["predicate"]
						begin_fid = relation_instance["begin_fid"]
						end_fid = relation_instance["end_fid"]

						instance["subject_tid"] = stid
						instance["object_tid"] = otid
						instance["triplet"] = (tid2category[stid],predicate,tid2category[otid])
						instance["duration"] = (begin_fid,end_fid)
						instance["sub_traj"] = [tid2box[stid] for tid2box in frames_boxes[begin_fid:end_fid]]
						instance["obj_traj"] = [tid2box[otid] for tid2box in frames_boxes[begin_fid:end_fid]]

						relationship_instances.append(instance)
					self.video_relations[video_id] = relationship_instances

		#index for objects and predicates
		self.object_categories = sorted(self.object_categories)
		self.predicate_categories = sorted(self.predicate_categories)
		for i,name in enumerate(self.object_categories):
			self.object_categories2name[i] = name
			self.object_categories2number[name] = i
		for i,name in enumerate(self.predicate_categories):
			self.predicate_categories2name[i] = name
			self.predicate_categories2number[name] = i

	#***********************************************************************************************second part
		self._train_triplet_id = OrderedDict()
		self._train_triplet_id_r = OrderedDict()
		self.phase = param.phase
		self.rng = np.random.RandomState(param.rng_seed)
		self.batch_size = param.batch_size
		self.max_sampling_in_batch = param.max_sampling_in_batch
		assert self.max_sampling_in_batch <= self.batch_size
		print('preparing video segments for {}...'.format(self.phase))
		self._train_triplet_id.clear()
		self._train_triplet_id_r.clear()
		triplets = self.get_triplets(split='train')
		for i,triplet in enumerate(triplets):
			#print(triplet)
			s_name,p_name,o_name = triplet
			s_id = self.get_object_number(s_name)
			o_id = self.get_object_number(o_name)
			p_id = self.get_predicate_number(p_name)
			self._train_triplet_id[(s_id,p_id,o_id)] = i
			self._train_triplet_id_r[i] = (s_id,p_id,o_id)
		if self.phase == 'train':
			# self._train_triplet_id.clear()
			# self._train_triplet_id_r.clear()
			# triplets = self.get_triplets(split='train')
			# for i,triplet in enumerate(triplets):
			# 	#print(triplet)
			# 	s_name,p_name,o_name = triplet
			# 	s_id = self.get_object_number(s_name)
			# 	o_id = self.get_object_number(o_name)
			# 	p_id = self.get_predicate_number(p_name)
			# 	self._train_triplet_id[(s_id,p_id,o_id)] = i
			# 	self._train_triplet_id_r[i] = (s_id,p_id,o_id)

			self.short_relation_instances = defaultdict(list)
			vids = self.get_index(split='train')
			for vid in vids:
				for short_relation_instance in self.video_relations[vid]:
					segments = segment_video(*short_relation_instance['duration'])
					for fstart,fend in segments:
						if extract_feature(vid,fstart,fend,dry_run=True):
							s_name,p_name,o_name = short_relation_instance['triplet']
							self.short_relation_instances[(vid,fstart,fend)].append((
								short_relation_instance['subject_tid'],
								short_relation_instance['object_tid'],
								self.get_object_number(s_name),
								self.get_predicate_number(p_name),
								self.get_object_number(o_name)
								))
			self.index = list(self.short_relation_instances.keys())#video segment instances
			self.index_iter = cycle(range(len(self.index)))
		elif self.phase=='test':
			self.index = []
			vids = self.get_index(split='test')
			for vid in vids:
				annotation =self.get_annotation(vid)
				segments = segment_video(0,annotation['frame_count'])
				for fstart,fend in segments:
					if extract_feature(vid,fstart,fend,dry_run=True):
						self.index.append((vid,fstart,fend))
			self.index_iter = iter(range(len(self.index)))
		else:
			raise ValueError('Unknown phase: {}'.format(self.phase))
	
	def _init_pool(self):
		prefetch_count = self.prefetch_count
		if prefetch_count>0:
			self._blob_pool = [list() for i in range(prefetch_count)]
			self._free_queue = Queue(prefetch_count)
			self._full_queue = Queue(prefetch_count)

			shapes = self.get_data_shapes()
			for i,shape in enumerate(shapes):
				for j in range(prefetch_count):
					self._blob_pool[j].append(SharedArray(shape,np.float32))

			for i in range(prefetch_count):
				self._free_queue.put(i)
			atexit.register(self._cleanup)
			self.start()
		else:
			print('prefetching disabled.')

	def _cleanup(self):
		if self.prefetch_count>0:
			print('Terminating DataFetcher')
			self.terminate()
			self.join()

	def get_prefetch_data(self):
		if not hasattr(self,'_full_queue') and self.prefetch_count>0:
			self._init_pool()
		blobs = []
		pool_ind = self._full_queue.get()
		for blob in self._blob_pool[pool_ind]:
			blobs.append(blob.get_value())
		self._free_queue.put(pool_ind)
		return blobs

	def run(self):
		signal.signal(signal.SIGINT,signal.SIG_DFL)
		print('DataFetcher started')
		while True:
			blobs = self.get_data()
			pool_ind = self._free_queue.get()
			for i,s_blob in enumerate(self._blob_pool[pool_ind]):
				s_blob.set_value(blobs[i])
			self._full_queue.put(pool_ind)

	def get_data_shapes(self):
		if not hasattr(self,'shapes'):
			print('Getting data to measure this shapes...')
			data = self.get_data()
			self.shapes = tuple(d.shape for d in data)
		return self.shapes

	def get_data(self):
		if self.phase == 'train':#sample instances from every video segment
			f = []
			r = []
			remaining_size = self.batch_size
			while remaining_size>0:
				i = self.index_iter.next()
				vid,fstart,fend = self.index[i]
				sample_num = np.minimum(remaining_size,self.max_sampling_in_batch)
				_f,_r = self._data_sampling(vid,fstart,fend,sample_num)#sample_num is the upper bound maybe the amount of instances belong to a video segment is not enough
				remaining_size -= _f.shape[0]
				if _f.shape[0]!=0:
					_f = feature_preprocess(_f)
				f.append(_f.astype(np.float32))
				r.append(_r.astype(np.float32))
			f = np.concatenate(f)
			r = np.concatenate(r)
			return f,r
		else:
			try:
				i = self.index_iter.next()
			except StopIteration:
				return None
			index = self.index[i]
			pairs,feats,iou,trackid = extract_feature(*index)
			test_inds = [ind for ind,(traj1,traj2) in enumerate(pairs) 
				if trackid[traj1]<0 and trackid[traj2]<0]#proposals
			pairs = pairs[test_inds]
			if feats[test_inds].shape[0] != 0:
				feats = feature_preprocess(feats[test_inds])
			feats = feats.astype(np.float32)
			return index,pairs,feats,iou,trackid

	def _data_sampling(self,vid,fstart,fend,sample_num,iou_thres=0.5):
		
		pairs,feats,iou,trackid = extract_feature(vid,fstart,fend)
		feats = feats.astype(np.float32)
		pair_to_ind = dict([((traj1,traj2),ind)
			for ind,(traj1,traj2) in enumerate(pairs)])
		tid_to_ind = dict([(tid,ind) 
			for ind,tid in enumerate(trackid) if tid >= 0])#groundtruth
		pos = np.empty((0,2),dtype=np.int32)#prepare for np.concatenate
		for tid1,tid2,s,p,o in self.short_relation_instances[(vid,fstart,fend)]:
			if tid1 in tid_to_ind and tid2 in tid_to_ind:#ensure it is groundtruth
				iou1 = iou[:,tid_to_ind[tid1]]
				iou2 = iou[:,tid_to_ind[tid2]]
				pos_inds1 = np.where(iou1>=iou_thres)[0]#choose subject proposals whose viou>=0.5 with respect to subject groundtruth
				pos_inds2 = np.where(iou2>=iou_thres)[0]#choose object proposals whose viou>=0.5 with respect to object groundtruth
				tmp = [(pair_to_ind[(traj1,traj2)],self._train_triplet_id[(s,p,o)])
					for traj1,traj2 in product(pos_inds1,pos_inds2) if traj1!=traj2]
				if len(tmp)>0:
					pos = np.concatenate((pos,tmp))
		num_pos_in_this = np.minimum(pos.shape[0],sample_num)
		if pos.shape[0]>0:
			pos = pos[np.random.choice(pos.shape[0],num_pos_in_this,replace=False)]
		return feats[pos[:,0]],pos[:,1]#Classeme,iDT,relativity feature with corresponding segment-level triplet

	def get_object_categories_types(self):
		return len(self.object_categories)

	def get_predicate_categories_types(self):
		return len(self.predicate_categories)

	def get_object_number(self,name):
		return self.object_categories2number[name]

	def get_object_name(self,number):
		return self.object_categories2name[number]

	def get_predicate_number(self,name):
		return self.predicate_categories2number[name]

	def get_predicate_name(self,number):
		return self.predicate_categories2name[number]

	def get_annotation(self,video_id):
		return self.annotations[video_id]

	def get_index(self,split="train"):
		if split=="train":
			return self.train_vids
		else:
			return self.test_vids

	def get_triplets(self,split="train"):
		triplets = set()
		for vid in self.get_index(split):
			insts = self.video_relations[vid]
			triplets.update(inst["triplet"] for inst in insts)
		return triplets

if __name__ == '__main__':
	dataset= Dataset("/home/szh/AAAI/VidVRD-dataset",param)
	# print(dataset.get_data()[0].shape)
	# print(dataset.get_predicate_categories_types())
	# print(dataset.get_object_categories_types())
	# vids = dataset.get_index("train")
	# print(len(vids))
	# print(len(dataset._train_triplet_id))
	# test_inds = dataset.get_index('test')
	# insts = dataset.video_relations[test_inds[111]]
	# print(insts)
	print(dataset.predicate_categories)
	print(dataset.object_categories)