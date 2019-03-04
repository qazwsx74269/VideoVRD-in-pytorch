#coding:utf8
import os
import h5py
from sklearn.preprocessing import normalize

def get_segment_signature(vid,fstart,fend):
	return "{}-{:04d}-{:04d}".format(vid,fstart,fend)

def get_feature_path(name,vid):
	path = os.path.join("/home/szh/AAAI/VidVRD-dataset","features",name)
	if not os.path.exists(path):
		os.makedirs(path)
	path = os.path.join(path,vid)
	if not os.path.exists(path):
		os.makedirs(path)
	return path

def get_model_path():
	path = os.path.join("/home/szh/AAAI/mybaseline","checkpoints")
	if not os.path.exists(path):
		os.makedirs(path)
	return path

def segment_video(fstart,fend):
	segs = [(i,i+30) for i in range(fstart,fend-30+1,15)]
	return segs

def extract_feature(vid,fstart,fend,dry_run=False,verbose=False):#relation feature
	vsig = get_segment_signature(vid,fstart,fend)
	path = get_feature_path('relation',vid)
	path = os.path.join(path,'{}-{}.h5'.format(vsig,'relation'))
	if os.path.exists(path):
		if dry_run:
			return None,None,None,None
		else:
			if verbose:
				print('loading relation feature for video segment {}...'.format(vsig))
			with h5py.File(path,'r') as fin:#format example 
				trackid = fin['trackid'][:]#(5)
				pairs = fin['pairs'][:]#(20,2) 20 pair
				feats = fin['feats'][:]#(20,11070)
				iou = fin['iou'][:]#(5,5)
			return pairs,feats,iou,trackid
	else:
		if verbose:
			print('no relation feature for video segment {}'.format(vsig))
		return None

def feature_preprocess(feat):


 
  	# (since this feature is Bag-of-Word type, we l1-normalize it so that
  	# each element represents the fraction instead of count)
 	# subject classeme + object classeme
 	# feat[:, 0: 70]
 	# subject TrajectoryShape + HoG + HoF + MBH motion feature
	feat[:,70:1070] = normalize(feat[:,70:1070],'l1',1)
	feat[:,1070:2070] = normalize(feat[:,1070:2070],'l1',1)
	feat[:,2070:3070] = normalize(feat[:,2070:3070],'l1',1)
	feat[:,3070:4070] = normalize(feat[:,3070:4070],'l1',1)
	# object TrajectoryShape + HoG + HoF + MBH motion feature
	feat[:,4070:5070] = normalize(feat[:,4070:5070],'l1',1)
	feat[:,5070:6070] = normalize(feat[:,5070:6070],'l1',1)
	feat[:,6070:7070] = normalize(feat[:,6070:7070],'l1',1)
	feat[:,7070:8070] = normalize(feat[:,7070:8070],'l1',1)
	# relative posititon + size + motion feature
  	# feat[:, 8070: 9070]
  	# feat[:, 9070: 10070]
  	# feat[:, 10070: 11070]
	return feat

def write_to_excel(file,values1,values2):
	from xlrd import open_workbook
	from xlutils.copy import copy
	# rexcel = open_workbook("experiment_results.xlsx") # 用wlrd提供的方法读取一个excel文件
	rexcel = open_workbook(file) # 用wlrd提供的方法读取一个excel文件
	rows = rexcel.sheets()[0].nrows # 用wlrd提供的方法获得现在已有的行数
	excel = copy(rexcel) # 用xlutils提供的copy方法将xlrd的对象转化为xlwt的对象
	table = excel.get_sheet(0) # 用xlwt对象的方法获得要操作的sheet
	# values = ["1", "2", "3"]
	row = rows
	for i,value in enumerate(values1):
	    table.write(row-100, i, value) # xlwt对象的写方法，参数分别是行、列、值
	for i,value in enumerate(values2):
	    table.write(row, i, value) # xlwt对象的写方法，参数分别是行、列、值
	    # table.write(row, 1, "haha")
	    # table.write(row, 2, "lala")
	    # row += 1
	# excel.save("experiment_results.xlsx") # xlwt对象的保存方法，这时便覆盖掉了原来的excel
	excel.save(file) # xlwt对象的保存方法，这时便覆盖掉了原来的excel


if __name__ == '__main__':
	feat = extract_feature('ILSVRC2015_train_00005003',15,45,verbose=True)
	print(feat[0].shape)
	print(feat[1].shape)
	print(feat[2].shape)
	print(feat[3].shape)