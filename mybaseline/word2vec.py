import cPickle
from dataset import Dataset
from config import param
import gensim

dataset= Dataset("/home/szh/AAAI/VidVRD-dataset",param)
objects = dataset.object_categories
model = gensim.models.KeyedVectors.load_word2vec_format('../GoogleNews-vectors-negative300.bin',binary=True)
d = list()
for i,name in enumerate(objects):
	if '_' in name:
		temp = str(name).split('_')
		name_ = temp[-1]
	else:
		name_ = name
	print i,name,name_
	d.append(model[name_])
with open('num2vec.pkl','wb') as f:
	cPickle.dump(d,f)

with open('num2vec.pkl','rb') as f:
	d = cPickle.load(f)

for i,name in enumerate(objects):
	if '_' in name:
		temp = str(name).split('_')
		name_ = temp[-1]
	else:
		name_ = name
	print i,name,name_
	assert (d[i] == model[name_]).all()