import os
import glob
import numpy as np
import pandas as pd
from skimage import io, transform, exposure, color
from skimage.viewer import ImageViewer
import Classifier

def PreprocessImage(im):
     im = np.resize(im,(128,128,3))
     hsvIm = color.rgb2hsv(im)
     hsvIm[:,:,2] = exposure.equalize_hist(hsvIm[:,:,2])
     im = color.hsv2rgb(hsvIm)
     im = np.rollaxis(im,-1)
     return im

### Read image ###
DATA_PATH = '..//Data//train-jpg//'

train_data = []


print 'Reading image....'
train_filenames = glob.glob(os.path.join(DATA_PATH,"*.jpg"))
np.random.shuffle(train_filenames)
train_filenames = train_filenames[0:15000]
labels = pd.read_csv('..//Data//new_labels.csv')


NoOfDataPoints = len(train_filenames)
Y = np.zeros((NoOfDataPoints,17))

for i,imPath in enumerate(train_filenames):
    try:
        if i % 100 == 0:
            print str(i) + '/' + str(len(train_filenames))
        tempImage = PreprocessImage(io.imread(imPath))
        train_data.append(tempImage)
        image_name = imPath.split('/')[-1][:-4]
        Y[i,:] = labels.loc[labels['image_name'] == image_name].values[0][3:]
    except:
        print 'Error in: ' + imPath
        

print 'Images read.'


X = np.array(train_data,dtype = 'float32')

oClassifier = Classifier.Classfier(128,17,0.01)
model = oClassifier.BuildModel()
trainedModel = oClassifier.Train(model,X,Y,32,1000)         