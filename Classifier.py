from keras.models import Sequential
from keras.layers import Dense,Dropout,Activation,Flatten,Convolution2D, MaxPooling2D, Conv2D
from keras.optimizers import SGD
from keras.callbacks import LearningRateScheduler,ModelCheckpoint
from keras import backend as K
K.set_image_dim_ordering('th')

class Classfier:
	model = Sequential()
	nIMG_SIZE = 0
	nNUM_CLASSES = 0
	dLRate = 0

	def __init__(self,nImageSize,nNumClasses,dLearningRate):
		self.nIMG_SIZE = nImageSize
		self.nNUM_CLASSES = nNumClasses
		self.dLRate = dLearningRate

	def BuildModel(self):
		self.model.add(Conv2D(32,(3,3),border_mode='same',input_shape=(3,self.nIMG_SIZE,self.nIMG_SIZE),activation='relu'))
		self.model.add(Conv2D(32,(3,3),activation='relu'))
		self.model.add(MaxPooling2D(pool_size=(2,2)))#, dim_ordering="th"))
		self.model.add(Dropout(0.2))
		
		self.model.add(Conv2D(64,(3,3),border_mode='same',activation='relu'))
		self.model.add(Conv2D(64,(3,3),activation='relu'))
		self.model.add(MaxPooling2D(pool_size=(2,2)))#,dim_ordering="th"))
		self.model.add(Dropout(0.2))

		self.model.add(Conv2D(128,(3,3),border_mode='same',activation='relu'))
		self.model.add(Conv2D(128,(3,3),activation='relu'))
		self.model.add(MaxPooling2D(pool_size=(2,2)))#,dim_ordering="th"))
		self.model.add(Dropout(0.2))

		self.model.add(Flatten())
		self.model.add(Dense(512,activation='relu'))
		self.model.add(Dropout(0.5))
		self.model.add(Dense(self.nNUM_CLASSES,activation='softmax'))

		return self.model

	def GetModel(self):
		return self.model

	def Train(self,oModel,X,Y,nBatchSize,nEpoch):
		oSGD = SGD(lr=self.dLRate,decay=1e-6,momentum=0.9,nesterov=True)
		oModel.compile(loss='categorical_crossentropy',optimizer=oSGD,metrics=['accuracy'])
		oModel.fit(X,Y,batch_size=nBatchSize,nb_epoch=nEpoch,validation_split=0.2,callbacks=[LearningRateScheduler(self.lr_schedule),ModelCheckpoint('model_run_3.h5',save_best_only=True)])

		return oModel

	def lr_schedule(self,nEpoch):
		return self.dLRate*(0.1**int(nEpoch/10))

