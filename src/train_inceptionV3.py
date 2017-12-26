#encoding: utf-8
import myinceptionv3
from keras.models import Model
from keras.layers import Dense, GlobalAveragePooling2D,AveragePooling2D,Flatten
from keras.callbacks import ModelCheckpoint,Callback,LearningRateScheduler
from generator import *
from keras import regularizers
from keras.optimizers import SGD
from SpatialPyramidPooling import SpatialPyramidPooling
from config import *

class LossHistory(Callback):
        def on_train_begin(self, logs={}):
                self.losses = []
        def on_batch_end(self, batch, logs={}):
                self.losses.append(logs.get('loss'))
				
#SGD学习率衰减速率设置
def step_decay(epoch):
    initial_rate=0.001
    drop=0.1
    epochs_drop=2.0
    lrate=initial_rate*math.pow(drop,math.floor((1+epoch)/epochs_drop))
    return lrate
lrate=LearningRateScheduler(step_decay)
sgd=SGD(lr=0.001,momentum=0.9,decay=0.0,nesterov=False)

#构建模型及参数设置
base_model = myinceptionv3.myInceptionV3(include_top=False,weights=None,input_shape=(None,None,3))
x = base_model.output
x=SpatialPyramidPooling([1,2,4])(x)
predictions = Dense(2, activation='softmax')(x)
model = Model(input=base_model.input, output=predictions)
model.compile(optimizer='sgd', loss='binary_crossentropy',metrics=['accuracy', 'binary_crossentropy'])
model.load_weights(train_weight_path,by_name=True)

checkpointer=ModelCheckpoint(filepath='/media/nie/Data/model/chestRay/DR2/local_multiscale/xRayModel-{epoch:02d}-{val_acc:.4f}.hdf5',verbose=2,save_best_only=False)
history=LossHistory()

#构建训练/验证数据集
valAbnormalList=os.listdir(val_abnormal_path)
valNormalList=os.listdir(val_normal_path)
trainAbnormalList=os.listdir(train_abnormal_path)
trainNormalList=os.listdir(train_normal_path)
f = open(txtPath)
content = f.readlines()
f.close()

#第一批数据all resize into 512*512训练
# hist=model.fit_generator(generator=all_resize_generator(trainList),
#                     steps_per_epoch=math.ceil(len(trainList)/4.0),
#                     epochs=10,
#                     validation_data=all_resize_generator(valList),
#                     validation_steps=math.ceil(len(valList)/4.0),
#                     workers=10,
#                     max_q_size=250,
#                     pickle_safe=True,
#                     callbacks=[checkpointer, history])

#第一批数据10*10crop训练
# hist=model.fit_generator(generator=TrainBatchGenerator(trainList,64),
#                     steps_per_epoch=math.ceil(len(trainList)/64.0),
#                     epochs=30,
#                     validation_data=ValBatchGenerator(valList,64),
#                     validation_steps=math.ceil(len(valList)/64.0),
#                     workers=10,
#                     max_q_size=250,
#                     pickle_safe=True,
#                     callbacks=[checkpointer, history])


#第二批数据集10*10crop训练
# hist=model.fit_generator(generator=onlineLocalGenerator(trainAbnormalList,trainNormalList,'./DR2count.txt',5888,26145),
#                     steps_per_epoch=5888,
#                     epochs=9,
#                     validation_data=onlineLocalGenerator(valAbnormalList,valNormalList,'./DR2count.txt',982,4358),
#                     validation_steps=4358,
#                     workers=10,
#                     max_q_size=250,
#                     pickle_safe=True,
#                     callbacks=[checkpointer, history])

#肺部感染多尺度训练
# cropXYlength=(223,446,316,316,387,258,446,223,316,632,447,447,547,365,632,316,447,894,632,632,670,447,591,1182,774,774,865,577)
# numPerPatch=[64,64,64,64,32,32,32,32,16,16,16,8,8,8]
# hist=model.fit_generator(generator=onlineMultiScaleSPPGenerator(trainAbnormalList,trainNormalList,content,259,780,cropXYlength,numPerPatch,mode='train'),
#                     steps_per_epoch=780,
#                     epochs=10,
#                     validation_data=onlineMultiScaleSPPGenerator(valAbnormalList,valNormalList,content,53,180,cropXYlength,numPerPatch,mode='val'),
#                     validation_steps=180,
#                     workers=10,
#                     max_q_size=250,
#                     pickle_safe=True,
#                     callbacks=[checkpointer, history])

#第二批数据集多尺度训练
cropXYlength=(173,577,223,446,316,316,387,258,446,223,500,200,244,816,316,632,447,447,547,365,632,316,705,282,346,1154,447,894,632,632,774,516,894,447,1000,400,
              424,1414,547,1094,774,774,948,632,1094,547,1222,489,490,1633,632,1264,894,894,1095,730,1322,529)
numPerPatch1=[64,64,64,64,64,64,32,32,32,32,32,32,16,16,16,16,16,16,8,8,8,8,8,8,4,4,4,4,4]
numPerPatch=[i-2 for i in numPerPatch1]
hist=model.fit_generator(generator=onlineMultiScaleSPPGenerator(trainAbnormalList,trainNormalList,content,5888,26145,cropXYlength,numPerPatch,mode='train'),
                    steps_per_epoch=5888,
                    epochs=9,
                    validation_data=onlineMultiScaleSPPGenerator(valAbnormalList,valNormalList,content,982,4358,cropXYlength,numPerPatch,mode='val'),
                    validation_steps=4358,
                    workers=10,
                    max_q_size=250,
                    pickle_safe=True,
                    callbacks=[checkpointer, history,lrate])

#保存训练结果
with open('lossAndacc.txt','w') as f:
    f.write(str(hist.history))
