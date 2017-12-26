import myinceptionv3
from keras.models import Model
from keras.layers import Dense, GlobalAveragePooling2D
from generator import *
from SpatialPyramidPooling import SpatialPyramidPooling
from keras.models import load_model
from config import *

base_model = myinceptionv3.myInceptionV3(include_top=False,weights=None,input_shape=(None,None,3))
x = base_model.output
x=SpatialPyramidPooling([1,2,4])(x)
predictions = Dense(2, activation='softmax')(x)
model = Model(input=base_model.input, output=predictions)
model.compile(optimizer='rmsprop', loss='binary_crossentropy',metrics=['accuracy','binary_crossentropy'])
model.load_weights(test_weight_path,by_name=True)

#第一批数据集all resize into 512*512测试
# result=model.predict_generator(generator=test_global_generator(test_img_path,2990,299,19),
#                     steps=173,
#                     workers=1,
#                     max_q_size=250,
#                     pickle_safe=True)
# np.save('./local_abnormal_test',result)

#第一批数据集/第二批数据集10*10crop测试
# result=model.predict_generator(generator=test_local_generator(test_img_path,2990,299,19),
#                     steps=173,
#                     workers=1,
#                     max_q_size=250,
#                     pickle_safe=True)
# np.save('./local_abnormal_test',result)

#肺部感染多尺度测试
# cropXYlength=(223,446,316,316,387,258,446,223,316,632,447,447,547,365,632,316,447,894,632,632,670,447,591,1182,774,774,865,577)
# numPerPatch=[64,64,64,64,32,32,32,32,16,16,16,8,8,8]
# abnormalSteps=[212,212,212,212,424,424,424,424,848,848,848,1696,1696,1696]
# normalSteps=[720,720,720,720,1440,1440,1440,1440,2880,2880,2880,5760,5760,5760]
#for i in xrange(14):
#    result=model.predict_generator(generator=test_multiScale_SPP_generator(test_img_path,cropXYlength[2*i],cropXYlength[2*i+1],16,numPerPatch[i]),
#                        steps=normalSteps[i],
#                       workers=1,
#                        max_q_size=250,
#                        pickle_safe=True)
#    np.save('./result/feibuganran/multiScale_'+str(i)+'_normalTest',result)


#第二批数据集多尺度测试
cropXYlength=(173,577,223,446,316,316,387,258,446,223,500,200,244,816,316,632,447,447,547,365,632,316,705,282,346,1154,447,894,632,632,774,516,894,447,1000,400,
              424,1414,547,1094,774,774,948,632,1094,547,1222,489,490,1633,632,1264,894,894,1095,730,1322,529)
numPerPatch=[64,64,64,64,64,64,32,32,32,32,32,32,16,16,16,16,16,16,8,8,8,8,8,8,4,4,4,4,4]

abnormalSteps=[3928,3928,3928,3928,3928,3928,7856,7856,7856,7856,7856,7856,15712,15712,15712,15712,15712,15712,31424,31424,31424,31424,31424,31424,
               62848,62848,62848,62848,62848]

normalSteps=[10000,10000,10000,10000,10000,10000,20000,20000,20000,20000,20000,20000,40000,40000,40000,40000,40000,40000,
             80000,80000,80000,80000,80000,80000,160000,160000,160000,160000,160000]

for i in xrange(29):
    result=model.predict_generator(generator=test_multiScale_SPP_generator(test_img_path,cropXYlength[2*i],cropXYlength[2*i+1],16,numPerPatch[i]),
                        steps=normalSteps[i],
                        workers=1,
                        max_q_size=250,
                        pickle_safe=True)
    np.save('./result/stage2Data/local_multi_scale/normal/multiScale_'+str(i)+'_normalTest',result)
