#encoding: utf-8
from utils import *
import os
import math
from keras.preprocessing.image import Iterator
from config import *

#-------train生成器--------
					
#10*10裁剪训练
class TrainBatchGenerator(Iterator):
    def __init__(self,datalist,batch_size,shuffle=True,seed=None):
        self.datalist=datalist
        self.batch_size=batch_size
        super(TrainBatchGenerator,self).__init__(len(datalist),batch_size,shuffle,seed)
    def next(self):
        with self.lock:
            index_array,current_index,current_batch_size=next(self.index_generator)
        batch_x=np.zeros((current_batch_size,299,299,3),dtype='float32')
        batch_y=np.zeros((current_batch_size,2),dtype='int32')
        for i,j in enumerate(index_array):
            sample=np.load('/media/nie/Data/data/JiuFeng/DR1/oneDises/FeiBuGanRan/trainPatch/'+self.datalist[j])
            patch=sample['patch']
            label=sample['label']
            batch_x[i,:,:,]=patch
            batch_y[i]=label
        return batch_x,batch_y

class ValBatchGenerator(Iterator):
    def __init__(self,datalist,batch_size,shuffle=True,seed=None):
        self.datalist=datalist
        self.batch_size=batch_size
        super(ValBatchGenerator,self).__init__(len(datalist),batch_size,shuffle,seed)
    def next(self):
        with self.lock:
            index_array,current_index,current_batch_size=next(self.index_generator)
        batch_x=np.zeros((current_batch_size,299,299,3),dtype='float32')
        batch_y=np.zeros((current_batch_size,2),dtype='int32')
        for i,j in enumerate(index_array):
            sample=np.load('/media/nie/Data/data/JiuFeng/DR1/oneDises/FeiBuGanRan/valPatch/'+self.datalist[j])
            patch=sample['patch']
            label=sample['label']
            batch_x[i,:,:,]=patch
            batch_y[i]=label
        return batch_x,batch_y
	
	
#all resiz成512*512进行训练
def all_resize_generator(imgsList):
    counter=0
    X,Y=[],[]
    while(1):
        #np.random.shuffle(imgsList)
        for idx,img in enumerate(imgsList):
            try:
                itkimage = sitk.ReadImage(img)
            except:
                print 'load error ',img
                continue
            img1cFirstArr = sitk.GetArrayFromImage(itkimage)
            x = img1cFirstArr.shape[1]
            y = img1cFirstArr.shape[2]
            x_rescale = 512.0 / x
            y_rescale = 512.0 / y

            img1cArr = np.reshape(img1cFirstArr, (x, y, 1))

            resizeImg1cArr = sdimg.zoom(img1cArr, [x_rescale, y_rescale, 1], prefilter=False)

            # normalize
            resizeImg3cArr = np.empty([resizeImg1cArr.shape[0], resizeImg1cArr.shape[1], 3],
                                      dtype='float32')  # train/test generator为3，save patch时为1以节省空间
            mean = np.mean(resizeImg1cArr)
            std = np.std(resizeImg1cArr)
            resizeImg3cArr[:, :, ] = (resizeImg1cArr - mean) / std

            X.append(resizeImg3cArr)
            if img.split('/')[7]=='abnormal':
                Y.append(np_utils.to_categorical(1, 2))
            else:
                Y.append(np_utils.to_categorical(0, 2))
            counter+=1
            if counter==4:
                X_batch = np.asarray(X)
                Y_batch = np.asarray(Y)
                Y_batch_reshape=np.reshape(Y_batch,(4,2))
                yield X_batch, Y_batch_reshape
                counter = 0
                X, Y = [], []
	
	
#在线单一尺度切割后resize训练
def onlineLocalGenerator(abnormalList,normalList,txtPath,abnormalNum,normalNum,resizeLength=1794,dstLength=299,numPerPatch=64):
    f = open(txtPath)
    content = f.readlines()
    f.close()
    abnormalIndex,normalIndex=0,0
    X, Y = [], []
    while (1):
        if abnormalIndex==0:
            np.random.shuffle(abnormalList)
        if normalIndex==0:
            np.random.shuffle(normalList)

        abnormalImg=abnormalList[abnormalIndex]
        normalImg=normalList[normalIndex]

        positArea,localPositList=[],[]
        abnormalIndex+=1
        normalIndex+=1

        if abnormalIndex==abnormalNum:
            abnormalIndex=0
        if normalIndex==normalNum:
            normalIndex=0
        try:
            abnormalImgArr,abnormal_x_rescale,abnormal_y_rescale = onlineLocalLoadImg('/media/nie/Data/data/JiuFeng/DR2/local/' + abnormalImg)
        except:
            print 'abnormal load err ',abnormalImg
            continue
        try:
            normalImgArr, normal_x_rescale, normal_y_rescale = onlineLocalLoadImg('/media/nie/2T/data/JiuFeng/DR2/normal/' + normalImg)
        except:
            print 'normal load err ', normalImg
            continue
        #count local abnormal position list
        for idx1 in xrange(len(content)):
            if abnormalImg == content[idx1].split(' ')[0]:
                positTmpArr = [int(int(content[idx1].split(' ')[2])*abnormal_x_rescale), int(int(content[idx1].split(' ')[3])*abnormal_y_rescale),
                            int(int(content[idx1].split(' ')[8])*abnormal_x_rescale), int(int(content[idx1].split(' ')[9])*abnormal_y_rescale)]

                tmp_area = (positTmpArr[2] - positTmpArr[0]) * (positTmpArr[3] - positTmpArr[1])
                if positTmpArr[0]>0 and positTmpArr[1]>0 and positTmpArr[2]>0 and positTmpArr[3]>0 and \
                                positTmpArr[0] < resizeLength and positTmpArr[1] < resizeLength and positTmpArr[2] < resizeLength and positTmpArr[3] < resizeLength and\
                                (positTmpArr[2] - positTmpArr[0])>=25 and (positTmpArr[3] - positTmpArr[1])>=25 and tmp_area*1.0/(resizeLength**2) < 0.1:
                    localPositList.append(positTmpArr)
        #abnormal img abnormal position crop
        for idx2 in xrange(len(localPositList)):
            cropCounter=0
            ite1Counter=0
            while True:
                ite1Counter+=1
                if ite1Counter==10000:
                    print 'abImg abPosit exceed ',abnormalImg,localPositList[idx2]
                    break
                left_up_x=np.random.randint(localPositList[idx2][0]-dstLength+100,localPositList[idx2][2]-100)
                left_up_y=np.random.randint(localPositList[idx2][1]-dstLength+100,localPositList[idx2][3]-100)
                if left_up_x>0 and left_up_y>0 and left_up_x+dstLength<resizeLength and left_up_y+dstLength<resizeLength:
                    #print 'abnormal img abnormal position crop ', localPositList,left_up_x,left_up_y,calIou([left_up_x,left_up_y,left_up_x+dstLength,left_up_y+dstLength],localPositList[idx2]),cropCounter
                    if calIou([left_up_x,left_up_y,left_up_x+dstLength,left_up_y+dstLength],localPositList[idx2])>=0.6:
                        cropCounter+=1
                        tmp_img=abnormalImgArr[left_up_x:left_up_x+dstLength,left_up_y:left_up_y+dstLength,]
                        X.append(tmp_img)
                        Y.append(1)
                        if cropCounter==math.ceil(numPerPatch*1.0/(4*len(localPositList))):
                            break
        #abnormal img normal position crop
        for idx3 in xrange(numPerPatch/4):
            ite2Counter=0
            while True:
                ite2Counter+=1
                if ite2Counter==10000:
                    print 'abImg noPosit exceed ',abnormalImg
                    break
                iouList=[]
                left_up_x = np.random.randint(50, resizeLength-50-dstLength)
                left_up_y = np.random.randint(50, resizeLength-50-dstLength)
                for localPosit in localPositList:
                    iouList.append(calIou([left_up_x, left_up_y, left_up_x + dstLength, left_up_y + dstLength], localPosit))
                iouArr=np.asarray(iouList)
                if (iouArr<0.2).all()==True:
                    tmp_img = abnormalImgArr[left_up_x:left_up_x + dstLength, left_up_y:left_up_y + dstLength, ]
                    X.append(tmp_img)
                    Y.append(0)
                    break
        #normal img crop
        for idx4 in xrange(numPerPatch/2):
            left_up_x = np.random.randint(50, resizeLength - 50-dstLength)
            left_up_y = np.random.randint(50, resizeLength - 50-dstLength)
            tmp_img = normalImgArr[left_up_x:left_up_x + dstLength, left_up_y:left_up_y + dstLength, ]
            X.append(tmp_img)
            Y.append(0)

        X_batch = np.asarray(X)
        Y_batch=np_utils.to_categorical(Y, 2)
        shuffle_in_unison_scary(X_batch, Y_batch)
        # print X_batch.shape,Y_batch.shape
        yield X_batch,Y_batch
        X, Y = [], []

		
#在线多尺度切割后resize训练
def onlineMultiScaleGenerator(abnormalList,normalList,txtPath,abnormalNum,normalNum,cropXYlength,mode):
    #cropXYlength tuple,从左到右，从上到下的xy值，
    f = open(txtPath)
    content = f.readlines()
    f.close()
    abnormalIndex,normalIndex=0,0
    X, Y = [], []
    X_resize_3cImg=[]
    while (1):
        if abnormalIndex==0:
            np.random.shuffle(abnormalList)
        if normalIndex==0:
            np.random.shuffle(normalList)

        abnormalImg=abnormalList[abnormalIndex]
        normalImg=normalList[normalIndex]

        positArea,localPositList=[],[]
        abnormalIndex+=1
        normalIndex+=1

        if abnormalIndex==abnormalNum:
            abnormalIndex=0
        if normalIndex==normalNum:
            normalIndex=0

        if mode=='train':
            try:
                abnormalImgArr = onlineMultiScaleLoadImg('/media/nie/Data/data/JiuFeng/DR1/oneDises/FeiBuGanRan/clean/train_abnormal/' + abnormalImg)
            except:
                print 'abnormal load err ',abnormalImg
                continue
            try:
                normalImgArr = onlineMultiScaleLoadImg('/media/nie/Data/data/JiuFeng/DR1/oneDises/FeiBuGanRan/clean/train_normal/' + normalImg)
            except:
                print 'normal load err ', normalImg
                continue
        elif mode=='val':
            try:
                abnormalImgArr = onlineMultiScaleLoadImg('/media/nie/Data/data/JiuFeng/DR1/oneDises/FeiBuGanRan/clean/val_abnormal/' + abnormalImg)
            except:
                print 'abnormal load err ',abnormalImg
                continue
            try:
                normalImgArr = onlineMultiScaleLoadImg('/media/nie/Data/data/JiuFeng/DR1/oneDises/FeiBuGanRan/clean/val_normal/' + normalImg)
            except:
                print 'normal load err ', normalImg
                continue

        normalImgX = normalImgArr.shape[1]
        normalImgY = normalImgArr.shape[0]
        abnormalImgX = abnormalImgArr.shape[1]
        abnormalImgY = abnormalImgArr.shape[0]

        #count local abnormal position list
        for idx1 in xrange(len(content)):
            if abnormalImg == content[idx1].split(' ')[0] and content[idx1].split(' ')[3]=='FeiBuGanRan':
                positTmpArr = [int(content[idx1].split(' ')[4]), int(content[idx1].split(' ')[5]),
                            int(content[idx1].split(' ')[10]), int(content[idx1].split(' ')[11])]

                if positTmpArr[0]>0 and positTmpArr[1]>0 and positTmpArr[2]>0 and positTmpArr[3]>0 and \
                                positTmpArr[0] < abnormalImgX and positTmpArr[1] < abnormalImgY and \
                                positTmpArr[2] < abnormalImgX and positTmpArr[3] < abnormalImgY and\
                                (positTmpArr[2] - positTmpArr[0])>=10 and (positTmpArr[3] - positTmpArr[1])>=10:
                    localPositList.append(positTmpArr)

        #abnormal img abnormal position crop
        for idx2 in xrange(len(localPositList)):
            cropCounter=0
            ite1Counter=0
            while True:
                ite1Counter+=1
                if ite1Counter==10000:
                    print 'abImg abPosit exceed ',localPositList[idx2],positWidth*1.0/positHeight,positWidth*positHeight
                    break
                positWidth=localPositList[idx2][2]-localPositList[idx2][0]
                positHeight=localPositList[idx2][3]-localPositList[idx2][1]

                if positWidth * 1.0 / positHeight <= 0.75:

                    if positWidth * positHeight <= 150000:
                        cropXlength = cropXYlength[0]
                        cropYlength = cropXYlength[1]
                    elif positWidth * positHeight > 150000 and positWidth * positHeight <= 300000:
                        cropXlength = cropXYlength[8]
                        cropYlength = cropXYlength[9]
                    elif positWidth * positHeight > 300000 and positWidth * positHeight <= 500000:
                        cropXlength = cropXYlength[16]
                        cropYlength = cropXYlength[17]
                    else:
                        cropXlength = cropXYlength[22]
                        cropYlength = cropXYlength[23]

                elif positWidth * 1.0 / positHeight <= 1.25 and positWidth * 1.0 / positHeight > 0.75:

                    if positWidth * positHeight <= 150000:
                        cropXlength = cropXYlength[2]
                        cropYlength = cropXYlength[3]
                    elif positWidth * positHeight > 150000 and positWidth * positHeight <= 300000:
                        cropXlength = cropXYlength[10]
                        cropYlength = cropXYlength[11]
                    elif positWidth * positHeight > 300000 and positWidth * positHeight <= 500000:
                        cropXlength = cropXYlength[18]
                        cropYlength = cropXYlength[19]
                    else:
                        cropXlength = cropXYlength[24]
                        cropYlength = cropXYlength[25]

                elif positWidth * 1.0 / positHeight > 1.25 and positWidth * 1.0 / positHeight <= 1.75:

                    if positWidth * positHeight <= 150000:
                        cropXlength = cropXYlength[4]
                        cropYlength = cropXYlength[5]
                    elif positWidth * positHeight > 150000 and positWidth * positHeight <= 250000:
                        cropXlength = cropXYlength[12]
                        cropYlength = cropXYlength[13]
                    elif positWidth * positHeight > 250000 and positWidth * positHeight <= 350000:
                        cropXlength = cropXYlength[20]
                        cropYlength = cropXYlength[21]
                    else:
                        cropXlength = cropXYlength[26]
                        cropYlength = cropXYlength[27]

                else:
                    if positWidth * positHeight <= 150000:
                        cropXlength = cropXYlength[6]
                        cropYlength = cropXYlength[7]
                    else:
                        cropXlength = cropXYlength[14]
                        cropYlength = cropXYlength[15]

                left_up_x = np.random.randint(localPositList[idx2][0] - cropXlength ,
                                              localPositList[idx2][2] )
                left_up_y = np.random.randint(localPositList[idx2][1] - cropYlength ,
                                              localPositList[idx2][3] )
                if left_up_x > 0 and left_up_y > 0 and left_up_x + cropXlength < abnormalImgX and left_up_y + cropYlength < abnormalImgY:
                    if calIou([left_up_x, left_up_y, left_up_x + cropXlength, left_up_y + cropYlength],localPositList[idx2]) >= 0.6:
                        cropCounter += 1
                        tmp_img = abnormalImgArr[left_up_y:left_up_y + cropYlength,left_up_x:left_up_x + cropXlength,]
                        X.append(tmp_img)
                        Y.append(1)
                        if cropCounter == math.ceil(56.0 / (4 * len(localPositList))):
                            break


        #abnormal img normal position crop
        for idx3 in xrange(14):
            if idx3>=14:
                idx3-=14
            ite2Counter=0
            while True:
                ite2Counter+=1
                if ite2Counter==10000:
                    print 'abImg noPosit exceed ',abnormalImg,cropXYlength[2*idx3],cropXYlength[2*idx3+1]
                    break
                iouList=[]
                left_up_x = np.random.randint(50, abnormalImgX-50-cropXYlength[2*idx3])
                left_up_y = np.random.randint(50, abnormalImgY-50-cropXYlength[2*idx3+1])
                for localPosit in localPositList:
                    iouList.append(calIou([left_up_x, left_up_y, left_up_x + cropXYlength[2*idx3], left_up_y + cropXYlength[2*idx3+1]], localPosit))
                iouArr=np.asarray(iouList)
                if (iouArr<0.1).all()==True:
                    tmp_img = abnormalImgArr[left_up_y:left_up_y + cropXYlength[2*idx3+1],left_up_x:left_up_x + cropXYlength[2*idx3], ]
                    X.append(tmp_img)
                    Y.append(0)
                    break


        #normal img crop
        for idx4 in xrange(14):#32
            if idx4>=14 and idx4<28:
                idx4-=14
            # elif idx4>=28:
            #     idx4 -= 28
            left_up_x = np.random.randint(50, normalImgX - 50-cropXYlength[2*idx4])
            left_up_y = np.random.randint(50, normalImgY - 50-cropXYlength[2*idx4+1])
            tmp_img = normalImgArr[left_up_y:left_up_y + cropXYlength[2*idx4+1],left_up_x:left_up_x + cropXYlength[2*idx4], ]
            X.append(tmp_img)
            Y.append(0)

        for resizeIdx in xrange(len(X)):
            patchX=X[resizeIdx].shape[1]
            patchY = X[resizeIdx].shape[0]
            x_rescale=316.0/patchX
            y_rescale = 316.0 / patchY
            resizeImg1cArr = sdimg.zoom(X[resizeIdx], [y_rescale, x_rescale, 1], prefilter=False)
            resizeImg3cArr = np.empty([316, 316, 3], dtype='float32')
            resizeImg3cArr[:, :, ] = resizeImg1cArr


            X_resize_3cImg.append(resizeImg3cArr)
        X_batch = np.asarray(X_resize_3cImg)
        Y_batch=np_utils.to_categorical(Y, 2)
        shuffle_in_unison_scary(X_batch, Y_batch)

        yield X_batch,Y_batch
        X, Y = [], []
        X_resize_3cImg = []


		
#在线多尺度切割后SPP训练
def onlineMultiScaleSPPGenerator(abnormalList,normalList,content,abnormalNum,normalNum,cropXYlength,numPerPatch,mode):
    #cropXYlength tuple,从左到右，从上到下的xy值,numPerPatch为一个同尺度数目相同数量的tuple
    abnormalIndex,normalIndex=0,0
    X, Y = [], []
    scaleImg=[[] for col in range(29)]
    scaleLabel=[[] for col in range(29)]
    allCounter=0
    positScaleIndex=[]
    while (1):
        allCounter+=1
        if abnormalIndex==0:
            np.random.shuffle(abnormalList)
        if normalIndex==0:
            np.random.shuffle(normalList)

        abnormalImg=abnormalList[abnormalIndex]
        normalImg=normalList[normalIndex]

        positArea,localPositList=[],[]
        abnormalIndex+=1
        normalIndex+=1

        if abnormalIndex==abnormalNum-1:
            abnormalIndex=0
        if normalIndex==normalNum-1:
            normalIndex=0

        if mode=='train':
            try:
                abnormalImgArr = onlineMultiScaleLoadImg(train_abnormal_path + abnormalImg)
            except:
                print 'abnormal load err ',abnormalImg
                continue
            try:
                normalImgArr = onlineMultiScaleLoadImg(train_normal_path + normalImg)
            except:
                print 'normal load err ', normalImg
                continue
        elif mode=='val':
            try:
                abnormalImgArr = onlineMultiScaleLoadImg(val_abnormal_path + abnormalImg)
            except:
                print 'abnormal load err ',abnormalImg
                continue
            try:
                normalImgArr = onlineMultiScaleLoadImg(val_normal_path + normalImg)
            except:
                print 'normal load err ', normalImg
                continue

        normalImgX = normalImgArr.shape[1]
        normalImgY = normalImgArr.shape[0]
        abnormalImgX = abnormalImgArr.shape[1]
        abnormalImgY = abnormalImgArr.shape[0]

        #count local abnormal position list
        for idx1 in xrange(len(content)):
            # if abnormalImg == content[idx1].split(' ')[0] and content[idx1].split(' ')[3]=='FeiBuGanRan':
            if abnormalImg == content[idx1].split(' ')[0]:
                positTmpArr = [int(content[idx1].split(' ')[4]), int(content[idx1].split(' ')[5]),
                            int(content[idx1].split(' ')[10]), int(content[idx1].split(' ')[11])]

                if positTmpArr[0]>0 and positTmpArr[1]>0 and positTmpArr[2]>0 and positTmpArr[3]>0 and \
                                positTmpArr[0] < abnormalImgX and positTmpArr[1] < abnormalImgY and \
                                positTmpArr[2] < abnormalImgX and positTmpArr[3] < abnormalImgY and\
                                (positTmpArr[2] - positTmpArr[0])>=10 and (positTmpArr[3] - positTmpArr[1])>=10:
                    localPositList.append(positTmpArr)

        #abnormal img abnormal position crop
        for idx2 in xrange(len(localPositList)):
            cropCounter=0
            ite1Counter=0
            positWidth = localPositList[idx2][2] - localPositList[idx2][0]
            positHeight = localPositList[idx2][3] - localPositList[idx2][1]

			#肺部感染多尺度
            # if positWidth * 1.0 / positHeight <= 0.75:
            #
            #     if positWidth * positHeight <= 150000:
            #         scaleIndex = 0
            #     elif positWidth * positHeight > 150000 and positWidth * positHeight <= 300000:
            #         scaleIndex = 4
            #     elif positWidth * positHeight > 300000 and positWidth * positHeight <= 500000:
            #         scaleIndex = 8
            #     else:
            #         scaleIndex = 11
            #
            # elif positWidth * 1.0 / positHeight <= 1.25 and positWidth * 1.0 / positHeight > 0.75:
            #
            #     if positWidth * positHeight <= 150000:
            #         scaleIndex = 1
            #     elif positWidth * positHeight > 150000 and positWidth * positHeight <= 300000:
            #         scaleIndex = 5
            #     elif positWidth * positHeight > 300000 and positWidth * positHeight <= 500000:
            #         scaleIndex = 9
            #     else:
            #         scaleIndex = 12
            #
            # elif positWidth * 1.0 / positHeight > 1.25 and positWidth * 1.0 / positHeight <= 1.75:
            #
            #     if positWidth * positHeight <= 150000:
            #         scaleIndex = 2
            #     elif positWidth * positHeight > 150000 and positWidth * positHeight <= 250000:
            #         scaleIndex = 6
            #     elif positWidth * positHeight > 250000 and positWidth * positHeight <= 350000:
            #         scaleIndex = 10
            #     else:
            #         scaleIndex = 13
            #
            # else:
            #     if positWidth * positHeight <= 150000:
            #         scaleIndex = 3
            #     else:
            #         scaleIndex = 7

			#第二批所有数据多尺度
            if  positWidth * 1.0 / positHeight <=0.3 :

                if positWidth * positHeight <= 150000:
                    scaleIndex = 0
                elif positWidth * positHeight > 150000 and positWidth * positHeight <= 300000:
                    scaleIndex = 6
                elif positWidth * positHeight > 300000 and positWidth * positHeight <= 500000:
                    scaleIndex = 12
                elif positWidth * positHeight > 500000 and positWidth * positHeight <= 700000:
                    scaleIndex = 18
                else:
                    scaleIndex = 24


            elif  positWidth * 1.0 / positHeight >0.3 and positWidth * 1.0 / positHeight <= 0.75:

                if positWidth * positHeight <= 150000:
                    scaleIndex = 1
                elif positWidth * positHeight > 150000 and positWidth * positHeight <= 300000:
                    scaleIndex = 7
                elif positWidth * positHeight > 300000 and positWidth * positHeight <= 500000:
                    scaleIndex = 13
                elif positWidth * positHeight > 500000 and positWidth * positHeight <= 700000:
                    scaleIndex = 19
                else:
                    scaleIndex = 25

            elif positWidth * 1.0 / positHeight <= 1.25 and positWidth * 1.0 / positHeight > 0.75:

                if positWidth * positHeight <= 150000:
                    scaleIndex = 2
                elif positWidth * positHeight > 150000 and positWidth * positHeight <= 300000:
                    scaleIndex = 8
                elif positWidth * positHeight > 300000 and positWidth * positHeight <= 500000:
                    scaleIndex = 14
                elif positWidth * positHeight > 500000 and positWidth * positHeight <= 700000:
                    scaleIndex = 20
                else:
                    scaleIndex = 26

            elif positWidth * 1.0 / positHeight > 1.25 and positWidth * 1.0 / positHeight <= 1.75:

                if positWidth * positHeight <= 150000:
                    scaleIndex = 3
                elif positWidth * positHeight > 150000 and positWidth * positHeight <= 300000:
                    scaleIndex = 9
                elif positWidth * positHeight > 300000 and positWidth * positHeight <= 500000:
                    scaleIndex = 15
                elif positWidth * positHeight > 500000 and positWidth * positHeight <= 700000:
                    scaleIndex = 21
                else:
                    scaleIndex = 27


            elif positWidth * 1.0 / positHeight > 1.75 and positWidth * 1.0 / positHeight <= 2.25:

                if positWidth * positHeight <= 150000:
                    scaleIndex = 4
                elif positWidth * positHeight > 150000 and positWidth * positHeight <= 300000:
                    scaleIndex = 10
                elif positWidth * positHeight > 300000 and positWidth * positHeight <= 500000:
                    scaleIndex = 16
                else:
                    scaleIndex = 22

            else:
                if positWidth * positHeight <= 150000:
                    scaleIndex = 5
                elif positWidth * positHeight > 150000 and positWidth * positHeight <= 300000:
                    scaleIndex = 11
                elif positWidth * positHeight > 300000 and positWidth * positHeight <= 500000:
                    scaleIndex = 17
                elif positWidth * positHeight > 500000 and positWidth * positHeight <= 700000:
                    scaleIndex = 23
                else:
                    scaleIndex = 28



            cropXlength = cropXYlength[2 * scaleIndex]
            cropYlength = cropXYlength[2 * scaleIndex + 1]
            positScaleIndex.append(scaleIndex)
            while True:
                ite1Counter+=1
                if ite1Counter==10000:
                    print 'abImg abPosit exceed ',localPositList[idx2],positWidth*1.0/positHeight,positWidth*positHeight
                    break
                left_up_x = np.random.randint(localPositList[idx2][0] - cropXlength ,
                                              localPositList[idx2][2] )
                left_up_y = np.random.randint(localPositList[idx2][1] - cropYlength ,
                                              localPositList[idx2][3] )
                if left_up_x > 0 and left_up_y > 0 and left_up_x + cropXlength < abnormalImgX and left_up_y + cropYlength < abnormalImgY:
                    if calIou([left_up_x, left_up_y, left_up_x + cropXlength, left_up_y + cropYlength],localPositList[idx2]) >= 0.6:
                        cropCounter += 1
                        tmp_img = abnormalImgArr[left_up_y:left_up_y + cropYlength,left_up_x:left_up_x + cropXlength,]
                        X.append(tmp_img)
                        Y.append(1)
                        if cropCounter == math.ceil(numPerPatch[scaleIndex]*1.0/ (4 * len(localPositList))):
                            break


        #abnormal img normal position crop
        for index in xrange(len(positScaleIndex)):
            for idx3 in xrange(numPerPatch[positScaleIndex[index]]/(4*len(positScaleIndex))):
                ite2Counter=0
                while True:
                    ite2Counter+=1
                    if ite2Counter==10000:
                        print 'abImg noPosit exceed ',abnormalImg,cropXYlength[2*idx3],cropXYlength[2*idx3+1]
                        break
                    iouList=[]
                    left_up_x = np.random.randint(50, abnormalImgX-50-cropXYlength[2*positScaleIndex[index]])
                    left_up_y = np.random.randint(50, abnormalImgY-50-cropXYlength[2*positScaleIndex[index]+1])
                    for localPosit in localPositList:
                        iouList.append(calIou([left_up_x, left_up_y, left_up_x + cropXYlength[2*positScaleIndex[index]], left_up_y + cropXYlength[2*positScaleIndex[index]+1]], localPosit))
                    iouArr=np.asarray(iouList)
                    if (iouArr<0.3).all()==True:
                        tmp_img = abnormalImgArr[left_up_y:left_up_y + cropXYlength[2*positScaleIndex[index]+1],left_up_x:left_up_x + cropXYlength[2*positScaleIndex[index]], ]
                        X.append(tmp_img)
                        Y.append(0)
                        break
        # print 'abnormalIndex crop end ', abnormalIndex-1


        #normal img crop
        for index in xrange(len(positScaleIndex)):
            if normalImgX - 50 - cropXYlength[2 * positScaleIndex[index]] > 50 and normalImgY - 50 - cropXYlength[2 * positScaleIndex[index] + 1] > 50:
                for idx4 in xrange(2*numPerPatch[positScaleIndex[index]]/(4*len(positScaleIndex))):
                    left_up_x = np.random.randint(50, normalImgX - 50-cropXYlength[2*positScaleIndex[index]])
                    left_up_y = np.random.randint(50, normalImgY - 50-cropXYlength[2*positScaleIndex[index]+1])
                    tmp_img = normalImgArr[left_up_y:left_up_y + cropXYlength[2*positScaleIndex[index]+1],left_up_x:left_up_x + cropXYlength[2*positScaleIndex[index]], ]
                    X.append(tmp_img)
                    Y.append(0)
        # print 'normalIndex crop end ', normalIndex-1
        #不同尺度patch到预设值时yield
        for resizeIdx in xrange(len(X)):
            patchX = X[resizeIdx].shape[1]
            patchY = X[resizeIdx].shape[0]
            for j in xrange(29):
                if patchX==cropXYlength[2*j] and patchY==cropXYlength[2*j+1]:
                    img3cArr = np.empty([patchY, patchX, 3], dtype='float32')
                    img3cArr[:, :, ] = X[resizeIdx]
                    scaleImg[j].append(img3cArr)
                    scaleLabel[j].append(Y[resizeIdx])

        for i in xrange(29):
            if len(scaleLabel[i])>=numPerPatch[i]:
                X_batch = np.asarray(scaleImg[i])
                Y_batch = np_utils.to_categorical(scaleLabel[i], 2)
                X_batch_setNum=X_batch[0:numPerPatch[i]]
                Y_batch_setNum = Y_batch[0:numPerPatch[i]]
                # print 'shape ',X_batch_setNum.shape,' ',Y_batch_setNum.shape
                # shuffle_in_unison_scary(X_batch_setNum, Y_batch_setNum)
                yield X_batch_setNum, Y_batch_setNum
                scaleImg[i], scaleLabel[i] = [], []
        # print 'normalIndex yield end ', normalIndex - 1
        X, Y = [], []
        positScaleIndex=[]
		
		
#-------test生成器---------

#10*10裁剪测试			
def test_local_generator(imgPath,resizeLength,cropLength,cropNum):
    X_list=[]
    counter=0
    while(1):
        imgs=os.listdir(imgPath)
        for idx,img in enumerate(imgs):
            print idx
            img3cArr = loadImg(imgPath + img,resizeLength,'generator',3)
            x, y, z = img3cArr.shape
            startX, startY = [], []
            intervalX = (x - cropLength) / (cropNum-1)
            intervalY = (y - cropLength) / (cropNum-1)
            for i in xrange(cropNum):
                tmp_startX = intervalX * i
                startX.append(tmp_startX)
            for i in xrange(cropNum):
                tmp_startY = intervalY * i
                startY.append(tmp_startY)
            for i in xrange(cropNum):
                for j in xrange(cropNum):
                    tmp_img = img3cArr[startX[i]:startX[i] + cropLength, startY[j]:startY[j] + cropLength, ]
                    X_list.append(tmp_img)
                    counter += 1
                    if counter==128:
                        X_Arr=np.asarray(X_list)
                        #print X_Arr.shape
                        yield X_Arr
                        counter=0
                        X_list = []


#all resize成512*512测试						
def test_global_generator(imgPath):
    X_list=[]
    counter=0
    while(1):
        imgs=os.listdir(imgPath)
        for idx,img in enumerate(imgs):
            print idx
            try:
                itkimage = sitk.ReadImage('/media/nie/Data/data/JiuFeng/DR1/backup/init/normal/' + imgs[idx])
            except:
                print 'err', imgs[idx]
                continue
            img1cFirstArr = sitk.GetArrayFromImage(itkimage)
            x = img1cFirstArr.shape[1]
            y = img1cFirstArr.shape[2]
            x_rescale = 512.0 / x
            y_rescale = 512.0 / y

            img1cArr = np.reshape(img1cFirstArr, (x, y, 1))

            resizeImg1cArr = sdimg.zoom(img1cArr, [x_rescale, y_rescale, 1], prefilter=False)

            # normalize
            resizeImgArr = np.empty([resizeImg1cArr.shape[0], resizeImg1cArr.shape[1], 3], dtype='float32')
            mean = np.mean(resizeImg1cArr)
            std = np.std(resizeImg1cArr)
            resizeImgArr[:, :, ] = (resizeImg1cArr - mean) / std
            X_list.append(resizeImgArr)
            counter+=1
            if counter==24:
                X_Arr = np.asarray(X_list)
                # print X_Arr.shape
                yield X_Arr
                counter = 0
                X_list = []

				
#在线多尺度切割后resize测试				
def test_multiScale_generator(imgPath,cropXlength,cropYlength,cropNum):
    X=[]
    X_resize_3cImg = []
    counter=0
    while(1):
        imgs=os.listdir(imgPath)
        for idx,img in enumerate(imgs):
            print idx
            try:
                itkimage = sitk.ReadImage('/media/nie/Data/data/JiuFeng/DR1/oneDises/FeiBuGanRan/clean/val_abnormal/'+img)
            except:
                print 'load err ',img
                continue
            img1cFirstArr = sitk.GetArrayFromImage(itkimage)
            y = img1cFirstArr.shape[1]
            x = img1cFirstArr.shape[2]
            img1cLastArr = np.reshape(img1cFirstArr, (y, x, 1))
            mean = np.mean(img1cLastArr)
            std = np.std(img1cLastArr)
            normlizeImg = np.empty([y, x, 1], dtype='float32')
            normlizeImg[:, :,] = (img1cLastArr - mean) / std

            startX, startY = [], []
            intervalX = (x - cropXlength) / (cropNum-1)
            intervalY = (y - cropYlength) / (cropNum-1)
            for i in xrange(cropNum):
                tmp_startX = intervalX * i
                startX.append(tmp_startX)
            for i in xrange(cropNum):
                tmp_startY = intervalY * i
                startY.append(tmp_startY)
            for i in xrange(cropNum):
                for j in xrange(cropNum):
                    tmp_img = normlizeImg[startY[j]:startY[j] + cropYlength,startX[i]:startX[i] + cropXlength,]
                    X.append(tmp_img)
                    counter += 1
                    if counter==128:
                        for resizeIdx in xrange(len(X)):
                            patchX = X[resizeIdx].shape[1]
                            patchY = X[resizeIdx].shape[0]
                            x_rescale = 316.0 / patchX
                            y_rescale = 316.0 / patchY
                            resizeImg1cArr = sdimg.zoom(X[resizeIdx], [y_rescale, x_rescale, 1], prefilter=False)
                            if resizeImg1cArr.shape[0] != 316 or resizeImg1cArr.shape[1] != 316:
                                print 'shape!=316'
                                continue
                            resizeImg3cArr = np.empty([316, 316, 3], dtype='float32')
                            resizeImg3cArr[:, :, ] = resizeImg1cArr
                            X_resize_3cImg.append(resizeImg3cArr)

                        X_Arr=np.asarray(X_resize_3cImg)
                        # print X_Arr.shape
                        yield X_Arr
                        counter=0
                        X = []
                        X_resize_3cImg = []

						
#在线多尺度切割后SPP测试
def test_multiScale_SPP_generator(imgPath,cropXlength,cropYlength,cropNum,numPerPatch):
    X=[]
    X_3cImg = []
    counter=0
    imgs = os.listdir(imgPath)
    while(1):
        for idx,img in enumerate(imgs):
            print idx
            try:
                itkimage = sitk.ReadImage('/media/nie/2T/data/JiuFeng/DR2/local_multiscale_normal/val/'+img)
            except:
                print 'load err ',img
                continue
            img1cFirstArr = sitk.GetArrayFromImage(itkimage)
            y = img1cFirstArr.shape[1]
            x = img1cFirstArr.shape[2]
            img1cLastArr = np.reshape(img1cFirstArr, (y, x, 1))
            mean = np.mean(img1cLastArr)
            std = np.std(img1cLastArr)
            normlizeImg = np.empty([y, x, 1], dtype='float32')
            normlizeImg[:, :,] = (img1cLastArr - mean) / std

            startX, startY = [], []
            intervalX = (x - cropXlength) / (cropNum-1)
            intervalY = (y - cropYlength) / (cropNum-1)
            for i in xrange(cropNum):
                tmp_startX = intervalX * i
                startX.append(tmp_startX)
            for i in xrange(cropNum):
                tmp_startY = intervalY * i
                startY.append(tmp_startY)
            for i in xrange(cropNum):
                for j in xrange(cropNum):
                    tmp_img = normlizeImg[startY[j]:startY[j] + cropYlength,startX[i]:startX[i] + cropXlength,]
                    X.append(tmp_img)
                    counter += 1
                    if counter==numPerPatch:
                        for resizeIdx in xrange(len(X)):
                            patchX = X[resizeIdx].shape[1]
                            patchY = X[resizeIdx].shape[0]
                            img3cArr = np.empty([patchY, patchX, 3], dtype='float32')
                            img3cArr[:, :, ] = X[resizeIdx]
                            X_3cImg.append(img3cArr)

                        X_Arr=np.asarray(X_3cImg)
                        # if X_Arr.shape!=(4, 1633, 490, 3):
                        #     print X_Arr.shape,img
                        yield X_Arr
                        counter=0
                        X = []
                        X_3cImg = []
		