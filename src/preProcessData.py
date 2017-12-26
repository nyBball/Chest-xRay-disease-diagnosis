#encoding: utf-8
from utils import *
import os,shutil
import SimpleITK as sitk
import numpy as np
import pydicom,json
from joblib import Parallel,delayed
import matplotlib.pyplot as plt
import matplotlib.patches as patches
from keras.utils import np_utils
from keras.preprocessing.image import flip_axis,random_shift,random_rotation,random_zoom
import math

#读取图片病变属性信息，并保存到txt文件
def countDises(imgPath):
    imgs = os.listdir(imgPath)
    print len(imgs)
    with open('DR1countNew.txt', 'w') as f:
        for idx,img in enumerate(imgs):
            if idx%10==0:
                print idx
            try:
                ds = pydicom.read_file(imgPath + img, force=True)
                infojson = ds[0x6000, 0x800].value
                info = json.loads(infojson)
                itkimage = sitk.ReadImage(imgPath+img)
            except:
                print 'err ',img
                shutil.copy(imgPath+img,'/media/nie/Data/data/JiuFeng/DR2/err/2/'+img)
                continue
            img1cFirstArr = sitk.GetArrayFromImage(itkimage)
            x = img1cFirstArr.shape[1]
            y = img1cFirstArr.shape[2]
            for i in xrange(int(info['markCount'])):
                f.write(img + ' ' +str(x)+' '+str(y)+' '+ info['marks'][i]['type'] + ' ' + str(info['marks'][i]['tl']['x']) + ' ' + str(
                    info['marks'][i]['tl']['y'])
                        + ' ' + str(info['marks'][i]['tr']['x']) + ' ' + str(info['marks'][i]['tr']['y'])
                        + ' ' + str(info['marks'][i]['bl']['x']) + ' ' + str(info['marks'][i]['bl']['y'])
                        + ' ' + str(info['marks'][i]['br']['x']) + ' ' + str(info['marks'][i]['br']['y'])
                        + ' ' + str(info['marks'][i]['angle']) + '\n')

#按病变面积区分成global、local两类疾病
def moveDisesRatio(imgPath):
    imgs=os.listdir(imgPath)
    f = open('./DR2count.txt')
    content = f.readlines()
    f.close()
    for idx,img in enumerate(imgs):
        if idx%10==0:
            print idx
        positArea=[]
        for k in xrange(len(content)):
            if img == content[k].split(' ')[0]:
                positArr = [int(int(content[k].split(' ')[2]) ),
                            int(int(content[k].split(' ')[3]) ),
                            int(int(content[k].split(' ')[8]) ),
                            int(int(content[k].split(' ')[9]) )]
                tmp_area=(positArr[2]-positArr[0])*(positArr[3]-positArr[1])
                positArea.append(tmp_area)
        if len(positArea)==0:
            print img,len(positArea)
            shutil.move(imgPath+img,'/media/nie/Data/data/JiuFeng/DR2/noAbnormal/'+img)
        else:
            positArea.sort()
            positMaxArea=positArea[-1]
            positMinArea=positArea[0]
            try:
                itkimage = sitk.ReadImage(imgPath+img)
            except:
                print 'err',img
                shutil.move(imgPath + img, '/media/nie/Data/data/JiuFeng/DR2/err/' + img)
                continue
            img1cFirstArr = sitk.GetArrayFromImage(itkimage)
            allArea=img1cFirstArr.shape[1]*img1cFirstArr.shape[2]
            maxRatio=positMaxArea*1.0/allArea
            minRatio=positMinArea*1.0/allArea
            if maxRatio>=0.1:
                shutil.copy(imgPath+img,'/media/nie/Data/data/JiuFeng/DR2/global/'+img)
            elif minRatio<0.1:
                shutil.copy(imgPath + img, '/media/nie/Data/data/JiuFeng/DR2/local/' + img)



#画正交比、面积统计直方图
def plotHist():
    ratioList, areaList=[],[]
    ratio1Area,ratio2Area,ratio3Area,ratio4Area,ratio5Area,ratioArea=[],[],[],[],[],[]
    area1Ratio,area2Ratio,area3Ratio,area4Ratio,areaRatio=[],[],[],[],[]
    imgPath='/media/nie/Data/data/JiuFeng/DR2/local/'
    imgs=os.listdir(imgPath)
    f=open('//home/nie/PycharmProjects/chestRay/result/count/DR2/DR2count.txt')
    content=f.readlines()
    f.close()
    for idx,img in enumerate(imgs):
        if idx%10==0:
            print idx
        for k in xrange(len(content)):
            if content[k].split(' ')[0]==img:
                # img_width=int(content[k].split(' ')[2])
                # img_height = int(content[k].split(' ')[1])
                # x_rescale = 2990 * 1.0 / img_width
                # y_rescale = 2990 * 1.0 / img_height
                # rec_width = math.floor((int(content[k].split(' ')[10]) - int(content[k].split(' ')[4]))*x_rescale)
                # rec_height = math.floor((int(content[k].split(' ')[11]) - int(content[k].split(' ')[5]))*y_rescale)

                rec_width = math.floor((int(content[k].split(' ')[10]) - int(content[k].split(' ')[4])))
                rec_height = math.floor((int(content[k].split(' ')[11]) - int(content[k].split(' ')[5])))
                if rec_width<=0 or rec_height<=0:
                    print content[k].split(' ')[0]
                    continue
                area=rec_width*rec_height
                ratio=rec_width*1.0/rec_height
                areaList.append(area)
                ratioList.append(ratio)


                if ratio<=0.75:
                    ratio1Area.append(area)
                elif ratio>0.75 and ratio<=1.25:
                    ratio2Area.append(area)
                elif ratio > 1.25 and ratio <= 1.75:
                    ratio3Area.append(area)
                elif ratio > 1.75 and ratio <= 2.25:
                    ratio4Area.append(area)
                else:
                    ratio5Area.append(area)

                # if area<=150000:
                #     area1Ratio.append(ratio)
                # elif area>150000 and area<=250000:
                #     area2Ratio.append(ratio)
                # elif area > 250000 and area <= 450000:
                #     area3Ratio.append(ratio)
                # else:
                #     area4Ratio.append(ratio)

    ratioArea.append(ratio1Area)
    ratioArea.append(ratio2Area)
    ratioArea.append(ratio3Area)
    ratioArea.append(ratio4Area)
    ratioArea.append(ratio5Area)

    # areaRatio.append(area1Ratio)
    # areaRatio.append(area2Ratio)
    # areaRatio.append(area3Ratio)
    # areaRatio.append(area4Ratio)

    for i in xrange(5):
        plt.subplot(1,5,i+1)
        plt.hist(ratioArea[i],bins=200)
    plt.show()

    # for i in xrange(4):
    #     plt.subplot(1,4,i+1)
    #     plt.hist(areaRatio[i],bins=200)
    # plt.show()


    # plt.subplot(121)
    # plt.hist(areaList, bins=2000)
    # plt.title('area')
    # plt.subplot(122)
    # plt.hist(ratioList, bins=2000)
    # plt.xlim(0,6)
    # plt.title('width/length')
    # plt.show()

	
#统计10*10裁剪后，abnormal patch和normal patch数量比例，以决定对abnormal patch增广多少倍
def countCropANratio(imgs,tid,resizeLength=2990,cropLength=299,cropNum=10):
    numNormal=0
    numAbnormal=0

    for idx in range(tid,len(imgs),12):
        if idx%10==0:
         print idx
        positArr = [0, 0, 0, 0]
        try:
            itkimage = sitk.ReadImage('/media/nie/Data/data/JiuFeng/DR1/backup/taxonomy/FeiBuGanRan/' + imgs[idx])
        except:
            print 'load error', imgs[idx]
            continue

        img1cFirstArr = sitk.GetArrayFromImage(itkimage)
        x = img1cFirstArr.shape[1]
        y = img1cFirstArr.shape[2]
        x_rescale = resizeLength*1.0 / x
        y_rescale = resizeLength*1.0 / y

        start_X, start_Y = [], []
        intervalX = (resizeLength - cropLength) / (cropNum-1)
        intervalY = (resizeLength - cropLength) / (cropNum-1)
        for startXid in xrange(cropNum):
            tmp_startX = intervalX * startXid
            start_X.append(tmp_startX)
        for startYid in xrange(cropNum):
            tmp_startY = intervalY * startYid
            start_Y.append(tmp_startY)
        startX = np.asarray(start_X)
        startY = np.asarray(start_Y)


        for k in xrange(len(content)):
            if imgs[idx] == content[k].split(' ')[0] and content[k].split(' ')[1]=='FeiBuGanRan':
                positArr = [int(int(content[k].split(' ')[2])*x_rescale),
                            int(int(content[k].split(' ')[3])*y_rescale),
                            int(int(content[k].split(' ')[8])*x_rescale),
                            int(int(content[k].split(' ')[9])*y_rescale)]

        for i in xrange(cropNum):
            for j in xrange(cropNum):
                ratio = calIou([startX[i], startY[j], startX[i] + cropLength, startY[j] + cropLength], positArr)
                if ratio < 0.5:
                    numNormal+=1
                else:
                    numAbnormal+=1

    print 'numNormal',numNormal,'numAbnormal',numAbnormal
    return numNormal,numAbnormal
	
	
#离线保存10*10裁剪训练集patch
def saveLocalPatch(imgs,tid,resizeLength=2990,cropLength=299,cropNum=10,augTimes=100):
    for idx in range(tid,len(imgs),12):
        print idx
        positArr = [0, 0, 0, 0]
        img, label, = [], []
        start_X, start_Y = [], []
        try:
            imgArr,x_rescale,y_rescale = loadImg(imgs[idx],resizeLength,'processData')
        except:
            print 'load error', imgs[idx]
            continue
        x, y, z = imgArr.shape

        intervalX = (x - cropLength) / (cropNum-1)
        intervalY = (y - cropLength) / (cropNum-1)
        for startXid in xrange(cropNum):
            tmp_startX = intervalX * startXid
            start_X.append(tmp_startX)
        for startYid in xrange(cropNum):
            tmp_startY = intervalY * startYid
            start_Y.append(tmp_startY)

        for k in xrange(len(content)):
            if imgs[idx].split('/')[-1] == content[k].split(' ')[0] and content[k].split(' ')[1]=='FeiBuGanRan':
                positArr = [int(int(content[k].split(' ')[2])*x_rescale),
                            int(int(content[k].split(' ')[3])*y_rescale),
                            int(int(content[k].split(' ')[8])*x_rescale),
                            int(int(content[k].split(' ')[9])*y_rescale)]
        for i in xrange(cropNum):
            for j in xrange(cropNum):
                tmp_img = imgArr[start_X[i]:start_X[i] + cropLength, start_Y[j]:start_Y[j] + cropLength, ]
                ratio = calIou([start_X[i], start_Y[j], start_X[i] + cropLength, start_Y[j] + cropLength], positArr)

                if ratio < 0.5:
                    img.append(tmp_img)
                    label.append(0)

                else:
                    img.append(tmp_img)
                    label.append(1)

                    for k in xrange(augTimes/3):
                        tmp_imgArr_shift = random_shift(tmp_img, 0.1, 0.1, 0, 1, 2, 'reflect')
                        tmp_imgArr_zoom = random_zoom(tmp_img, [0.85, 1.15], 0, 1, 2, 'reflect')
                        tmp_imgArr_rotate = random_rotation(tmp_img, 15, 0, 1, 2, 'reflect')
                        img.append(tmp_imgArr_shift)
                        img.append(tmp_imgArr_zoom)
                        img.append(tmp_imgArr_rotate)
                        label.append(1)
                        label.append(1)
                        label.append(1)
        cropImg = np.asarray(img)
        cropLabel = np_utils.to_categorical(label, 2)
        for saveIdx in xrange(cropLabel.shape[0]):
            np.savez('/media/nie/Data/data/JiuFeng/DR1/oneDises/FeiBuGanRan/valPatch/'+str(idx)+'_'+str(saveIdx),patch=cropImg[saveIdx,:,:,],label=cropLabel[saveIdx])


#离线保存resize 512*512 patch
def saveGlobalPatch(imgs,tid):
    for idx in range(tid,len(imgs),10):
        print idx
        try:
            itkimage = sitk.ReadImage('/media/nie/2T/data/JiuFeng/DR2/abnormal/' + imgs[idx])
        except:
            print 'err',imgs[idx]
            continue
        img1cFirstArr = sitk.GetArrayFromImage(itkimage)
        x=img1cFirstArr.shape[1]
        y=img1cFirstArr.shape[2]
        x_rescale=512.0/x
        y_rescale=512.0/y

        img1cArr=np.reshape(img1cFirstArr,(x,y,1))

        resizeImg1cArr=sdimg.zoom(img1cArr, [x_rescale, y_rescale,1], prefilter=False)

        #normalize
        resizeImgArr=np.empty([resizeImg1cArr.shape[0],resizeImg1cArr.shape[1],1],dtype='float32')
        mean=np.mean(resizeImg1cArr)
        std=np.std(resizeImg1cArr)
        resizeImgArr[:, :,] = (resizeImg1cArr-mean)/std
        # resizeImgArr_shift = random_shift(resizeImgArr, 0.1, 0.1, 0, 1, 2, 'reflect')
        # resizeImgArr_zoom = random_zoom(resizeImgArr, [0.85, 1.15], 0, 1, 2, 'reflect')

        np.savez('/media/nie/Data/data/JiuFeng/DR2/globalPatch/abnormal/' + str(idx),#normal abnormal 此处序号切记要改 725
                 patch=resizeImgArr, label=np_utils.to_categorical(1,2))#normal abnormal 此处标签切记要改
        # np.savez('/media/nie/Data/data/JiuFeng/DR/global/trainPatch/' + str(idx)+'_1',
        #          patch=resizeImgArr_shift, label=np_utils.to_categorical(1, 2))
        # np.savez('/media/nie/Data/data/JiuFeng/DR/global/trainPatch/' + str(idx)+'_2',
        #          patch=resizeImgArr_zoom, label=np_utils.to_categorical(1, 2))









