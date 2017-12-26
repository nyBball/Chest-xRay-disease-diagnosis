#encoding: utf-8
import SimpleITK as sitk
import numpy as np
from keras.utils import np_utils
from keras.preprocessing.image import flip_axis,random_shift,random_rotation,random_zoom
import scipy.ndimage as sdimg
import math
import os


def loadImg(imgPath,resizeLength,mode,channelNum=1):
    '''
    :param imgPath:
    :param resizeLength:
    :param mode: generator or processData
    :param channelNum: save patch is 1,generator is 3
    :return:
    '''
    itkimage = sitk.ReadImage(imgPath)
    img1cFirstArr = sitk.GetArrayFromImage(itkimage)
    y=img1cFirstArr.shape[1]
    x=img1cFirstArr.shape[2]
    x_rescale=resizeLength*1.0/x
    y_rescale=resizeLength*1.0/y

    img1cArr=np.reshape(img1cFirstArr,(x,y,1))

    resizeImg1cArr=sdimg.zoom(img1cArr, [x_rescale, y_rescale,1], prefilter=False)

    #normalize
    resizeImgArr=np.empty([resizeImg1cArr.shape[0],resizeImg1cArr.shape[1],channelNum],dtype='float32')
    mean=np.mean(resizeImg1cArr)
    std=np.std(resizeImg1cArr)
    resizeImgArr[:, :,] = (resizeImg1cArr-mean)/std

    #no normalize
    # resizeImg3cArr=np.empty([resizeImg1cArr.shape[0],resizeImg1cArr.shape[1],1],dtype='uint16')
    # resizeImg3cArr[:, :,] = resizeImg1cArr
    if mode=='processData':
        return resizeImgArr, x_rescale, y_rescale
    elif mode=='generator':
        return resizeImgArr

def onlineLocalLoadImg(imgPath):
    itkimage = sitk.ReadImage(imgPath)
    img1cFirstArr = sitk.GetArrayFromImage(itkimage)
    y=img1cFirstArr.shape[1]
    x=img1cFirstArr.shape[2]
    x_rescale=1794.0/x
    y_rescale=1794.0/y

    img1cArr=np.reshape(img1cFirstArr,(y,x,1))

    resizeImg1cArr=sdimg.zoom(img1cArr, [x_rescale, y_rescale,1], prefilter=False)

    #normalize
    resizeImg3cArr=np.empty([resizeImg1cArr.shape[0],resizeImg1cArr.shape[1],3],dtype='float32')
    mean=np.mean(resizeImg1cArr)
    std=np.std(resizeImg1cArr)
    resizeImg3cArr[:, :,] = (resizeImg1cArr-mean)/std

    return resizeImg3cArr, x_rescale, y_rescale

def onlineMultiScaleLoadImg(imgPath):
    itkimage = sitk.ReadImage(imgPath)
    img1cFirstArr = sitk.GetArrayFromImage(itkimage)
    y=img1cFirstArr.shape[1]
    x=img1cFirstArr.shape[2]

    img1cLastArr=np.reshape(img1cFirstArr,(y,x,1))
    normlizeImg = np.empty([y, x, 1], dtype='float32')
    # normalize
    mean=np.mean(img1cLastArr)
    std=np.std(img1cLastArr)
    normlizeImg[:, :,] = (img1cLastArr-mean)/std
    return normlizeImg


def shuffle_in_unison_scary(a, b):
    rng_state = np.random.get_state()
    np.random.shuffle(a)
    np.random.set_state(rng_state)
    np.random.shuffle(b)




def calIou(CropArr,positArr):
    # input duijiao axis,positArr为病变位置坐标
    x1=CropArr[0]
    y1=CropArr[1]
    width1=CropArr[2]-CropArr[0]
    height1=CropArr[3]-CropArr[1]

    x2=positArr[0]
    y2=positArr[1]
    width2=positArr[2]-positArr[0]
    height2=positArr[3]-positArr[1]

    endx=max(x1+width1,x2+width2)
    startx=min(x1,x2)
    width=width1+width2-(endx-startx)

    endy=max(y1+height1,y2+height2)
    starty=min(y1,y2)
    height=height1+height2-(endy-starty)

    if width<=0 or height<=0:
        ratio=0
    #crop in posit
    elif CropArr[0]>=positArr[0] and CropArr[1]>=positArr[1] and CropArr[2]<=positArr[2] and CropArr[3]<=positArr[3]:
        ratio=1
    #posit in crop
    elif CropArr[0]<=positArr[0] and CropArr[1]<=positArr[1] and CropArr[2]>=positArr[2] and CropArr[3]>=positArr[3]:
        ratio=1
    else:
        area=width*height
        area1=width1*height1
        area2=width2*height2
        ratio=area*1.0/(area1+area2-area)
    return ratio



def cropAugImg(img3cArr,positArr,ifAug='false',dstWNum=10,dstHNum=10,dstW=299,dstH=299,mode='uniform',Iou=0.6,numPerAug=1):
    '''
    :param dstWNum: target x crop num
    :param dstHNum: target y crop num
    :param dstW: target patch width
    :param dstH: target patch height
    :param mode: uniform or random crop
    :param Iou: Iou
    :param numPerAug: every augment method repeat time
    :return: dstWNum*dstHNum tf (channel_last) img and responding label
    '''

    img,label,=[],[]
    x,y,z=img3cArr.shape
    #print x,y,z,positArr[0],positArr[1],positArr[2],positArr[3]
    if mode=='random':
        startX = np.random.randint(0, x - dstW, size=(dstWNum))
        startY = np.random.randint(0, y - dstH, size=(dstHNum))
    elif mode=='uniform':
        startX, startY = [], []
        intervalX=(x-dstW)/(dstWNum-1)
        intervalY=(y-dstH)/(dstHNum-1)
        for i in xrange(dstWNum):
            tmp_startX=intervalX*i
            startX.append(tmp_startX)
        for i in xrange(dstHNum):
            tmp_startY=intervalY*i
            startY.append(tmp_startY)

    if ifAug=='false':
        for i in xrange(dstWNum):
            for j in xrange(dstHNum):
                tmp_img = img3cArr[startX[i]:startX[i] + dstW, startY[j]:startY[j] + dstH, ]

                ratio = calIou([startX[i], startY[j], startX[i] + dstW, startY[j] + dstH], positArr)
                if ratio < Iou:
                    tmp_label=0
                else:
                    tmp_label=1

                img.append(tmp_img)
                label.append(tmp_label)


    elif ifAug=='true':
        for i in xrange(dstWNum):
            for j in xrange(dstHNum):
                tmp_img = img3cArr[startX[i]:startX[i] + dstW, startY[j]:startY[j] + dstH, ]
                ratio = calIou([startX[i], startY[j], startX[i] + dstW, startY[j] + dstH], positArr)

                if ratio < Iou:
                    img.append(tmp_img)
                    label.append(0)

                else:
                    img.append(tmp_img)
                    label.append(1)

                    for k in xrange(numPerAug):
                        tmp_img3cArr_shift = random_shift(tmp_img, 0.1, 0.1, 0, 1, 2, 'reflect')
                        tmp_img3cArr_rotate = random_rotation(tmp_img, 30, 0, 1, 2, 'reflect')
                        tmp_img3cArr_zoom = random_zoom(tmp_img, [0.8, 1.2], 0, 1, 2, 'reflect')
                        img.append(tmp_img3cArr_shift)
                        img.append(tmp_img3cArr_rotate)
                        img.append(tmp_img3cArr_zoom)
                        label.append(1)
                        label.append(1)
                        label.append(1)
                    img.append(flip_axis(tmp_img, 0))
                    img.append(flip_axis(tmp_img, 1))
                    label.append(1)
                    label.append(1)

    cropImg=np.asarray(img)

    cropLabel=np_utils.to_categorical(label,2)
    return cropImg ,cropLabel


def cropImg(img3cArr):
    img=[]
    x, y, z = img3cArr.shape
    start_X, start_Y = [], []
    intervalX = (x - 299) / 18
    intervalY = (y - 299) / 18
    for i in xrange(19):
        tmp_startX = intervalX * i
        start_X.append(tmp_startX)
    for i in xrange(19):
        tmp_startY = intervalY * i
        start_Y.append(tmp_startY)
    startX = np.asarray(start_X)
    startY = np.asarray(start_Y)
    for i in xrange(19):
        for j in xrange(19):
            tmp_img = img3cArr[startX[i]:startX[i] + 299, startY[j]:startY[j] + 299, ]
            img.append(tmp_img)
    cropImg=np.asarray(img)
    return cropImg
