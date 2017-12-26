#encoding: utf-8
import numpy as np
import math
import matplotlib.pyplot as plt
import os
import SimpleITK as sitk
import matplotlib.patches as patches
import pylab

#------统计单一尺度每张图片最大有病概率-------
def countMaxPro(npyPath,imgNum,cropNum):
    max_list=[]
    allPro=np.load(npyPath)
    for i in xrange(imgNum):
        max=np.max(allPro[i*cropNum:(i+1)*cropNum,1])
        max_list.append(max)
    max_arr=np.asarray(max_list)
    print max_arr.shape
    np.save('local_abnormal_max',max_arr)

#-------统计单一尺度FN取固定值时，normal图片判错个数------
def countFNnormalNum(mode):
    counter=0
    if mode=='all_crop_model'or mode=='local_model':
        p_abnormal=np.load('local_abnormal_max.npy')
        p_normal = np.load('local_normal_max.npy')
        for i in xrange(2500):
            if p_normal[i]>np.sort(p_abnormal)[9]:
                counter+=1
        print counter

    elif mode=='global_model':
        abnormal_list=[]
        p_abnormal = np.load('./global_abnormal_test.npy')
        p_normal=np.load('./global_normal_test.npy')
        for i in xrange(1850):
            abnormal_list.append(p_abnormal[i][1])
        for i in xrange(7850):
            if p_normal[i][1]>np.sort(abnormal_list)[19]:
                counter+=1
        print counter

    elif mode=='global_local_model':
        abnormalNum=0
        normalNum=0
        p_abnormal_max=[]
        p_normal_max=[]
        p_abnormal_global=np.load('./result/global_local_val/global_abnormal_test.npy')
        p_abnormal_local=np.load('./result/global_local_val/local_abnormal_max.npy')
        p_normal_global=np.load('./result/global/normal_test.npy')
        p_normal_local=np.load('./result/local/normal_max.npy')
        for i in xrange(489):
            if p_abnormal_global[i][1]<=p_abnormal_local[i]:
                p_abnormal_max.append(p_abnormal_local[i])
            else:
                p_abnormal_max.append(p_abnormal_global[i][1])
        # print np.sort(p_abnormal_max)
        # for j in xrange(356):
        #     if p_abnormal_max[j]<0.5:
        #         abnormalNum+=1

        for i in xrange(1322):
            if p_normal_global[i][1]<=p_normal_local[i]:
                p_normal_max.append(p_normal_local[i])
            else:
                p_normal_max.append(p_normal_global[i][1])
        for j in xrange(1322):
            if p_normal_max[j]>np.sort(p_abnormal_max)[5]:
                normalNum+=1
        print abnormalNum,normalNum


#------统计多尺度每张图片每个尺度最大有病概率-------
def countMulScalePro(scaleNum,imgNum,cropNum):
    #cropNum:an img total crop num
    #return a numpy array compise of maxValue and maxPosit
    for i in xrange(scaleNum):
        npyPath='/home/nie/PycharmProjects/chestRay/result/stage2Data/local_multiScale/abnormal/test/multiScale_'+str(i)+'_abnormalTest.npy'
        max_list = []
        allPro = np.load(npyPath)
        for j in xrange(imgNum):
            max = np.max(allPro[j * cropNum:(j + 1) * cropNum, 1])
            max_list.append(max)
            maxPosit = np.argmax(allPro[j * cropNum:(j + 1) * cropNum, 1])
            max_list.append(maxPosit)
        max_arr = np.asarray(max_list)
        print max_arr.shape
        np.save('/home/nie/PycharmProjects/chestRay/result/stage2Data/local_multiScale/abnormal/max/multiScale_'+str(i)+'_abnormalMax.npy', max_arr)

#-----统计多尺度FN取固定值时，normal图片判错个数------
def countMultiScaleFNnormalNum(scaleNum,normalNum,abnormalNum):
    counter=0
    p_allAbnormal,p_allNormal=[],[]
    abnormal_max_list,normal_max_list = [],[]
    for i in xrange(scaleNum):
        p_singleAbnormal=np.load('/home/nie/PycharmProjects/chestRay/result/stage2Data/local_multiScale/abnormal/max/multiScale_'+str(i)+'_abnormalMax.npy')
        p_allAbnormal.append(p_singleAbnormal)
        p_allAbnormal_array=np.asarray(p_allAbnormal)

        p_singleNormal=np.load('/home/nie/PycharmProjects/chestRay/result/stage2Data/local_multiScale/normal/max/multiScale_'+str(i)+'_normalMax.npy')
        p_allNormal.append(p_singleNormal)
        p_allNormal_array=np.asarray(p_allNormal)

    for j in xrange(abnormalNum):
        max=np.max(p_allAbnormal_array[:,2*j])
        abnormal_max_list.append(max)
    for k in xrange(normalNum):
        max=np.max(p_allNormal_array[:,2*k])
        normal_max_list.append(max)
    # print abnormal_max_list
    min=sorted(abnormal_max_list)[9]
    print min
    for z in xrange(normalNum):
        if normal_max_list[z] > min:
            counter += 1
    print counter







