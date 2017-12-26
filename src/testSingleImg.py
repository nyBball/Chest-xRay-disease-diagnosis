#encoding: utf-8
import SimpleITK as sitk
import scipy.ndimage as sdimg
import numpy as np
from keras.models import load_model

itkimage = sitk.ReadImage(imgPath)
img1cFirst = sitk.GetArrayFromImage(itkimage)
y = img1cFirst.shape[1]
x = img1cFirst.shape[2]
#resize into 2990*2990
x_rescale = 2990 * 1.0 / x
y_rescale = 2990 * 1.0 / y
img1cLast = np.reshape(img1cFirst, (y, x, 1))
resizeImg1c = sdimg.zoom(img1cLast, [y_rescale, x_rescale, 1], prefilter=False)
#Z-score normalize
resizeImg = np.empty([resizeImg1c.shape[0], resizeImg1c.shape[1], 3], dtype='float32')
mean = np.mean(resizeImg1c)
std = np.std(resizeImg1c)
resizeImg[:, :, ] = (resizeImg1c - mean) / std

#19*19 crop to 299*299
startX, startY = [], []
X=[]
intervalX = (x - 299) / 18
intervalY = (y - 299) / 18
for i in xrange(19):
    tmp_startX = intervalX * i
    startX.append(tmp_startX)
for i in xrange(19):
    tmp_startY = intervalY * i
    startY.append(tmp_startY)
for i in xrange(19):
    for j in xrange(19):
        tmp_img = resizeImg[startY[j]:startY[j] + 299, startX[i]:startX[i] + 299, ]
        X.append(tmp_img)

cropImgArr=np.asarray(X)
model=load_model('/media/nie/Data/model/chestRay/stage1Data/all_crop/xRayModel-02-0.94.hdf5')
result=model.predict(cropImgArr)

abnormalMaxPro=np.max(result[:,1])
print 'abnormal probability is ',abnormalMaxPro

