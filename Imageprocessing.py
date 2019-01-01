#!/usr/bin/env python
# coding: utf-8

# In[53]:


import cv2
import numpy as np
import sys
import math 
import random
from matplotlib import pyplot as plt
#====================================================================
#=============# 旋轉圖像--旋转任意角度 ================================
# 旋轉angle角度，缺失背景白色（255, 255, 255）填充
def rotate_bound_white_bg(image, angle):
    #抓住圖像的尺寸，然後確定中央
    (h, w) = image.shape[:2]
    (cX, cY) = (w // 2, h // 2)

    # -angle位置参数为角度参数负值表示顺时针旋转; 1.0位置参数scale是调整尺寸比例（图像缩放参数），建议0.75
    #抓住旋轉矩陣（應用角度的負值順時針旋轉），然後抓住正弦和余弦（即矩陣的旋轉分量）
    M = cv2.getRotationMatrix2D((cX, cY), -20*angle, 1.0)
    cos = np.abs(M[0, 0])
    sin = np.abs(M[0, 1])
 
    # compute the new bounding dimensions of the image
    #計算圖像的新邊界尺寸
    nW = int((h * sin) + (w * cos))
    nH = int((h * cos) + (w * sin))
 
    # adjust the rotation matrix to take into account translation
    #調整旋轉矩陣以考慮翻譯(若調整數字圖片大小的邊界會改變)
    M[0, 2] += (nW / 2) - cX
    M[1, 2] += (nH / 2) - cY
 
    # perform the actual rotation and return the image
    #執行實際旋轉並返回圖像
    # borderValue 缺失背景填充色彩，此处为白色，可自定義
    return cv2.warpAffine(image, M, (nW, nH),borderValue=(255,255,255))
    # borderValue 缺省，默认是黑色（0, 0 , 0）
    # return cv2.warpAffine(image, M, (nW, nH))
 
for g in range(30) :
    imgg = cv2.imread(r"D:\Users\iComputer\Desktop\imgchange\Original\realgun\rg%d.jpg" % (g))
    l = [random.randint(0, 10) for a in range(10)]
    #print(l)
    for i in l:
        imgRotation = rotate_bound_white_bg(imgg, i)
        cv2.imwrite(r"D:\Users\iComputer\Desktop\imgchange\images\realgun\Rotate(%d)" % (g)+ str(i) + '.jpg', imgRotation)
#=======================================================
#=============# 旋轉圖像--旋转4個角度 ================================
def rotate_about_center(src, angle, scale=1.):
    w = src.shape[1]
    h = src.shape[0]
    rangle = np.deg2rad(angle)  # angle in radians
    # now calculate new image width and height
    nw = (abs(np.sin(rangle)*h) + abs(np.cos(rangle)*w))*scale
    nh = (abs(np.cos(rangle)*h) + abs(np.sin(rangle)*w))*scale
    # ask OpenCV for the rotation matrix
    rot_mat = cv2.getRotationMatrix2D((nw*0.5, nh*0.5), angle, scale)
    # calculate the move from the old center to the new center combined
    # with the rotation
    rot_move = np.dot(rot_mat, np.array([(nw-w)*0.5, (nh-h)*0.5,0]))
    # the move only affects the translation, so update the translation
    # part of the transform
    rot_mat[0,2] += rot_move[0]
    rot_mat[1,2] += rot_move[1]
    return cv2.warpAffine(src, rot_mat, (int(math.ceil(nw)), int(math.ceil(nh))), flags=cv2.INTER_LANCZOS4)


for v in range(30): 
    imgv = cv2.imread(r"D:\Users\iComputer\Desktop\imgchange\Original\realgun\rg%d.jpg"%(v))
    imgRotation = rotate_about_center(imgv,90, scale=1.)
    r1=rotate_about_center(imgv,180, scale=1.)
    r2=rotate_about_center(imgv,-90, scale=1.)
    r3=rotate_about_center(imgv,360, scale=1.)
    cv2.imwrite(r"D:\Users\iComputer\Desktop\imgchange\images\realgun\Rotate90%d.jpg"%(v) , imgRotation)
    cv2.imwrite(r"D:\Users\iComputer\Desktop\imgchange\images\realgun\Rotate180%d.jpg"%(v) , r1)
    cv2.imwrite(r"D:\Users\iComputer\Desktop\imgchange\images\realgun\Rotate-90%d.jpg"%(v) , r2)
    cv2.imwrite(r"D:\Users\iComputer\Desktop\imgchange\images\realgun\Rotate360%d.jpg"%(v) , r3)
#=============# 添加雜訊 ================================
#===========# 隨機添加椒鹽的函數  #===========
def salt(img, n):
    saltImage = img.copy()  #複製圖像
    for k in range(n):   # 循環添加n個椒鹽
        i = int(np.random.random() * img.shape[1]); # 隨機選擇椒鹽的x座標
        j = int(np.random.random() * img.shape[0]); # 隨機選擇椒鹽的y座標
        if img.ndim == 2:  # 如果是灰度圖
            saltImage[j,i] = 255
        elif img.ndim == 3:  # 如果是RBG圖片
            saltImage[j,i,0]= 255
            saltImage[j,i,1]= 255
            saltImage[j,i,2]= 255
    return saltImage 
 
#讀取資料夾圖片
for o in range(30) :
    imgo = cv2.imread(r"D:\Users\iComputer\Desktop\imgchange\Original\realgun\rg%d.jpg" % (o))
    #plt.imshow(img)
    #plt.show()
    saltImage = salt(imgo,10000)#調整雜訊的密集度，此處我們設定10000
    cv2.imwrite(r"D:\Users\iComputer\Desktop\imgchange\images\realgun\Salt-result%d.jpg"% (o),saltImage)
    #plt.imshow(saltImage),plt.title('Salt')
    #plt.show()
    
#=================================================
#=================調整圖像亮度========================   
#讀取資料夾圖片
for z in range(30):
    im = cv2.imread(r"D:\Users\iComputer\Desktop\imgchange\Original\realgun\rg%d.jpg" % (z))
    # CHeck this image is color or gray
    #查看此圖像是彩色還是灰色
    if im.ndim > 1: 
        (rows, cols, c) = im.shape
    else:
        (rows, cols) = im.shape
        c=1
        # Convert BGR type to RGB type for display...
        #將BGR類型轉換為RGB類型進行顯示...
    im = cv2.cvtColor(im, cv2.COLOR_BGR2RGB)

    # Make a empty result sized of the size of original image
    #製作一個大小與原始圖像大小相同的空結果
    result = np.zeros((rows, cols, c), np.uint8)
    for i in range(c):
        I = np.reshape(im[:,:,i], [rows, cols])
        I = np.asarray(I, np.float32)
        # Get each channel (RGB) to individually process.
        #獲取每個通道（RGB）以單獨處理。
    
    #=================================================
        res1 = np.power(I, 1.1)#對比亮度增強
        res2 = np.power(res1, 0.8)#變暗
    #=================================================
        result[:,:,i] = np.asarray(res1, np.uint8)
  
    cv2.imwrite(r"D:\Users\iComputer\Desktop\imgchange\images\realgun\resultlg%d.jpg" % (z),result)
    cv2.imwrite(r"D:\Users\iComputer\Desktop\imgchange\images\realgun\resultBK%d.jpg" % (z),res2)

    imgshow = np.asarray(im, np.uint8)
    cv2.imwrite(r'D:\Users\iComputer\Desktop\imgchange\images\realgun\Original%d.jpg'% (z),imgshow)


# In[ ]:





# In[ ]:




