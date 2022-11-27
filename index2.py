# encoding:utf-8

import sys
import easyocr
import cv2
import numpy as np
from PIL import Image, ImageDraw, ImageFont
import requests
import json
from matplotlib import pyplot as plt 

url = "http://localhost:3336/overseas/translate"
headers = {'content-type': 'application/json'}
#微软雅黑字体
fontType = './msyh.ttf'   

def transformText(text):
    # test
    # texts = []
    # requestData = {"text": text}
    # ret = requests.post(url, json=requestData, headers=headers)
    # if ret.status_code == 200:
    #     texts = json.loads(ret.text)
    #     texts = texts['data']['TargetTextList']
    # return texts
    return text

def detect(imagePath, lang):
    # 解析繁体/英文/数字
    reader = easyocr.Reader(lang)
    RST = reader.readtext(imagePath)
    
    # origin_cv_image = cv2.imread(imagePath, cv2.IMREAD_UNCHANGED)
    origin_cv_image = cv2.imread(imagePath)
    result = []
    texts = []

    for detection in RST:
        # easyocr获取到文字区域的左上角坐标
        left_top_coordinate = tuple(detection[0][0])
        # easyocr获取到文字区域的右上角坐标
        right_bottom_coordinate = tuple(detection[0][2])
        # easyocr获取到的文字
        text = detection[1]
        texts.append(text)
              
        result.append([
            left_top_coordinate, 
            right_bottom_coordinate, 
            text, 
        ])
        
    print(result)
    allTexts = transformText(texts)
    print(allTexts)
    for inx, val in enumerate(result):
        ltc, rbc, t = val
        font = ImageFont.load_default()
        #字体大小
        fontScale = int(abs(ltc[0] - rbc[0]) / len(allTexts[inx]))
        # if 'en' in lang:
        #     fontScale = fontScale * 2
            # test
            # fontScale = int(fontScale * 3)
        # 加载字体并定义其编码方式和大小


        # todo, 5个思路
        # 字体颜色
        # 1. 获取轮廓，缩小轮廓让其刚好经过线上的点，取点的颜色
        # 2. 获取轮廓，缩小轮廓让其刚好经过线上的点，取所有点的颜色的平均值
        # 3. 将原图进行腐蚀，获取轮廓，缩小轮廓让其刚好经过线上的点，取点的颜色
        # 4. 直方图，取2个波峰的值，一个是背景一个是字体颜色
        # 5. 取区域内平均色值
        # 裁剪一部分边角，去除多余部分
        color = get_text_color(
            # cv2.resize(img, (0, 0), fx=fx, fy=fy, interpolation=cv2.INTER_CUBIC)
            cv2.resize(
                origin_cv_image[ltc[1] + 10:rbc[1] - 10, ltc[0] + 10:rbc[0] - 10], 
                (0, 0),
                fx=10, 
                fy=10,
                interpolation=cv2.INTER_CUBIC
            )
            # origin_cv_image[ltc[1] + 10:rbc[1] - 10, ltc[0] + 10:rbc[0] - 10],
            # origin_cv_image[ltc[1]:rbc[1], ltc[0]:rbc[0]]
        )
        font = ImageFont.truetype(
            fontType, 
            fontScale, 
            encoding="unic"
        )
        origin_cv_image = put_text_into_image(
            origin_cv_image, ltc, rbc, allTexts[inx], font, color, fontScale
        )
    return origin_cv_image


def replace_zeroes(data):
    min_nonzero = min(data[np.nonzero(data)])
    data[data == 0] = min_nonzero
    return data

# 图像增强
def enhance(originImg, size):
    # 不污染原图
    img = originImg.copy()
    if len(img.shape)==3 and img.shape[2]==4:
        rgb = cv2.split(img)
    elif len(img.shape)==3 and img.shape[2]==3:
        rgb = cv2.split(img)  

    result = []
    for it in rgb:
        L_blur = cv2.GaussianBlur(it, (size, size), 0)
        img = replace_zeroes(it)
        L_blur = replace_zeroes(L_blur)

        dst_Img = cv2.log(img / 255.0)
        dst_Lblur = cv2.log(L_blur / 255.0)
        dst_IxL = cv2.multiply(dst_Img, dst_Lblur)
        log_R = cv2.subtract(dst_Img, dst_IxL)

        dst_R = cv2.normalize(log_R, None, 0, 255, cv2.NORM_MINMAX)
        log_uint8 = cv2.convertScaleAbs(dst_R)
        
        result.append(log_uint8)
    
    return cv2.merge(result)

# 修复图像
def repair(origin_cv_image, lt, rb):
    # 读取梯度运算后的图片
    gradient_png = cv2.imread('gradient_img.png', 0)

    # 裁剪原图坐标为[y0:y1, x0:x1]
    src_img = origin_cv_image[lt[1]:rb[1], lt[0]:rb[0]]  
    # 将通过梯度运算出来的图片与裁剪原图进行融合修复，去除掉梯度运算轨迹包括到的内容
    dst = cv2.inpaint(src_img, gradient_png, 5, cv2.INPAINT_TELEA)
    # 将修复完的图片块再放回去
    origin_cv_image[lt[1]:rb[1], lt[0]:rb[0]] = dst

    return origin_cv_image

def perimeter(poly):
    p = 0
    nums = poly.shape[0]
    for i in range(nums):
        p += abs(np.linalg.norm(poly[i % nums] - poly[(i + 1) % nums]))
    return p

def draw(img, cnts):
    cv2.imshow('描边', cv2.drawContours(img,cnts,-1,(255,73,95),5))
    cv2.waitKey()

def show(name, img):
    cv2.imshow(name, img)
    cv2.waitKey()

def calcGrayHist(grayimage):
  # 灰度图像矩阵的高，宽
  rows, cols = grayimage.shape

  # 存储灰度直方图
  grayHist = np.zeros([256], np.uint64)
  for r in range(rows):
    for c in range(cols):
      grayHist[grayimage[r][c]] += 1

  return grayHist

# 阈值分割：直方图技术法
def threshTwoPeaks(image):
  #转换为灰度图
  if len(image.shape) == 2:
    gray = image
  else:
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

  # 计算灰度直方图
  histogram = calcGrayHist(gray)
  # 寻找灰度直方图的最大峰值对应的灰度值
  maxLoc = np.where(histogram == np.max(histogram))
  # print(maxLoc)
  firstPeak = maxLoc[0][0] #灰度值
  # 寻找灰度直方图的第二个峰值对应的灰度值
  measureDists = np.zeros([256], np.float32)
  for k in range(256):
    measureDists[k] = pow(k - firstPeak, 2) * histogram[k] #综合考虑 两峰距离与峰值
  maxLoc2 = np.where(measureDists == np.max(measureDists))
  secondPeak = maxLoc2[0][0]
  print('双峰为：',firstPeak,secondPeak)
  
  return [firstPeak,secondPeak]

def get_max_color(originImg):
  img = originImg.copy()

  # 灰度
  gray_img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
  
  # 将图片数据转一维
  n, bins, patches = plt.hist(gray_img.ravel(), 256, [0, 256])
  plt.show()
  plt.close(1)
  # max_i = np.array(n).argmax(axis=0)
  max_i = np.argsort(n)[-1]
  max_i_f = np.argsort(n)[-2]
  return [max_i, max_i_f]

def get_redThresh(gray_img, min_color, max_color):
  gap = int(abs(max_color - min_color) / 2)
  
  # 黑色居多，但不知道是背景还是字
  if min_color < max_color:
    _,RedThresh = cv2.threshold(gray_img, min_color + gap, 255,cv2.THRESH_BINARY_INV)
  else:
    _,RedThresh = cv2.threshold(gray_img, max_color + gap, 255,cv2.THRESH_BINARY)
  
  return RedThresh

def get_font_color_ratio(originImg, min_color, max_color):

  img = originImg.copy()
  canny_img = cv2.Canny(img, 200, 150)
  gray_img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
  RedThresh = get_redThresh(gray_img, min_color, max_color)
  # show('RedThresh', RedThresh)

  cnts,hierarchy = cv2.findContours(RedThresh,cv2.RETR_TREE,cv2.CHAIN_APPROX_SIMPLE)
  # cnts = np.delete(cnts, 0)

  # show(RedThresh)
  # draw(img, cnts)
  
  # 计算轮廓面积，与总面积进行对比，算出字体占比，如果字体面积较大，则背景色应该相反，因为背景色取值原理为遍历所有像素点颜色进行大(255)小(0)匹配    
  area = 0
  for item in cnts:
    # print(1)
    # draw(img, [item])
    item_area = cv2.contourArea(item)
    # print(item_area)
    area += item_area

  w = originImg.shape[0]
  h = originImg.shape[1]

  return area / (w * h)

def get_text_color(originImg):
  img = originImg.copy()
  img = cv2.pyrMeanShiftFiltering(img, 10, 50)
  # 获取占比最大的颜色rgb
  # [max_color, min_color] = get_max_color(img)
  [max_color, min_color] = threshTwoPeaks(img)
  
  # 获取数值最大的颜色rgb
  max_value = max([max_color, min_color])
  min_value = min([max_color, min_color])
  print(max_color, 'max_color')
  print(min_color, 'min_color')
  # 获取占比最大的颜色

  # 获取字体轮廓内面积、计算其与总面积占比
  ratio = get_font_color_ratio(img, min_color, max_color)
  # 获取字体轮廓内面积、计算其与总面积占比

  # 计算出阈值与二值化的策略和应该填充的颜色值
  gray_img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
  # 如果字体轮廓占比大于总面积一半，则字体颜色为颜色最大值
  gap = int(abs(max_color - min_color) / 2)
  
  kernel = cv2.getStructuringElement(cv2.MORPH_RECT,(8, 8))
  print(ratio, 'ratio')
  if ratio > 0.5:  

    RedThresh = get_redThresh(gray_img, min_color, max_color)

    # if max_color > min_color:
    #   print('黑底白字1')
    #   _,RedThresh = cv2.threshold(gray_img, min_color + gap, 255,cv2.THRESH_BINARY)
    # else:
    #   print('白底黑字2')
    #   _,RedThresh = cv2.threshold(gray_img, max_color + gap, 255,cv2.THRESH_BINARY_INV)

    # # 白底黑字
    # if max_color < 127:
    #   binarization = cv2.dilate(RedThresh,kernel)
    # # 黑底白字
    # if max_color > 127:
    #   binarization = cv2.erode(RedThresh,kernel)
  else:

    # 白底黑字
    if max_color > min_color:
      print('白底黑字3')
      _,RedThresh = cv2.threshold(gray_img, min_color + gap, 255,cv2.THRESH_BINARY)
    # 黑底白字
    else:
      print('黑底白字4')
      _,RedThresh = cv2.threshold(gray_img, max_color + gap, 255,cv2.THRESH_BINARY_INV)

    # print(cv2.THRESH_BINARY_INV, 'THRESH_BINARY_INV')
    # # 黑底白字
    # if max_color < 127:
    #   binarization = cv2.erode(RedThresh,kernel)
    # # 白底黑字
    # if max_color > 127:
    #   binarization = cv2.dilate(RedThresh,kernel)

  # 统一成白底黑字
  binarization = cv2.dilate(RedThresh,kernel)

  # print(gap, 'gap')
  
  # draw(img, cnts)
  # show('gray_img', gray_img)
  # show('RedThresh', RedThresh)
  # show('binarization', binarization)

  cnts,hierarchy = cv2.findContours(binarization,cv2.RETR_TREE,cv2.CHAIN_APPROX_SIMPLE)
  if len(cnts)>=2 and (cnts[0] & cnts[0][0] & cnts[0][0][0]).any():
      point = list(cnts[0][0][0])
  else:
      point = None
  if point:
      # GBR
      color = tuple(originImg[point[1], point[0]])
  else:
      color = (0,0,0)
  return color[::-1]

  # 计算出阈值与二值化的策略和应该填充的颜色值

def put_text_into_image(origin_cv_image, lt, rb, text, font, color, fontScale):
    size = 3
    # 裁剪文本坐标，[y0:y1, x0:x1]
    text_img = origin_cv_image[lt[1]:rb[1], lt[0]:rb[0]] 
    cv2.imwrite('text_img.png', text_img)
    # 裁剪文本坐标，[y0:y1, x0:x1]

    # 增强图片
    # result = enhance(text_img, size)
    # # print(color_name)
    # cv2.imwrite('text_img.png', result)
    # 增强图片

    # 边缘检测
    img = cv2.imread('text_img.png')
    canny_img = cv2.Canny(img, 200, 150)
    cv2.imwrite('canny_img.png', canny_img)
    # 边缘检测
    # print(fontScale, 'fontscale', text)
    # 梯度运算
    rate = int(fontScale/2)
    if rate > 15:
        rate = 15
    # test
    rate = 15
    # print(rate)
    img = cv2.imread('canny_img.png', 1)
    k = np.ones((rate, rate), np.uint8)
    img2 = cv2.morphologyEx(img, cv2.MORPH_GRADIENT, k)  
    cv2.imwrite('gradient_img.png', img2)
    # 梯度运算

    # 图像修复，将通过梯度运算出来的图片与原始图像进行融合
    repair_cv_image = repair(origin_cv_image, lt, rb)
    # 图像修复

    # 在修复完的图片块上写入文本
    return write(repair_cv_image, lt, rb, text, font, color)
    # 在修复完的图片块上写入文本

def write(IMG, lt, rb, text, font, color):
    # print(color)
    # 将图片的array转换成image
    img_pil = Image.fromarray(cv2.cvtColor(IMG, cv2.COLOR_BGR2RGB))
    # 进行图像绘制
    draw = ImageDraw.Draw(img_pil)
    # 进行文本写入
    draw.text(lt, text, font=font, fill=color)  #PIL中BGR=(255,0,0)表示红色
    #PIL图片转换为numpy
    img_ocv = np.array(img_pil)                     
    IMG = cv2.cvtColor(img_ocv,cv2.COLOR_RGB2BGR)
    # 圈出图片上文字区域
    # return cv2.rectangle(IMG, lt, rb, (0,255,0), 3)
    return IMG

if __name__ == '__main__':
    # 读取文件
    # imagePath = sys.argv[1]
    _, imagePath, *lang = sys.argv
    if(len(lang) == 0):
        lang = ["ch_sim", "en"]
    new_image = detect(sys.argv[1], lang)
    
    cv2.imshow('new_image', new_image)
    cv2.waitKey()
    cv2.destroyAllWindows()