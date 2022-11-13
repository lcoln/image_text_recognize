

# # for it in out:



# def get_text_color(originImg, cannyImg):

#     # 指定范围为3*3的矩阵，kernel（卷积核核）指定为全为1的33矩阵，卷积计算后，该像素点的值等于以该像素点为中心的3*3范围内的最大值。
#     # kernel = np.ones((3, 3),np.uint8)
#     kernel = cv2.getStructuringElement(cv2.MORPH_RECT,(2, 2))

#     # 灰度
#     gray_img = cv2.cvtColor(originImg, cv2.COLOR_BGR2GRAY)
#     # 二值化
#     _,RedThresh = cv2.threshold(gray_img,127,255,cv2.THRESH_BINARY)
#     # 腐蚀，由于我们是二值图像，所以只要包含周围黑的部分，就变为黑的。
#     # binarization = cv2.erode(RedThresh,kernel)
#     # 膨胀，由于我们是二值图像，所以只要包含周围白的部分，就变为白的。
#     binarization = cv2.dilate(RedThresh,kernel)

#     cnts,hierarchy = cv2.findContours(binarization,cv2.RETR_TREE,cv2.CHAIN_APPROX_SIMPLE)
#     if len(cnts)>=2 and (cnts[1] & cnts[1][0] & cnts[1][0][0]).any():
#         point = list(cnts[1][0][0])
#         # test
#         # img=cv2.drawContours(originImg,cnts,-1,(0,0,0),1)
#         # cv2.imshow('new_image', img)
#         # cv2.waitKey()
#     else:
#         point = None
#     if point:
#         color = tuple(originImg[point[1], point[0]])
#     else:
#         color = (0,0,0)
#     return color

# print(out)

# from paddleocr import PaddleOCR,draw_ocr
# # Paddleocr supports Chinese, English, French, German, Korean and Japanese.
# # You can set the parameter `lang` as `ch`, `en`, `fr`, `german`, `korean`, `japan`
# # to switch the language model in order.
# ocr = PaddleOCR(use_angle_cls=True, lang='en') # need to run only once to download and load model into memory
# img_path = 'images/10.png'
# result = ocr.ocr(img_path, cls=True)