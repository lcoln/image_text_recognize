# import cv2

# img = cv2.imread('./images/demo/origin.png', cv2.IMREAD_COLOR)
# gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
# kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (3, 3))
# _,RedThresh = cv2.threshold(gray, 127, 255,cv2.THRESH_BINARY)

# binarization = cv2.dilate(RedThresh, kernel, 1, (2, 2), 1, cv2.BORDER_CONSTANT, 9)
# # binarization = cv2.erode(RedThresh, kernel, 1, (2, 2), 5)
# # binarization = cv2.morphologyEx(RedThresh, cv2.MORPH_OPEN, kernel, 1)
# # binarization = cv2.morphologyEx(RedThresh, cv2.MORPH_CLOSE, kernel)
# # ss = np.hstack((img, opening))

# cv2.imshow('binarization', binarization)
# cv2.waitKey()
# # cv2.imwrite('images/demo/close.png', binarization)

# # img1 = cv2.imread('./images/13.jpeg', cv2.IMREAD_REDUCED_GRAYSCALE_2)
# # img2 = cv2.imread('./images/13.jpeg', cv2.IMREAD_REDUCED_COLOR_2)
# # img3 = cv2.imread('./images/13.jpeg', cv2.IMREAD_REDUCED_GRAYSCALE_4)
# # img4 = cv2.imread('./images/13.jpeg', cv2.IMREAD_REDUCED_COLOR_4)
# # img5 = cv2.imread('./images/13.jpeg', cv2.IMREAD_REDUCED_GRAYSCALE_8)
# # img6 = cv2.imread('./images/13.jpeg', cv2.IMREAD_IGNORE_ORIENTATION)

# # print(img[50, 50])
# # print(img1[50, 50])
# # print(img2[50, 50])
# # print(img3[50, 50])
# # print(img4[50, 50])
# # print(img5[50, 50])
# # print(img6[50, 50])
# # cv2.imshow('img', img)
# # cv2.waitKey()
# # cv2.imshow('img1', img1)
# # cv2.waitKey()
# # cv2.imshow('img2', img2)
# # cv2.waitKey()
# # cv2.imshow('img3', img3)
# # cv2.waitKey()
# # cv2.imshow('img4', img4)
# # cv2.waitKey()
# # cv2.imshow('img5', img5)
# # cv2.waitKey()
# # cv2.imshow('img6', img6)
# # cv2.waitKey()

# # import numpy as np
# # arr = np.zeros([2, 2, 2], np.uint64)
# # print('result', arr)

# # import numpy as np
# # arr = np.array([5,6,88,0,-7,1,-4])

# # # print(np.argsort(arr,axis=0))

# # x = np.array([[6,88,1],[0,-7,5]])
# # # 沿着行向下(每列)的元素进行排序
# # print(np.argsort(x,axis=0))
# # # 沿着列向右(每行)的元素进行排序
# # print(np.argsort(x,axis=1))



import re
a = 'fdas.,as34.tf发生'
# print(a.encode().isalpha())

pattern = re.compile('^[A-Za-z0-9.,:;!?()_*"\' ]+$')
print(pattern.fullmatch(a))