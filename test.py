import cv2
import numpy as np
from matplotlib import pyplot as plt

img = cv2.imread('../images/1.png',cv2.IMREAD_GRAYSCALE)

plt.imshow(img, cmap = 'gray', interpolation = 'bicubic')
# plt.xticks([]), plt.yticks([])  # to hide tick values on X and Y axis
# plt.plot([300,300,400],[100,200,300],'c', linewidth=5)
# plt.show()

cv2.rectangle(img, (x1, y1), (x1 + w1], y1 + h1), blue, -1)
font = cv2.FONT_HERSHEY_COMPLEX
text = '标注的文字'
# 字体标注的位置， 内容，字体设置
cv2.putText(img, text, (x1, y1), font, 2, (0, 0, 255), 1)
img_save_path = '/home/shichao/save/2.jpg'
# 保存标注好的图像
cv2.imwrite(img_save_path)
plt.show()
