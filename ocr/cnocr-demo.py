# 字体坐标只能获取到4个顶点
from cnocr import CnOcr
import time

start = time.time()

img_fp = 'images/10.png'
ocr = CnOcr()  # 所有参数都使用默认值
out = ocr.ocr(img_fp)

end = time.time()
duration = end - start

print(duration)
print(out)