import pytesseract
from PIL import Image
import time

start = time.time()

# 打开一张图片
image = Image.open('../images/15.jpeg')
# 提取中文，如果是提取英文，则先下载语言包，然后设置以下参数lang='eng'即可。
code = pytesseract.image_to_string(image, lang='chi_sim')

end = time.time()
duration = end - start

print(duration)
print(code)