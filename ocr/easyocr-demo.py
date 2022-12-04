import easyocr
import time
import cv2

# origin_cv_image = cv2.imread('./images/13.jpeg')
# gray = cv2.cvtColor(origin_cv_image, cv2.COLOR_BGR2GRAY)
# print(origin_cv_image[50, 50])
# print(gray[50, 50])
start = time.time()

reader = easyocr.Reader(["ch_sim", "en"])
RST = reader.readtext('../images/15.jpeg')

end = time.time()
duration = end - start
print(duration)
print(RST)