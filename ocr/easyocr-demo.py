import easyocr
import time

start = time.time()

reader = easyocr.Reader(["ch_sim", "en"])
RST = reader.readtext('../images/15.jpeg')

end = time.time()
duration = end - start
print(duration)
print(RST)