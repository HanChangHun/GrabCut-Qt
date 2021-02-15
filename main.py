from grabcut import grabcut
from utils import show
import cv2

if __name__ == '__main__':
    img = cv2.imread("images/llama1.jpg")
    p1, p2 = (55, 0), (600, 599)
    result, _ = grabcut(img, p1, p2)
    show(result)