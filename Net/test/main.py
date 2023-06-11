import Hilbert as hl
import cv2
from matplotlib import pyplot as plt

im = cv2.imread("test.jpg")
im = cv2.resize(im, (256,256))
plt.imshow(im)
plt.title("Original Image")
plt.show()
########展平############
result = hl.HilbertFlatten(im, 8)


########重建############
#较慢方法，在函数内部计算希尔伯特曲线，每次调用都需重新计算
im2 = hl.HilbertBuild(result, 8)
#较快方法，在函数外部计算希尔伯特曲线
hilbert8 = hl.getHilbert(8)
im3 = hl.HilbertBuild2(result,hilbert8,8)

plt.imshow(result)
plt.title("image flatteen")
plt.show()
plt.imshow(im2)
plt.title("Rebuild Image")
plt.show()
plt.imshow(im3)
plt.title("Rebuild Image(Fast)")
plt.show()
