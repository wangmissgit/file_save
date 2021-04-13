import cv2
import numpy as np
import matplotlib.pyplot as plt


img = cv2.imread("lena.jpg")
(b,g,r) = cv2.split(img)
img = cv2.merge([r, g, b])
kernel = np.array([[0, -1, 0], [0, 5, 0], [0, -1, 0]])  # 定义卷积核
imageEnhance = cv2.filter2D(img,-1, kernel)  # 进行卷积运算

fig, ax = plt.subplots(nrows=1, ncols=2, figsize=(8, 5))

ax[0].imshow(img)
ax[0].axis('off')
ax[0].set_title('Original')


ax[1].imshow(imageEnhance)
ax[1].axis('off')
ax[1].set_title('imageEnhance_Laplacian')


fig.tight_layout()
fig.savefig('enhance_laplacian.jpg')
plt.show()
