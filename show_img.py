# import cv2
# import numpy as np

# def showImage(img):
    

#     cv_img = cv2.cvtColor(np.array(img), cv2.COLOR_RGB2BGR)

#     cv2.imshow("Detected Image", cv_img)
#     cv2.waitKey(0)
#     cv2.destroyAllWindows()

import matplotlib.pyplot as plt

def showImage(img):
    plt.imshow(img)
    plt.axis('off')
    plt.title("Detected Image")
    plt.show()