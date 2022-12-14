import tensorflow as tf
import cv2 as cv
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
from model import U_Net

res = 224
model = U_Net(n_classes=18)
model.load_weights('last_model_weights.h5')
#model weights file = #https://drive.google.com/file/d/1-E9leFxYl-nlJtHCYVDJhh7CAARumtdA/view?usp=sharing

# df = pd.read_csv(os.path.join(base_path, 'labels.csv'))
# hues = {}
# for v, i in enumerate(df['value']):
#     hues[i] = (v,int(v*180/18))

# def get_color(mask):
#     for i, v in hues.values():
#         mask[mask==i] = v
#     k = np.ones((res, res, 2))*1
#     mask = np.concatenate((tf.expand_dims(mask, axis=-1),k), axis=-1)
#     print(mask.shape)
#     # mask = np.array(cv.cvtColor(mask, cv.COLOR_HSV2RGB))
#     return mask

imgg = cv.resize(cv.cvtColor(cv.imread('any_image.png'), cv.COLOR_BGR2RGB), (res,res), interpolation=cv.INTER_NEAREST)
img = tf.expand_dims(imgg, axis=0)
img = img/255
Y = model(img)
Y = np.squeeze(np.argmax(Y, axis=-1))
# Y_rgb = get_color(Y)
plt.imshow(imgg)
plt.show()
plt.imshow(Y, cmap='jet')
plt.show()
# plt.imshow(np.add(Y_rgb/255, imgg))
# plt.show()

