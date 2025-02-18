# import matplotlib.pyplot as plt
import os

import matplotlib.pyplot as plt
from tensorflow.keras import models
import numpy as np
import dataSet

os.environ['TF_FORCE_GPU_ALLOW_GROWTH'] = 'true'


'''
geoPath = "../../data/ERA5/geopotential2000-2001.nc"
prePath = "../../data/ERA5/totalPrecipitation1995.nc"
geonpyPath = "./2000-2001_geo.npy"
prenpyPath = "./pre1995.npy"
x, y = dataSet.build_dataset(geoPath, prePath, geonpyPath, prenpyPath)

num = x.shape[0]
'''

'''
# 随机取trainset
random_index = np.random.randint(0, int(num * 0.8), [200])
train_rand_x = x[1000:1200]
train_rand_y = y[1000:1200]
# np.save("./visualization/train_rand_label.npy", random_index)
np.save("./visualization/train_seq_x.npy", train_rand_x)
np.save("./visualization/train_seq_y.npy", train_rand_y)

x_test = x[int(num * 0.8):, :]
x = None
y = y[:, :, :, None]
y_test = y[int(num * 0.8):, :]
y = None
print(y_test.shape)
'''

geo_height = np.load("./2000-2009_heavyrain_0829.npz")["geo"][0:1800]
index = np.load("./2000-2009_heavyrain_0829.npz")["index"]
gt_list = np.load("./2000-2009_heavyrain_0829.npz")["precipitation"][:1800]


# 存储所有的预测结果
'''
files = os.listdir("./models/0322/")
for name in files:
    if name.split('_')[1] == "0.5":
        model_path = "./models/0322/" + name
        model = models.load_model(model_path, compile=False)  # 只做预测，不训练，可以关掉compile
        y_pred = model.predict(geo_height)
        np.save("./pred/2000-2009heavyRain/"+os.path.splitext(name)[0]+".npy", y_pred)
'''




files = os.listdir("./models/0322/")
for name in files:
    # if os.path.splitext(name)[1] == ".h5":
    if name.split('_')[1] == "5":
        model_path = "./models/0322/"+name
        model = models.load_model(model_path, compile=False)  # 只做预测，不训练，可以关掉compile
        # simiindex = [1000, 1001, 1537, 1539, 1538, 1002, 100, 1526, 1596, 85, 1536]
        # simiindex = [1000, 961, 1508, 598, 599, 1596, 1694,  13, 659,  881,  695]
        simiindex = [1440, 1410, 10, 1061, 42, 44, 1062, 188, 896, 38]
        simidata =  [geo_height[i] for i in simiindex]
        simiresult = model.predict(np.array(simidata))
        fig = plt.figure()
        for num in range(0, len(simiresult)):
            ax = fig.add_subplot(3, 4, num + 1)
            ax.axis('off')
            ax.imshow(simiresult[num, :, :, 0], cmap="plasma", vmin=0, vmax=30)
        plt.suptitle("1441 similar data")
        plt.show()


        # np.save(save_root+"y_pred_{}.npy".format(name.split('.')[0]), y_pred)
        # print(name)




prediction_files = os.listdir("./pred/2000-2009heavyRain")
selectedID = 1441
compare_list = []
for name in prediction_files:
    if (len(name.split('_'))> 4):
        if (name.split('_')[1] == "2"):
            result = np.array(np.load("./pred/2000-2009heavyRain/"+name)[selectedID]).reshape((80, 120))
            compare_list.append(result)
compare_list.append(gt_list[selectedID])
fig = plt.figure()
for num in range(0, len(compare_list)):
    ax = fig.add_subplot(2, 3, num + 1)
    ax.axis('off')
    im = ax.imshow(np.array(compare_list)[num, :, :], cmap='plasma', vmin=0, vmax=30)
plt.show()
