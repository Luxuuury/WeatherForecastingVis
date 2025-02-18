import copy

# import matplotlib.pyplot as plt
import matplotlib.pyplot as plt
import numpy as np
import dataSet
import json
import os
import warnings

from sklearn.manifold import TSNE
import sklearn.metrics as skm
from sklearn.metrics import confusion_matrix, precision_score, recall_score, accuracy_score, f1_score, mean_squared_error
from sklearn.preprocessing import MinMaxScaler
from scipy import stats
import scipy.ndimage
import seaborn as sns

warnings.filterwarnings('ignore')


def ts(preds, labels, threshold, modelname):
    ts = []
    for theta in threshold:
        obs = np.where(labels >= theta, 1, 0)
        pre = np.where(preds >= theta, 1, 0)
        # True positive (TP)
        hits = np.sum((obs == 1) & (pre == 1))
        # False negative (FN)
        misses = np.sum((obs == 1) & (pre == 0))
        # False positive (FP)
        falsealarms = np.sum((obs == 0) & (pre == 1))
        # True negative (TN)
        correctnegatives = np.sum((obs == 0) & (pre == 0))
        TS = hits / (hits+misses+falsealarms)
        ts.append(TS)

    return modelname, threshold, ts

def ts_level(preds, labels, theta_1, theta_2, modelname):
    ts = []

    obs = np.where((labels >= theta_1) & (labels < theta_2), 1, 0)
    pre = np.where((preds >= theta_1) & (preds < theta_2), 1, 0)
    # True positive (TP)
    hits = np.sum((obs == 1) & (pre == 1))
    # False negative (FN)
    misses = np.sum((obs == 1) & (pre == 0))
    # False positive (FP)
    falsealarms = np.sum((obs == 0) & (pre == 1))
    # True negative (TN)
    correctnegatives = np.sum((obs == 0) & (pre == 0))
    TS = hits / (hits+misses+falsealarms)
    ts.append(TS)

    return modelname, [theta_1, theta_2], ts

# 降水数据转降雨等级
def data2rainlevel(data):
    newdata = np.where((data <= 0.5), 0, data)
    newdata = np.where(((newdata > 0.5)&(newdata<=1)), 1, newdata)
    newdata = np.where(((newdata > 1)&(newdata<=2)), 2, newdata)
    newdata = np.where(((newdata > 2)&(newdata<=5)), 3, newdata)
    newdata = np.where((newdata > 5), 4, newdata)
    return newdata

# 降水数据  转  5个降水量级格点个数
def data2raingridnum(data):
    newdata = []
    for i in range(len(data)):
        one = np.count_nonzero(data <= 0.5)
        two = np.count_nonzero((data > 0.5)&(data  <= 1))
        three = np.count_nonzero((data > 1)&(data  <= 2))
        four = np.count_nonzero((data > 2)&(data  <= 5))
        five = np.count_nonzero(data > 5)
        newdata.append([one, two, three, four, five])
    return newdata

# 评估latent space中每个grid的影响力
def pd_grid(perturb, pred, gt, th_a, th_b):
    mask = np.array(copy.deepcopy(gt))
    mask[(mask <= th_a) | (mask > th_b)] = 0
    mask[(mask>th_a) & (mask<=th_b)] = 1

    perturb_m = perturb * mask
    pred_m = pred * mask
    non_zero = np.count_nonzero(mask)
    if(non_zero > 0):
        pd_value = np.sum(perturb_m - pred_m) / non_zero
    else:
        pd_value = 0
    return pd_value

# 按perturb和pred的格子变化数衡量影响力
def pd_grid_v2(perturb, pred, th_a, th_b):
    perturb_m = copy.deepcopy(perturb)
    perturb_m[(perturb_m <= th_a) | (perturb_m > th_b)] = 0
    perturb_m[(perturb_m > th_a) & (perturb_m <= th_b)] = 1
    perturb_num = np.count_nonzero(perturb_m)

    pred_m = copy.deepcopy(pred)
    pred_m[(pred_m <= th_a) | (pred_m > th_b)] = 0
    pred_m[(pred_m > th_a) & (pred_m <= th_b)] = 1
    pred_num = np.count_nonzero(pred_m)

    if(pred_num == 0):
        print(perturb_num, th_a, th_b)
        return  perturb_num
    else:
        return (perturb_num - pred_num) / pred_num

# 按perturb和pred的格子变化数衡量影响力
# 同时处理一个batchsize的数据
def pd_grid_v3(perturb, pred, th_a, th_b):
    perturb_m = copy.deepcopy(perturb)
    perturb_m[(perturb_m <= th_a) | (perturb_m > th_b)] = 0
    perturb_m[(perturb_m > th_a) & (perturb_m <= th_b)] = 1
    perturb_num = np.count_nonzero(perturb_m, axis=(1, 2))

    pred_m = copy.deepcopy(pred)
    pred_m[(pred_m <= th_a) | (pred_m > th_b)] = 0
    pred_m[(pred_m > th_a) & (pred_m <= th_b)] = 1
    pred_num = np.count_nonzero(pred_m, axis=(1, 2))

    temp = (perturb_num - pred_num) / pred_num
    return (perturb_num - pred_num) / pred_num

#

def dtb(pred, th_a, th_b):
    pred_m = copy.deepcopy(pred)
    pred_m[(pred_m <= th_a) | (pred_m > th_b)] = 0
    pred_m[(pred_m > th_a) & (pred_m <= th_b)] = 1
    pred_num = np.count_nonzero(pred_m)
    return pred_num

# 获取输入geoheight概率密度图
def getInputHistogram(geoheight, index):
    for L in range(0, 4):
        if L < 3:
            geo = geoheight[:, :, L].flatten()
        else:
            geo = np.mean(geoheight, axis=2).flatten()
        g = sns.distplot(geo,
                         hist=False,
                         kde=True,  # 开启核密度曲线kernel density estimate (KDE)
                         kde_kws={'linestyle': '-', 'linewidth': '1', 'color': '#c72e29',
                                  # 设置外框线属性
                                  },
                         # fit= norm,
                         color='#098154',
                         axlabel='data_{}-geo level{}'.format(index, L),  # 设置x轴标题
                         )
        d = g.get_lines()[0].get_data()
        # print(d)
        plt.show()

# 获取salience map概率密度图
def getSalienceHistogram(salience, index):
    for L in range(0, 4):
        s = salience[L, :, :].flatten()
        g = sns.distplot(s,
                         hist=False,
                         kde=True,  # 开启核密度曲线kernel density estimate (KDE)
                         kde_kws={'linestyle': '-', 'linewidth': '1', 'color': '#c72e29',
                                  # 设置外框线属性
                                  },
                         # fit= norm,
                         color='BLUE',
                         axlabel='data_{}-salience{}'.format(index, L),  # 设置x轴标题
                         )
        plt.show()

# 计算两图互信息-->比较两图相似性
def mutual_similarity(img1, img2):
    return skm.mutual_info_score(img1, img2)


# 计算两图的JS散度-->比较两图相似性
# JS计算前要将数据归一化为数据和相加为1，并且均为正数
def JS_similarity(p, q):
    p_minmaxscale = np.array(MinMaxScaler(feature_range=(0, 1)).fit_transform(p.reshape(-1, 1)).transpose())[0]
    q_minmaxscale = np.array(MinMaxScaler(feature_range=(0, 1)).fit_transform(q.reshape(-1, 1)).transpose())[0]
    p_scale = p_minmaxscale/np.sum(p_minmaxscale)
    q_scale = q_minmaxscale/np.sum(q_minmaxscale)
    M = (p_scale + q_scale) / 2
    return 0.5 * stats.entropy(p_scale, M, base=2) + 0.5 * stats.entropy(q_scale, M, base=2)

# 计算两个vector相似性
def cosine_similarity(v1, v2):
    return np.dot(v1, v2) / (np.linalg.norm(v1) * np.linalg.norm(v2))

# 预测结果uncertainty
def cal_uncertainty(y_pred, y_true):
    y_pred = y_pred.flatten()
    y_true = y_true.flatten()
    bins = np.arange(int(min(y_pred)-1), int(max(y_pred)+1), 2)
    y_true_bins = np.digitize(y_true, bins)
    y_pred_bins = np.digitize(y_pred, bins)
    # 创建混淆矩阵
    conf_matrix = confusion_matrix(y_true_bins, y_pred_bins)
    accuracy = accuracy_score(y_true_bins, y_pred_bins)
    return conf_matrix, accuracy


# CAM与5*512的相关性分析
def correlation(CAM, pd):
    p = stats.pearsonr(CAM, pd)
    return p


# 不同模型5层salience average value统计直方图
def histogram(metrics, model_name, layer_name):  # metrics存放所有模型(行)的所有层(列)
    shapes = np.shape(np.array(metrics))
    rows = shapes[0]
    cols = shapes[1]
    fig, big_axes = plt.subplots( nrows=rows, ncols=1)
    # for r, big_ax in enumerate(big_axes, start=1):
    #     big_ax.set_title(model_name[r-1])
    for r in range(rows):
        for c in range(cols):
            ax = fig.add_subplot(rows, cols, r*c+c+1)
            ax.set_title(layer_name[c])
            ax.hist(metrics[r][c], bins=10)
    plt.tight_layout()
    plt.show()

# 图像连通性
def connectivity(input, param=4):
    if param == 4:
        labeled_array, num_features = scipy.ndimage.label(input)  # 默认四连通
    elif param == 8:
        structure = [[1,1,1],
                             [1,1,1],
                             [1,1,1]]
        labeled_array, num_features = scipy.ndimage.label(input, structure)
    return labeled_array, num_features

# LAM高关注区域
def LAMarea(LAM, input):
    min = np.min(LAM)
    max = np.max(LAM)
    min_ = min + (max - min)/5
    max_ = max - (max - min)/5
    min_area_mask = np.where((LAM >= min)&(LAM <= min_), 1, 0)
    max_area_mask = np.where((LAM <= max)&(LAM >= max_), 1, 0)
    mask = min_area_mask + max_area_mask
    mask_dim = np.expand_dims(mask,2).repeat(3,axis=2)

    # min_area = np.expand_dims(min_area_mask,2).repeat(3,axis=2) * input
    # max_area = np.expand_dims(max_area_mask,2).repeat(3,axis=2) * input
    # return min_area, max_area

    important_area = mask_dim * input
    return important_area


def evaluate_metrics(predictions, true_values):
    # 将预测值和真实值转换为平面数组
    pred_flat = predictions.flatten()
    true_flat = true_values.flatten()

    # 计算准确率
    accuracy = accuracy_score(true_flat, pred_flat)

    # 计算F1分数
    f1 = f1_score(true_flat, pred_flat, average='weighted')

    # 计算混淆矩阵
    # conf_matrix = confusion_matrix(true_flat, pred_flat, labels=[0, 1, 2, 3, 4])

    return accuracy, f1


# MSE误差计算


# 按降水量排序

def sort_rainfall(datalist, level):
    rainfall_sum = []
    for data in datalist:
        # 将数据转换为 numpy 数组，便于处理
        data = np.array(data)

        # 定义不同level的阈值
        thresholds = {
            "1": [0, 0.5],  # 全部格点
            "2": [0.5, 1],  # 降水量大于5的格点
            "3": [1, 2],  # 降水量大于10的格点
            "4": [2, 5],  # 降水量大于15的格点
            "5": [5, 100],  # 降水量大于20的格点
        }

        if level == "all":
            total_sum = np.sum(data)
            rainfall_sum.append(total_sum)
        elif level in thresholds:
            threshold = thresholds[level]
            filtered_data = data[(data > threshold[0]) & (data < threshold[1])]
            total_sum = np.sum(filtered_data)
            rainfall_sum.append(total_sum)
        else:
            raise ValueError("无效的level值，仅支持 'all' 或 '1'-'5'")

    rainfall_sum = np.array(rainfall_sum)
    sort_indices = np.argsort(rainfall_sum)
    sorted_rainfall_sum = rainfall_sum[sort_indices]

    return sorted_rainfall_sum, sort_indices




def main():
    # geoPath = "../../data/ERA5/geopotential/geopotential.nc"
    # prePath = "../../data/ERA5/totalPrecipitation1979-2000.nc"
    # geonpyPath = "./trainset_geo.npy"
    # prenpyPath = "./trainset_pre.npy"
    # x, y = dataSet.build_dataset(geoPath, prePath, geonpyPath, prenpyPath)
    # num = x.shape[0]
    # y_test = y[int(num * 0.8):, :]

    # 相关数据子集
    M14_L2_ind = [193, 1678, 384, 411, 405, 404]
    psnr_abnormal = [10, 136, 139, 229, 299, 502, 1646, 1670]


    # 加载测试集的ground truth
    # 计算ts评分
    # pred_list = np.load("./2000-2009_heavyrain_pred1800.npy")
    # pred_mse_list = np.load("./2000-2009_heavyRain_pred_mse.npy")
    # pred_list = pred_list.reshape((16, 1800, 80, 120, 1))

    gt_list = np.load("./2000-2009_heavyrain_0829.npz")["precipitation"][:1800]

    threshold = [0.5, 1, 2, 5]



    # 统计降水量大于20的格点个数，取前10%
    '''
    heavy_grid_num_list = []
    for data in gt_list:
        heavy_grid = np.where(data >= 5, 1, 0)
        heavy_grid_num = np.sum(heavy_grid == 1)
        heavy_grid_num_list.append(-1 * heavy_grid_num)
    heavy_index = np.argsort(heavy_grid_num_list)[:180]
    '''

    # 降水量排序
    sorted_rainfall_sum, sort_indices = sort_rainfall(gt_list, "5")
    print(sorted_rainfall_sum[-20:])
    print("===================")
    # print(sort_indices[-20:])
    print(list(sort_indices).index(193))

    # 每个模型accuracy对比
    pred_path = "./pred/2000-2009heavyRain/"
    pred_files = sorted(os.listdir(pred_path))
    selectID = 1441
    groundtruth =  gt_list[selectID]
    scores = []
    for file in pred_files:
        prediction = np.load(pred_path + file)[selectID]
        mse = mean_squared_error(prediction[:,:,0], groundtruth)
        scores.append(mse)
    print(scores)
    print("===================")


    # 获取原数据集子集
    # sub_gt_list = gt_list[heavy_index]
    # sub_pred_list = pred_list[:, heavy_index, :, :, :]

    ts_array = []
    section = [[0, 0.5], [0.5, 1], [1, 2], [2, 5], [5, 500]]
    tsJSON = {}
    for i in range(16):
        ts_level_array = []
        # ts_ = ts(pred_list[i, :, :, :, 0], gt_list[:, :, :], threshold)
        # ts_ = ts(sub_pred_list[i, :, :, :, 0], sub_gt_list[:, :, :], threshold, "M{}".format(i))
        # ts_level_ = ts_level(sub_pred_list[i, :, :, :, 0], sub_gt_list[:, :, :], 2, 5, "M{}".format(i))
        # ts_ = ts(sub_pred_list[i, :, :, 0], sub_gt_list[:, :], threshold, "M{}".format(i))  # 只取1个数

        for s in section:
            mean_list = []
            for j in range(0, 1800):
                # ts_level_ = ts_level(pred_list[i, j, :, :, 0], gt_list[j, :, :], 0, 0.5, "M{}".format(i))  # 只取1个数
                # ts_level_ = ts_level(pred_list[i, j, :, :, 0], gt_list[j, :, :], 0.5, 1, "M{}".format(i))  # 只取1个数
                # ts_level_ = ts_level(pred_list[i, j, :, :, 0], gt_list[j, :, :], 1, 2, "M{}".format(i))  # 只取1个数
                ts_level_ = ts_level(pred_list[i, j, :, :, 0], gt_list[j, :, :], s[0], s[1], "M{}".format(i))  # 只取1个数
                # ts_level_ = ts_level(pred_mse_list[j, :, :, 0], gt_list[j, :, :], s[0], s[1], "M{}".format(i))  # 只取1个数
                mean_list.append(ts_level_[2][0])
            # ts_array.append(ts_)
            ts_level_array.append(np.mean(mean_list))
        tsJSON["model_{}".format(i)] = ts_level_array
        ts_array.append(ts_level_array)
    print(ts_array)

    tsJSONDATA = json.dumps(tsJSON, indent=4, separators=(',', ': '))
    jsonPATH = "./casedata/"
    if not os.path.exists(jsonPATH):
        os.makedirs(jsonPATH)
    # f = open(jsonPATH + 'tsne_M10train+14model.json', 'w')
    jsonname = 'ts_score.json'
    f = open(jsonPATH + jsonname, 'w')
    f.write(tsJSONDATA)
    f.close()
    print(jsonname + " has been saved.")







    lambda_miu = ["00", "02", "20", "22"]
    # index = [2, 13, 34, 47]
    # pred_root = "./pred/"
    # pred_list = os.listdir(pred_root)
    # for p in range(8,12):
    #     y_pred = np.load(pred_root + pred_list[p])
    #     for i in index:
    #         ts_array = ts(y_pred[i, :, :, 0], y_test[i, :, :, 0], threshold)
    #         print("{} - test {}: {}".format(lambda_miu[p-8], i, ts_array))


    '''
    # CAM与5*512的相关性分析
    pd_layer = np.load("./2000-2009_heavyRain_pdLayer100_part0_0925.npy")
    for i in range(0, 5):
        print("data {}:".format(i))
        CAM = np.load("./visualization/CAM/data{}_0.5_00.npy".format(i))
        for level in range(0, 5):
            pd_layer_data1 = abs(pd_layer[0, :, level, i])
            # pd_layer_data1_mean = np.mean(pd_layer_data1, axis=1)
            CAM_mean = np.sum(abs(CAM), axis=(1, 2))
            CAM_std = np.std(CAM, axis=(1, 2))
            p = correlation(CAM_std, pd_layer_data1)
            print("level {} -- {}".format(level, p))
    '''


    threshold = [0.5, 1, 2, 5]
    lambda_miu = ["00", "02", "20", "22"]
    '''
    for alpha in threshold:
        for lm in lambda_miu:
            name = "group3_{}_{}_perturb.npy".format(alpha, lm)
            perturb_list = np.load("./visualization/latent_perturb/groups/"+name)
            pred_list = np.load("./2000-2009_heavyrain_pred.npy")[1238:1246]  # 换group记得换下标
            gt_list = np.load("./2000-2009_heavyrain_0829.npz")["precipitation"][1238:1246]
            index = np.load("./2000-2009_heavyrain_0829.npz")["index"]

            pd_list = []

            # print(np.argwhere(index == 23), np.argwhere(index == 34)) # 13, 22 --group1
            # print(np.argwhere(index == 1374), np.argwhere(index == 1383)) # 271, 280 --group2
            # print(np.argwhere(index == 5108), np.argwhere(index == 5115)) # 1238, 1245 --group3
            # print(np.argwhere(index == 7218), np.argwhere(index == 7225)) # 1749, 1756



            for i in range(0, len(pred_list)):
                # plt.figure()
                # plt.subplot(1, 3, 1)
                # plt.imshow(gt_list[i], cmap="plasma", vmin=0, vmax=30)
                # plt.subplot(1, 3, 2)
                # plt.imshow(pred_list[i][0, :, :, 0], cmap="plasma", vmin=0, vmax=30)
                # plt.subplot(1, 3, 3)
                # plt.imshow(perturb_list[i][0][0, :, :, 0], cmap="plasma", vmin=0, vmax=30)
                # plt.show()

                pd_value_05, pd_value_1, pd_value_2, pd_value_5, pd_value_300 = [], [], [], [], []
                for j in range(0, len(perturb_list[i])):
                    pd_value_05.append(pd_grid_v2(perturb_list[i][j][0,:,:,0], pred_list[i][0,:,:,0], gt_list[i], 0.0, 0.5))
                    pd_value_1.append(pd_grid_v2(perturb_list[i][j][0,:,:,0], pred_list[i][0,:,:,0], gt_list[i], 0.5, 1.0))
                    pd_value_2.append(pd_grid_v2(perturb_list[i][j][0,:,:,0], pred_list[i][0,:,:,0], gt_list[i], 1.0, 2.0))
                    pd_value_5.append(pd_grid_v2(perturb_list[i][j][0,:,:,0], pred_list[i][0,:,:,0], gt_list[i], 2, 5))
                    pd_value_300.append(pd_grid_v2(perturb_list[i][j][0,:,:,0], pred_list[i][0,:,:,0], gt_list[i], 5, 300))
                # pd_value_total = np.abs(pd_value_05) + np.abs(pd_value_1) + np.abs(pd_value_2) + np.abs(pd_value_5) + np.abs(pd_value_300)
                pd_value_total = np.abs(np.array(pd_value_05)) + np.abs(np.array(pd_value_1)) + np.abs(np.array(pd_value_2)) + np.abs(np.array(pd_value_5)) + np.abs(np.array(pd_value_300))
                # 注：array才可以对应元素相加，list的加法是两个列表相连
                pd_list.append([pd_value_05, pd_value_1, pd_value_2, pd_value_5, pd_value_300, pd_value_total])
            np.save("./visualization/latent_perturb/group_pd_v2/group3_{}_{}_pdPerGrid.npy".format(alpha, lm), pd_list)
            print("SAVE: group3_{}_{}_pdPerGrid.npy".format(alpha, lm))
    '''

#     生成训练集
    EPOCH = 10
    BATCH_SIZE = 40

    pred_list = np.load("./1979-2000_heavyRain_pred0919.npy")  # 换group记得换下标
    pred_list = pred_list.reshape((16, -1, 80, 120, 1))
    gt_list = np.load("./2000-2009_heavyrain_0829.npz")["precipitation"][0:1800]
    gt_dtb = []
    for i, j in enumerate(gt_list):
        a= dtb(j, 0.0, 0.5)
        b= dtb(j, 0.5, 1.0)
        c= dtb(j, 1.0, 2.0)
        d= dtb(j, 2.0, 5.0)
        e= dtb(j, 5.0, 300)
        gt_dtb.append([a,b,c,d,e])
    # tsne = TSNE(n_components=2, verbose=1, init="pca", perplexity=30, n_iter=3000, random_state=2)
    # result = tsne.fit_transform(gt_dtb)
    # label = range(0, 1800)
    # for i in range(0, len(result)):
    #     plt.scatter(result[i, 0], result[i, 1], s=6, c=label[i], cmap='cool', vmin=0,
    #                 vmax=1800)
    # plt.show()
    level5_dtb = np.array(gt_dtb)[:, 4]
    ind = np.argpartition(level5_dtb, -100)[-100:]  # 展平grid_pd，获取前20最大值索引
    ind_ = np.unravel_index(ind, level5_dtb.shape)  # 获取原维度下最大值的索引
    print(level5_dtb[ind_])  # 获取最大值




    pd_list_all = []
    for p in range(0, 4):
        name = "1979-2000_heavyRain_perturb_part{}_0919.npy".format(p)
        perturb_list = np.load("./" + name)
        perturb_newshape = np.transpose(perturb_list, (0, 1, 3, 2, 4, 5, 6))
        perturb_newshape = perturb_newshape.reshape((16, -1, 25, 80, 120, 1))
        # perturb_list = []
        # perturb_list = perturb_list.reshape((2, 25, -1, 80, 120, 1))
        # perturb_test = np.load("./2000-2009_heavyRain_perturb_test_1.npy")
        # perturb_test = perturb_test.reshape((2, 25, -1, 80, 120, 1))
        # perturb_list_250 = np.load("./2000-2009_heavyRain_perturb.npy")
        # perturb_list_250 = perturb_list_250.reshape((16, -1, 25, 80, 120, 1))
        # index = np.load("./2000-2009_heavyrain_0829.npz")["index"]
        # pd_list = np.load("./2000-2009_heavyRain_pdPerGrid1800.npy")

        # for i in range(0, 10):
        #     fig = plt.figure()
        #     ax = fig.add_subplot(2, 1, 1)
        #     im = ax.imshow(pred_list[4, i * 40 + 2, :, :, 0], cmap="coolwarm", vmin=0, vmax=20)
        #     ax = fig.add_subplot(2, 1, 2)
        #     im = ax.imshow(perturb_newshape[4, i * 40 + 2, i, :, :, 0], cmap="coolwarm", vmin=0, vmax=20)
        #     plt.show()
        #
        # for i in range(0, 10):
        #     fig = plt.figure()
        #     ax = fig.add_subplot(2, 1, 1)
        #     im = ax.imshow(perturb_list[1, i, i, 2, :, :, 0], cmap="coolwarm", vmin=0, vmax=20)
        #     ax = fig.add_subplot(2, 1, 2)
        #     im = ax.imshow(perturb_newshape[1, i*40+2, i, :, :, 0], cmap="coolwarm", vmin=0, vmax=20)
        #     plt.show()

        pd_list = []
        for i in range(0, 16):  # model个数
            for d in range(0, EPOCH * BATCH_SIZE):
                pd_value_05, pd_value_1, pd_value_2, pd_value_5, pd_value_300 = [], [], [], [], []
                for j in range(0, 25):
                    pd_value_05.append(pd_grid_v2(perturb_newshape[i][d][j,:,:,0], pred_list[i][p*EPOCH*BATCH_SIZE + d][:,:,0], 0.0, 0.5))
                    pd_value_1.append(pd_grid_v2(perturb_newshape[i][d][j,:,:,0], pred_list[i][p*EPOCH*BATCH_SIZE + d][:,:,0], 0.5, 1.0))
                    pd_value_2.append(pd_grid_v2(perturb_newshape[i][d][j,:,:,0], pred_list[i][p*EPOCH*BATCH_SIZE + d][:,:,0], 1.0, 2.0))
                    pd_value_5.append(pd_grid_v2(perturb_newshape[i][d][j,:,:,0], pred_list[i][p*EPOCH*BATCH_SIZE + d][:,:,0], 2, 5))
                    pd_value_300.append(pd_grid_v2(perturb_newshape[i][d][j,:,:,0], pred_list[i][p*EPOCH*BATCH_SIZE + d][:,:,0], 5, 300))
                pd_value_total = np.abs(np.array(pd_value_05)) + np.abs(np.array(pd_value_1)) + np.abs(np.array(pd_value_2)) + np.abs(np.array(pd_value_5)) + np.abs(np.array(pd_value_300))
                # 注：array才可以对应元素相加，list的加法是两个列表相连
                pd_list.append([pd_value_05, pd_value_1, pd_value_2, pd_value_5, pd_value_300, pd_value_total])


        pd_list_all.append(pd_list)
    # np.save("./1979-2000_heavyRain_pdPerGrid_0919.npy", pd_list_all)
    print("SAVE: pdPerGrid.npy")


if __name__ == '__main__':
    main()

    # pd_list = np.load("./1979-2000_heavyRain_pdPerGrid_0919.npy")
    # pd_list = pd_list.reshape((4, 16, -1, 6, 25))
    # pd_list_new = np.transpose(pd_list, (1, 0, 2, 3, 4)).reshape((16, 400 * 4, 6, 25))
    # np.save("./1979-2000_heavyRain_pdPerGrid_reshape_0919.npy", pd_list_new)
    # pd_list_new = np.load("./1979-2000_heavyRain_pdPerGrid_reshape_0919.npy")
    #
    # for i in range(0, 4):
    #     fig = plt.figure()
    #     ax = fig.add_subplot(2, 2, 1)
    #     im = ax.imshow(pd_list_new[0, 25 + i*10, :, :], cmap="coolwarm", vmin=-1, vmax=1)
    #     ax = fig.add_subplot(2, 2, 2)
    #     im = ax.imshow(pd_list_new[1, 25 + i*10, :, :], cmap="coolwarm", vmin=-1, vmax=1)
    #     ax = fig.add_subplot(2, 2, 3)
    #     im = ax.imshow(pd_list_new[2, 25 + i*10, :, :], cmap="coolwarm", vmin=-1, vmax=1)
    #     ax = fig.add_subplot(2, 2, 4)
    #     im = ax.imshow(pd_list_new[3, 25 + i*10, :, :], cmap="coolwarm", vmin=-1, vmax=1)
    #     plt.show()
    # print()
