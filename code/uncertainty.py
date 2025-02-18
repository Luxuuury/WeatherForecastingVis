# 根据历史相似性数据计算uncertainty
# 设计校验方法
import os.path
import sys
sys.path.append(os.path.join(os.path.dirname(__file__), "subdirectory"))
import numpy as np
import json
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import mean_squared_error
from evaluate import cosine_similarity, JS_similarity
from json_utils.modelCompare import calculate_threat_score, classify_precipitation

# 定义等级划分函数
def classify_precipitation(arr):
    classification = np.zeros_like(arr)
    classification[arr > 5] = 5  # level 4
    classification[(arr > 2) & (arr <= 5)] = 4  # level 3
    classification[(arr > 1) & (arr <= 2)] = 3  # level 2
    classification[(arr > 0.5) & (arr <= 1)] = 2  # level 1
    classification[(arr > 0) & (arr <= 0.5)] = 1  # level 0
    return classification

# 贝叶斯方法计算每个格点的Uncertainty
def estimate_uncertainty_bayesian(model_preds, ground_truths, model_pred_selected, ground_truth_selected, similar_indices):
    #
    errors = model_preds - ground_truths
    # 计算全局误差的均值和方差（每个格点）
    mu_global = np.mean(errors, axis=0)  # 形状：(80, 120)
    sigma2_global = np.var(errors, axis=0, ddof=1)

    epsilon = 1e-6
    sigma2_global += epsilon

    # 获取相似天的误差
    errors_similar = errors[similar_indices]
    # 计算相似天误差的均值和方差（每个格点）
    mu_similar = np.mean(errors_similar, axis=0)  # 形状：(80, 120)
    sigma2_similar = np.var(errors_similar, axis=0, ddof=1)  # 形状：(80, 120)
    sigma2_similar += epsilon  # 防止除以零

    # 先验（全局误差统计量）
    mu_prior = mu_global
    sigma2_prior = sigma2_global

    # 似然（来自相似天的数据）
    mu_likelihood = mu_similar
    sigma2_likelihood = sigma2_similar / len(similar_indices)  # 样本均值的方差
    sigma2_likelihood += epsilon  # 防止除以零

    # 计算后验方差
    sigma2_posterior = 1 / (1 / sigma2_prior + 1 / sigma2_likelihood)
    # 计算后验均值
    mu_posterior = sigma2_posterior * (mu_prior / sigma2_prior + mu_likelihood / sigma2_likelihood)
    posterior_std = np.sqrt(sigma2_posterior)  # 后验标准差

    # 使用后验均值校正selected day的预测值
    corrected_prediction = model_pred_selected - mu_posterior

    return corrected_prediction




if __name__ == "__main__":
    geoheight = np.load("./2000-2009_heavyrain_0829.npz")["geo"][0:1800]
    geoheight = geoheight.reshape((1800, -1))
    groundtruth = np.load("./2000-2009_heavyrain_0829.npz")["precipitation"][0:1800]
    # groundtruth = groundtruth.reshape((1800, -1))
    geo_latentvec = np.load("2000-2009_heavyRain_latentvec_mseModel.npy")
    geo_latentvec = geo_latentvec.reshape((1800, -1))
    prediction = np.load("./pred/2000-2009heavyRain/unet_5_unet_comb3_mse_b5_22_10levelsb2.npy")[0: 1800]
    selectedID = 1441
    selected_vector = geo_latentvec[selectedID]
    # k = 10

    # the 10 most similar latent vectors
    similarities = []
    for j, vector in enumerate(geo_latentvec):
        sim = cosine_similarity(selected_vector, vector)
        similarities.append((j, sim))
    similarities.sort(key=lambda x: x[1], reverse=True)
    # top_k_indices = [idx for idx, sim in similarities[: k + 1]]
    top_k_indices = []
    # num = 0
    # while(len(top_k_indices) < k):
    #     if(similarities[num][0] < 1441):
    #         top_k_indices.append(similarities[num][0])
    #     num = num + 1
    # print(top_k_indices)

    # the 10 most similar geo_height image
    '''
    geo_array = np.reshape(geoheight, (len(geoheight), -1))
    geo = geo_array[1000]
    similarity = []
    for j, geo_array in enumerate(geoheight):
        # similarity.append(JS_similarity(geo, geo_array[j]))
        simi = cosine_similarity(geo, geo_array)
        similarity.append((j, simi))
    similarity.sort(key=lambda x: x[1], reverse=True)
    argsort_simi = [idx for idx, sim in similarity[: k + 1]]
    '''

    # calculate uncertainty -- 按等级计算，每个等级只有一个uncertainty
    '''
    # predict the similar data
    history_prediction = []
    history_gt = []
    ### latent vector
    # for i in top_k_indices[1:]:
    #     history_data.append(np.array(prediction[i]).reshape((80, 120)))
    # geo height
    for i in top_k_indices[1:]:
        history_prediction.append(np.array(prediction[i]).reshape((80, 120)))
        history_gt.append(np.array(groundtruth[i]).reshape((80, 120)))

    classified_prediction = np.array([classify_precipitation(d) for d in history_prediction])
    classified_gt = np.array([classify_precipitation(d) for d in history_gt])

    # 初始化一个空字典来存储每个降雨等级的误差和不确定性
    error_by_level = {}
    uncertainty_by_level = {}

    for level in [10, 1, 2, 3, 4]:
        true_mask = (classified_gt == level)
        error_matrix = np.where(classified_prediction != classified_gt, 1, 0)
        total_points = np.sum(true_mask)
        if total_points > 0:
            level_error = np.sum(error_matrix[true_mask]) / total_points  # 计算平均误差
            uncertainty = np.std(error_matrix[true_mask])  # 计算不确定性（标准差）
        else:
            level_error = 0
            uncertainty = 0

        # 存储该等级的误差和不确定性
        error_by_level[level] = level_error
        uncertainty_by_level[level] = uncertainty

    # 在当前预测数据上绘制uncertainty
    current = np.array(prediction[1000]).reshape((80, 120))
    classified_current = classify_precipitation(current)
    error_map = np.zeros_like(classified_current, dtype=float)
    for level, error_value in error_by_level.items():
        error_map[classified_current == level] = error_value
    # plt.figure(figsize=(10, 8))
    fig, axes = plt.subplots(1, 2, figsize=(20, 7))
    sns.heatmap(error_map, cmap="YlOrRd", annot=False, cbar=True, ax=axes[0])  # 颜色映射可以根据需要调整
    axes[0].set_title("Heatmap of Prediction Errors by Precipitation Level")

    sns.heatmap(current, cmap="viridis", annot=False, cbar=True, ax=axes[1])
    axes[1].set_title("Precipitation Ground Truth")

    plt.tight_layout()
    plt.show()
    '''

    # 根据uncertainty校正
    for k in range(12, 13):
        num = 0
        top_k_indices = []
        while (len(top_k_indices) < k):
            if (similarities[num][0] < 1441):
                top_k_indices.append(similarities[num][0])
            num = num + 1
        print(top_k_indices)
        corrected_prediction = estimate_uncertainty_bayesian(prediction[0: selectedID, :, :, 0], groundtruth[0: selectedID], prediction[selectedID, :, :, 0], groundtruth[selectedID],
                                      top_k_indices[0:])
        # np.save("./json_utils/data/correction.npy", corrected_prediction)

        # 写入json文件
        '''
        correction_json = {
            "id": 1441,
            "correction": []
        }
        for x in range(0, 80):
            for y in range(0, 120):
                correction_json["correction"].append([x, y, float(corrected_prediction[79 - x][y])])

        with open("./json_utils/data/correction.json", 'w') as json_file:
            json.dump(correction_json, json_file, indent=4)
        '''

        '''
        fig = plt.figure()
        ax = fig.add_subplot(2, 3, 1)
        ax.axis('off')
        ax.imshow(prediction[1441, :, :, 0], cmap="plasma", vmin=0, vmax=30)
        ax = fig.add_subplot(2, 3, 2)
        ax.axis('off')
        ax.imshow(corrected_prediction, cmap="plasma", vmin=0, vmax=30)
        ax = fig.add_subplot(2, 3, 3)
        ax.axis('off')
        ax.imshow(groundtruth[1441], cmap="plasma", vmin=0, vmax=30)
        ax = fig.add_subplot(2, 3, 4)
        ax.axis('off')
        fig1 = ax.imshow(groundtruth[1441] - prediction[1441, :, :, 0], cmap="plasma")
        fig.colorbar(fig1, ax=ax)
        ax = fig.add_subplot(2, 3, 5)
        ax.axis('off')
        fig2 = ax.imshow(groundtruth[1441] - corrected_prediction, cmap="plasma")
        fig.colorbar(fig2, ax=ax)
        plt.show()
        '''

        # mse_before_correct = mean_squared_error(prediction[1441, :, :, 0], groundtruth[1441])
        # mse_after_correct = mean_squared_error(corrected_prediction, groundtruth[1441])
        # print(mse_before_correct, mse_after_correct)

        pred_classify = classify_precipitation(prediction[1441, :, :, 0])
        gt_classify = classify_precipitation(groundtruth[1441])
        corrected_classify = classify_precipitation(corrected_prediction)
        ts_before_correct =calculate_threat_score(pred_classify, gt_classify, [1,2,3,4,5])
        ts_after_correct = calculate_threat_score(corrected_classify, gt_classify, [1,2,3,4,5])
        print(ts_before_correct)
        print(ts_after_correct)