# -*- coding:utf-8 -*-
import numpy as np
from matplotlib import pyplot


class K_Means(object):
    # k是分组数；tolerance‘中心点误差’；max_iter是迭代次数
    def __init__(self, k=2, tolerance=100, max_iter=300):
        self.k_ = k
        self.tolerance_ = tolerance
        self.max_iter_ = max_iter

    def fit(self, data, cen_vec, eid_list):
        self.centers_ = {}
        for i in range(self.k_):
            self.centers_[i] = cen_vec[i]

        for i in range(self.max_iter_):
            self.clf_ = {}
            self.clf_id_ = {}
            for j in range(self.k_):
                self.clf_[j] = []
                # print("质点:",self.centers_)
            index = 0
            closest_dis_0, closest_dis_1 = 50, 50
            closest_dis = [closest_dis_0, closest_dis_1]
            closest_dis_eid, closest_dis_index = [0, 0], [0, 0]
            for feature in data:
                eid = eid_list[index]
                # distances = [np.linalg.norm(feature-self.centers[center]) for center in self.centers]
                distances = []
                for center in self.centers_:
                    # 欧拉距离
                    # np.sqrt(np.sum((features-self.centers_[center])**2))
                    distances.append(np.linalg.norm(feature - self.centers_[center]))
                    if distances[center] < closest_dis[center]:
                        closest_dis[center] = distances[center]
                        closest_dis_eid[center] = eid
                        closest_dis_index[center] = index
                classification = distances.index(min(distances))
                self.clf_[classification].append(feature)
                self.clf_id_[eid] = classification
                index += 1
            for center in self.centers_:
                if len(self.clf_[center]) == 0:
                    self.clf_[center].append(data[closest_dis_index[center], :])
                    self.clf_id_[closest_dis_eid[center]] = center
                    del self.clf_[1 - center][closest_dis_index[center]]
                    del self.clf_id_[closest_dis_eid[center]]

            # print("分组情况:",self.clf_)
            prev_centers = dict(self.centers_)
            for c in self.clf_:
                self.centers_[c] = np.average(self.clf_[c], axis=0)

            # '中心点'是否在误差范围
            optimized = True
            for center in self.centers_:
                org_centers = prev_centers[center]
                cur_centers = self.centers_[center]
                print(np.linalg.norm(cur_centers - org_centers, ord=1))
                if np.linalg.norm(cur_centers - org_centers, ord=1) > self.tolerance_:
                    optimized = False
            if optimized:
                return self.clf_id_

    def predict(self, p_data):
        distances = [np.linalg.norm(p_data - self.centers_[center]) for center in self.centers_]
        index = distances.index(min(distances))
        return index


if __name__ == '__main__':
    x = np.array([[1, 2], [1.5, 1.8], [5, 8], [8, 8], [1, 0.6], [9, 11]])
    k_means = K_Means(k=2)
    k_means.fit(x, x[0:2])
    print(k_means.centers_)
    for center in k_means.centers_:
        pyplot.scatter(k_means.centers_[center][0], k_means.centers_[center][1], marker='*', s=150)

    for cat in k_means.clf_:
        for point in k_means.clf_[cat]:
            pyplot.scatter(point[0], point[1], c=('r' if cat == 0 else 'b'))

    predict = [[2, 1], [6, 9]]
    for feature in predict:
        cat = k_means.predict(predict)
        pyplot.scatter(feature[0], feature[1], c=('r' if cat == 0 else 'b'), marker='x')

    pyplot.show()