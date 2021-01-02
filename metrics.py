from sklearn import metrics
labels_true = [0, 0, 0, 1, 1, 1]
labels_pred = [0, 0, 1, 1, 2, 2]

print('调整兰德指数ARI=',metrics.adjusted_rand_score(labels_true, labels_pred))
print('调整互信息AMI=',metrics.adjusted_mutual_info_score(labels_true, labels_pred))

print('同质性homogeneity=',metrics.homogeneity_score(labels_true, labels_pred))
#每个群集只包含单个类的成员。

print('完整性completeness=',metrics.completeness_score(labels_true, labels_pred))
#给定类的所有成员都分配给同一个群集。

print('调和平均V-measure=',metrics.v_measure_score(labels_true, labels_pred))
#两者的调和平均V-measure

#内部指标：
import numpy as np
from sklearn.cluster import KMeans
from sklearn import metrics
from sklearn.metrics import pairwise_distances
from sklearn import datasets

X, y = datasets.load_iris(return_X_y=True)
#print('X=',X)
#print('y=',y)
#轮廓系数silhouette coefficient
kmeans_model = KMeans(n_clusters=3, random_state=1).fit(X)
labels = kmeans_model.labels_
print('silhouette coefficient=',metrics.silhouette_score(X, labels, metric='euclidean'))

#Calinski-Harabasz Index
kmeans_model = KMeans(n_clusters=3, random_state=1).fit(X)
print('Calinski-Harabasz Index=',metrics.calinski_harabasz_score(X, labels))
#数值越小可以理解为：组间协方差很小，组与组之间界限不明显。与轮廓系数的对比，最大的优势：快，毫秒级
