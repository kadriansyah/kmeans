import pandas as pd
import numpy as np
from sklearn.cluster import KMeans

# import matplotlib.pyplot as plt
# from matplotlib import style
# style.use("ggplot")

df = pd.read_csv('data_1024.csv',sep='\t')

f1 = df['Distance_Feature'].values
f2 = df['Speeding_Feature'].values

X = np.matrix(list(zip(f1,f2))) # http://stackoverflow.com/questions/40282290/float-argument-must-be-a-string-or-a-number-not-zip
kmeans = KMeans(n_clusters=2).fit(X)

centroids = kmeans.cluster_centers_
labels = kmeans.labels_

print(centroids)
print(labels)

# colors = ["g.","r."]
#
# for i in range(len(X)):
#     # print("coordinate:", X.item((i, 0)), "label:", labels[i])
#     plt.plot(X.item((i, 0)), X.item((i, 1)), colors[labels[i]], markersize = 10)
#
# plt.scatter(centroids[:, 0],centroids[:, 1], marker = "x", s=150, linewidths = 5, zorder = 10)
#
# plt.show()

indices = np.random.permutation(len(X))
predictions = kmeans.predict(X[indices[-10:]])
