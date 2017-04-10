import pandas as pd
import numpy as np
from sklearn.cluster import KMeans

import matplotlib.pyplot as plt
plt.rcParams['figure.figsize'] = (12, 6)

df = pd.read_csv('data_1024.csv',sep='\t')

# plt.figure()
# plt.plot(df.Distance_Feature,df.Speeding_Feature,'ko')
# plt.ylabel('Speeding Feature')
# plt.xlabel('Distance Feature')
# plt.ylim(0,100)
# plt.show()

f1 = df['Distance_Feature'].values
f2 = df['Speeding_Feature'].values

X = np.matrix(list(zip(f1,f2))) # http://stackoverflow.com/questions/40282290/float-argument-must-be-a-string-or-a-number-not-zip
kmeans = KMeans(n_clusters=4).fit(X)

# # elbow point method to define K
# K = range(1, 10)
# km = [KMeans(n_clusters=i) for i in K]
# score = [km[i].fit(X).score(X) for i in range(len(km))]
#
# plt.figure()
# plt.plot(K, score)
# plt.show()

# # Plot the results 2
# plt.figure()
# h1,=plt.plot(f1[kmeans.labels_==0],f2[kmeans.labels_==0],'go')
# plt.plot(np.mean(f1[kmeans.labels_==0]),np.mean(f2[kmeans.labels_==0]),'g*',markersize=20,mew=3)
# # print centroid 1
# print(np.mean(f1[kmeans.labels_==0]),np.mean(f2[kmeans.labels_==0]))
# h2,=plt.plot(f1[kmeans.labels_==1],f2[kmeans.labels_==1],'bo')
# plt.plot(np.mean(f1[kmeans.labels_==1]),np.mean(f2[kmeans.labels_==1]),'b*',markersize=20,mew=3)
# # print centroid 2
# print(np.mean(f1[kmeans.labels_==1]),np.mean(f2[kmeans.labels_==1]))
# plt.ylabel('Speeding Feature')
# plt.xlabel('Distance Feature')
#
# plt.legend([h1,h2],['Group 1','Group 2'], loc='upper left')
# plt.show()

# # Plot the results 4
# plt.figure()
# h1,=plt.plot(f1[kmeans.labels_==0],f2[kmeans.labels_==0],'go')
# plt.plot(np.mean(f1[kmeans.labels_==0]),np.mean(f2[kmeans.labels_==0]),'g*',markersize=20,mew=3)
# h2,=plt.plot(f1[kmeans.labels_==1],f2[kmeans.labels_==1],'bo')
# plt.plot(np.mean(f1[kmeans.labels_==1]),np.mean(f2[kmeans.labels_==1]),'b*',markersize=20,mew=3)
# h3,=plt.plot(f1[kmeans.labels_==2],f2[kmeans.labels_==2],'mo')
# plt.plot(np.mean(f1[kmeans.labels_==2]),np.mean(f2[kmeans.labels_==2]),'m*',markersize=20,mew=3)
# h4,=plt.plot(f1[kmeans.labels_==3],f2[kmeans.labels_==3],'ro')
# plt.plot(np.mean(f1[kmeans.labels_==3]),np.mean(f2[kmeans.labels_==3]),'r*',markersize=20,mew=3)
# plt.ylabel('Speeding Feature')
# plt.xlabel('Distance Feature')
# plt.legend([h1,h2,h3,h4],['Group 1','Group 2','Group 3','Group 4'], loc='upper left')
# plt.show()

# centroids = kmeans.cluster_centers_
# labels = kmeans.labels_

# print(centroids)
# print(labels)
#
# indices = np.random.permutation(len(X))
# predictions = kmeans.predict(X[indices[-10:]])
