import numpy as np
from sklearn.cluster import KMeans
import pandas as pd

#################################
# DATA: On retire la colonne "school", "reason" and maybe "traveltime" 
#################################
data_df = pd.read_csv("student-mat.csv")

ids=data_df['Id'].values;print(ids)
data_df=data_df.drop('Id',axis=1)
data_df=data_df.drop('school',axis=1)
data_df=data_df.drop('reason', axis=1)
data_df = data_df.replace(
    ['F','M','U', 'R','LE3','GT3','no','yes','A','T','teacher','health','services','at_home','other','mother','father'],
    [0,1,0,1,0,1,0,1,0,1,1,2,3,4,0,1,2])
#print(data_df.head(5))
data=data_df.values
#################################
# PREPARE DATA: "STANDARDIZATION"
#################################
from sklearn.preprocessing import StandardScaler
scaler = StandardScaler(copy=True, with_mean=True, with_std=True)
# Apprentissage
data=scaler.fit_transform(data) #Apprend moyenne et ecart-type pour normalisation par [transform]

#################################
# CLUSTERING
#################################
kmeans = KMeans(n_clusters=5, n_init=10)
kmeans.fit(data)
print("Identifiants: ", ids)
print("Categories:", kmeans.labels_)

for i,c in zip(ids,kmeans.labels_):
    print("id",i," -> categorie: ",c)

#print(kmeans.predict(np.array([1.14,297.90]).reshape(1, -1)))
#################################
# AFFICHAGE
#################################
from matplotlib import pyplot as plt

fig, ax = plt.subplots()
ax.plot(data[:, 0], data[:, 1], 'k.', markersize=10)
for i, txt in enumerate(ids):
    ax.annotate(txt, (data[i, 0]+0.1,data[i, 1]+0.1))
plt.scatter(kmeans.cluster_centers_[:, 0], kmeans.cluster_centers_[:, 1],marker='x', s=169, linewidths=3,color='r', zorder=10)
# Plot separations
aa, bb, cc, dd, ee, ff, gg, hh, jj, kk, ll, mm, nn, oo, pp, qq, rr, ss, tt, uu, vv, ww, xx, yy, zz, aaa, bbb, ccc, ddd, eee = 
    np.meshgrid(np.arange(data[:, 0].min() - 1, data[:, 0].max() + 1, .02), np.arange(data[:, 1].min() - 1, data[:, 1].max() + 1, .02),np.arange(data[:, 2].min() - 1, data[:, 2].max() + 1, .02))
Z = kmeans.predict(np.c_[xx.ravel(), yy.ravel(), zz.ravel()])
Z = Z.reshape(xx.shape)
print(np.min(Z),np.max(Z))
plt.imshow(Z, interpolation='nearest',extent=(xx.min(), xx.max(), yy.min(), yy.max()),cmap=plt.cm.Paired,aspect='auto', origin='lower')
plt.xlabel("AxisRatio (standardized)")
plt.ylabel("Perimeter (standardized)")
plt.show()
