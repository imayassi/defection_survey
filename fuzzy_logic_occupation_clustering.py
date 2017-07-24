from numpy import genfromtxt
import pandas as pd
import numpy as np
import pyodbc
import pandas as pd

conn = pyodbc.connect(dsn='VerticaProd')
# df = pd.read_csv(, engine='python', header=0, delim_whitespace=False, squeeze=True)
# df = pd.read_csv("occupation_taxpayer.csv", sep='\s+',  header=0, engine='python')
query ="select occupations from (select lower(occupation_taxpayer) as occupations, count(*)  from CTG_ANALYTICS_WS.SM_RETENTION_SOT where tax_year=2015 and lower(occupation_taxpayer) not like'%xx%' group by 1 order by 2 desc )a order by random() limit 1000"
df = pd.read_sql(query, conn,coerce_float=False)

s = df['occupations'].str.split(' ').apply(pd.Series, 1).stack()
# s2=pd.DataFrame(s)
# s2 = df['occupations'].str.split(',').apply(pd.Series, 1).stack()
s.index = s.index.droplevel(-1)
s.name='occupation_expanded'
del df['occupations']
df2=df.join(s)
df2.dropna(inplace=True)
# df2.set_index(['auth_id'], inplace=True)
df_values=df2.values
flattened  = [val for sublist in df_values for val in sublist]
print flattened
import numpy as np
import scipy.linalg as lin
import Levenshtein as leven
import matplotlib.pyplot as plt
from sklearn.cluster import KMeans
from sklearn.cluster import AffinityPropagation
import itertools

words=np.array(flattened)
print words[1]


print "calculating distances..."

(dim,) = words.shape

f = lambda (x,y): leven.distance(x,y)
res=np.fromiter(itertools.imap(f, itertools.product(words, words)),
                dtype=np.uint8)
A = np.reshape(res,(dim,dim))

print "svd..."

u,s,v = lin.svd(A, full_matrices=False)

# print u.shape
# print s.shape
# print s
# print v.shape

data = u[:,0:]
k_model=KMeans(n_clusters=10)
k_model.fit(data)
centroids = k_model.cluster_centers_
labels = k_model.labels_
print labels

for i in range(np.max(labels)):
    print words[labels==i]

def dist(x,y):
    return np.sqrt(np.sum((x-y)**2, axis=1))

print "centroid points.."
for i,c in enumerate(centroids):
    idx = np.argmin(dist(c,data[labels==i]))
    print words[labels==i][idx]
    print words[labels==i]

plt.plot(centroids[:, 0], centroids[:, 1], 'x')
plt.hold(True)
plt.plot(u[:, 0], u[:, 1], '.')
plt.show()

from mpl_toolkits.mplot3d import Axes3D

fig = plt.figure()
ax = Axes3D(fig)
ax.plot(u[:, 0], u[:, 1], u[:, 2], '.', zs=0,
        zdir='z', label='zs=0, zdir=z')
plt.show()
