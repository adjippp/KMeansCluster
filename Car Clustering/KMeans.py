# K-Means Clustering

# Importing the libraries
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from sklearn.preprocessing import MinMaxScaler
# Importing the dataset
dataset = pd.read_csv('cars.csv')

X = dataset.iloc[:,:-1].values

X = pd.DataFrame(X)
#konversi semua data menjadi numeric
X = X.convert_objects(convert_numeric=True)
X.columns = ['mpg', 'cylinders', 'cubicinches', 'hp', 'weightlbs', 'timeto60', 'year']


# Eliminating null values
for i in X.columns:
    X[i] = X[i].fillna(int(X[i].mean()))
for i in X.columns:
    print(X[i].isnull().sum())
    

# gunakan elbow method untuk menemukan nilai k (cluster) terbaik
from sklearn.cluster import KMeans
wcss = []
for i in range(1,11):
    kmeans = KMeans(n_clusters=i,init='k-means++',max_iter=300,n_init=10,random_state=0)
    kmeans.fit(X)
    wcss.append(kmeans.inertia_)
plt.plot(range(1,11),wcss)
plt.title('The Elbow Method')
plt.xlabel('Number of clusters')
plt.ylabel('WCSS')
plt.show()

# Applying k-means to the cars dataset
kmeans = KMeans(n_clusters=3,init='k-means++',max_iter=300,n_init=10,random_state=0) 
y_kmeans = kmeans.fit_predict(X)

X = X.values

# Visualising the clusters
plt.figure()
plt.subplot(121)
plt.scatter(X[y_kmeans == 0, 0], X[y_kmeans == 0,1],c='red',label='US')
plt.scatter(X[y_kmeans == 1, 0], X[y_kmeans == 1,1],c='blue',label='Japan')
plt.scatter(X[y_kmeans == 2, 0], X[y_kmeans == 2,1],c='green',label='Europe')
plt.scatter(kmeans.cluster_centers_[:,0],kmeans.cluster_centers_[:,1],color='purple',marker='*',label='centroid')
plt.title('Clusters of car brands')
plt.legend()

#Normalisasi data

dataset = dataset.iloc[:,:-1]
dataset = dataset.convert_objects(convert_numeric=True)
dataset.columns = ['mpg', 'cylinders', 'cubicinches', 'hp', 'weightlbs', 'timeto60', 'year']
for i in dataset.columns:
    dataset[i] = dataset[i].fillna(int(dataset[i].mean()))
for i in dataset.columns:
    print(dataset[i].isnull().sum())
scaler = MinMaxScaler()
scaler.fit(dataset[['mpg']])
dataset.mpg=scaler.transform(dataset[['mpg']])
scaler.fit(dataset[['cylinders']])
dataset.cylinders=scaler.transform(dataset[['cylinders']])
scaler.fit(dataset[['cubicinches']])
dataset.cubicinches=scaler.transform(dataset[['cubicinches']])
scaler.fit(dataset[['hp']])
dataset.hp=scaler.transform(dataset[['hp']])
scaler.fit(dataset[['weightlbs']])
dataset.weightlbs=scaler.transform(dataset[['weightlbs']])
scaler.fit(dataset[['timeto60']])
dataset.timeto60=scaler.transform(dataset[['timeto60']])
scaler.fit(dataset[['year']])
dataset.year=scaler.transform(dataset[['year']])
y_pred=kmeans.fit_predict(dataset[['mpg','cylinders','cubicinches','hp','weightlbs','timeto60','year']])
dataset=dataset.values

plt.subplot(122)
plt.scatter(dataset[y_pred == 0, 0], dataset[y_pred == 0,1],c='red',label='US')
plt.scatter(dataset[y_pred == 1, 0], dataset[y_pred == 1,1],c='blue',label='Japan')
plt.scatter(dataset[y_pred == 2, 0], dataset[y_pred == 2,1],c='green',label='Europe')
plt.scatter(kmeans.cluster_centers_[:,0],kmeans.cluster_centers_[:,1],color='purple',marker='*',label='centroid')
plt.title('Clusters of car brands')
plt.legend()

plt.show()
