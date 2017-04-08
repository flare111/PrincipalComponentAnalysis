import sklearn.manifold
import scipy.stats as ss
import pandas as pd
import random
from sklearn import cluster
from sklearn.decomposition import PCA
from sklearn.preprocessing import scale
import numpy as np
import matplotlib.pyplot as plt
import csv
from scipy.spatial.distance import cdist
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler
from sklearn.manifold import Isomap,MDS
from sklearn import cluster as Kcluster, metrics as SK_Metrics


def random_sampling(data_frame, fraction):
    # selection holds the rows we are randomly picking from the DF
    selection = random.sample(data_frame.index, (int)(len(data_frame) * fraction))
    return data_frame.ix[selection]


def stratified_sampling(data_frame, cluster_count, fraction):
    k_means = cluster.KMeans(n_clusters = cluster_count)
    k_means.fit(data_frame)
    data_frame['cluster'] = k_means.labels_  # generates labels on each row
    strat_rows = []
    for i in range(cluster_count):
        strat_rows.append(data_frame.ix[random.sample(data_frame[data_frame['cluster'] == i].index,
                                                      (int)(len(data_frame[data_frame['cluster'] == i]) * fraction))])
    strat_sample = pd.concat(strat_rows)
    del strat_sample['cluster']
    return strat_sample


def getCluster(cluster_count, data_frame):  # returns the optimal cluster number using elbowing
    meandist = []
    clusters = range(1, cluster_count)
    for k in range(1, cluster_count):
        model = KMeans(n_clusters=k)
        model.fit(data_frame)
        meandist.append(sum(np.min(cdist(data_frame, model.cluster_centers_, 'euclidean'), axis=1))
                        / data_frame.shape[0])
    plt.plot(clusters, meandist)
    plt.title('Elbow method for k value')
    plt.show(block = True)


def getIntrinsic(data_frame, components, choice):  # perform on strat sample and random sample
    data = scale(data_frame.values)
    pca = PCA(components)
    pca.fit_transform(data)
    eigen_vals = pca.explained_variance_
    components = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14]
    if choice == 1:
        op = open('screeplot_rand.csv', 'wb')
    elif choice == 0:
        op = open('screeplot_strat.csv', 'wb')
    writer = csv.writer(op)
    pairs = zip(eigen_vals, components)
    for i, j in pairs:
        writer.writerow([i, j])
    op.close()
    plt.plot(components, eigen_vals )
    labels = list(data)
    plt.xticks(components, labels, rotation='vertical')
    plt.title('Intrinsic Dimensionality')
    plt.show(block = True)


def topPcaLoadings(data_frame, components, choice):
    data = scale(data_frame.values)
    pca = PCA(components)
    pca.fit_transform(data)
    feature_loadings = pca.components_[:4] # 4 is the intrinsic dimensionality
    feature_loadings = np.sum(np.square(feature_loadings), axis=0) # summing contribution of each feature
    top =  feature_loadings.argsort()[-3:][::-1]
    headers = list(data_frame)
    df = pd.DataFrame()
    print "Top 3 features are"
    for i in top:
        print headers[i], '\t', feature_loadings[i]
        df[headers[i]] = data_frame[headers[i]]
    if choice == 1:
        df.to_csv('top3PCA_rand.csv')
    elif choice == 0:
        df.to_csv('top3PCA_strat.csv')


def top2Projection(data_frame, choice):
    data = scale(data_frame.values)
    pca = PCA(n_components=2)
    pca.fit_transform(data)
    vecs = pca.components_
    f1 = np.matmul(data, vecs[0])
    f2 = np.matmul(data, vecs[1])
    pairs = zip(f1, f2)
    if choice == 1:
        op = open("top2_rand.csv",'wb')
    elif choice == 0:
        op = open("top2_strat.csv", 'wb')
    writer = csv.writer(op)
    for i, j in pairs:
        writer.writerow([i, j])
    op.close()
    plt.scatter(f1,f2)
    plt.title('Data along top 2 components')
    plt.show(block = True)


def euclideanMDS(dataframe, choice):
    dis_mat = SK_Metrics.pairwise_distances(dataframe, metric= "euclidean")
    mds = MDS(n_components=2, dissimilarity='precomputed')
    data = mds.fit_transform(dis_mat)
    print 'this is euclidean \n'
    print data
    if choice == 1:
        op = open("mdse_rand.csv", 'wb')
    elif choice == 0:
        op = open("mdse_strat.csv", 'wb')
    writer = csv.writer(op)
    x = []
    y = []
    for row in data:
        writer.writerow([row[0], row[1]])
        x.append(row[0])
        y. append(row[1])
    op.close()
    plt.scatter(x,y)
    plt.title('MDS Euclidean')
    plt.show(block = True)


def correlationMDS(dataframe, choice):
    dis_mat = SK_Metrics.pairwise_distances(dataframe, metric = "correlation")
    mds = MDS(n_components=2, dissimilarity='precomputed')
    data = mds.fit_transform(dis_mat)
    if choice == 1:
        op = open("mdsc_rand.csv", 'wb')
    elif choice == 0:
        op = open("mdsc_strat.csv", 'wb')
    writer = csv.writer(op)
    x,y = [],[]
    for row in data:
        writer.writerow([row[0], row[1]])
    for row in data:
        writer.writerow([row[0], row[1]])
        x.append(row[0])
        y. append(row[1])
    op.close()
    plt.scatter(x,y)
    plt.title('MDS Correlation')
    plt.show(block = True)

def top3PCA(data_frame, choice):
    data = scale(data_frame.values)
    pca = PCA(n_components=3)
    pca.fit_transform(data)
    vecs = pca.components_
    df = pd.DataFrame()
    df['f1'] = np.matmul(data, vecs[0])
    df['f2'] = np.matmul(data, vecs[1])
    df['f3'] = np.matmul(data, vecs[2])
    if choice == 1:
        df.to_csv('top3PCA_rand.csv')
    elif choice == 0:
        df.to_csv('top3PCA_strat.csv')
#################################################################################################################

data = pd.read_csv("housing.csv")
sample_strat = stratified_sampling(data, 4, 0.80)
sample_rand = random_sampling(data, 0.80)
correlationMDS(sample_rand, choice = 1) # for MDS
correlationMDS(sample_strat, choice = 0)
euclideanMDS(sample_rand, choice = 1)  # for Euclidean
euclideanMDS(sample_strat, choice = 0)
getCluster(14, data)
getIntrinsic(sample_rand, 14, 1) # for scree plot
getIntrinsic(sample_strat, 14, 0)
top2Projection(sample_rand, choice = 1) # for top 2 components
top2Projection(sample_strat, choice = 0)
topPcaLoadings(sample_rand, 14, choice = 1) # for scatter plot matrix
topPcaLoadings(sample_strat, 14, choice = 0)



