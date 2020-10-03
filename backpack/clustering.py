import numpy as np

import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

from sklearn.cluster import MeanShift, estimate_bandwidth, KMeans

from itertools import cycle

'''
This is a clusterer for a particular frame
'''
class Cluster:
    def __init__(self, frame, agents):
        self._frame = frame
        self._agents = agents  # reference to all agents
        self._aii = frame['agent_index_interval']
        
        self._frame_data = np.array([[0, 0, 0, 0, 0]])

        for agent in agents[self._aii[0]: self._aii[1]]:
            embedding = np.concatenate((agent['centroid'], agent['velocity'], [agent['yaw']]))
            self._frame_data = np.append(self._frame_data, [embedding], axis=0)

        self._frame_data = self._frame_data[1:]

    def normalize_data(self):
        '''
        Normalize the frame data by by applying the standard score (X - mu) / sigma
        '''
        means = self._frame_data.mean(axis=0)
        stds = self._frame_data.std(axis=0)

        for i in range(len(means)):
            # ensure no nans by replacing last column with zeros if std = 0
            if stds[i] != 0:
                self._frame_data[:, i] = (self._frame_data[:, i] - means[i]) / stds[i]
            else:
                self._frame_data[:, i] = np.zeros((len(self._frame_data)))

    def meanshift_cluster(self, quantile=0.3):
        '''
        Perfoms meanshift clustering algorithm on the data. stores labels, cluster centers, unique labels, and number of clusters
        @param quantile - (default: 0.3). should be between [0, 1]. 0.5 means median of all pairwise distances is used
        '''
        bandwidth = estimate_bandwidth(self._frame_data, quantile=quantile, n_samples=len(self._frame_data))
        ms = MeanShift(bandwidth=bandwidth, bin_seeding=True)

        ms.fit(self._frame_data)

        self._labels = ms.labels_
        self._cluster_centers = ms.cluster_centers_
        self._labels_unique = np.unique(self._labels)
        self._n_clusters = len(self._labels_unique)

    def kmeans_cluster(self, n_clusters=8):
        '''
        Performs k means clustering algorith on the frame data. stores labels, cluster centers, unique labels, and number of clusters
        @param n_clusters - (default: 8). number of clusters to fit on the data
        '''
        km = KMeans(n_clusters=n_clusters)

        km.fit(self._frame_data)

        self._labels = km.labels_
        self._cluster_centers = km.cluster_centers_
        self._labels_unique = np.unique(self._labels)
        self._n_clusters = len(self._labels_unique)

    def plot(self, other=4):
        '''
        Plots the current cluster configuration
        @param other - the z axis data to be ploteed. 2 for vx, 3 for xy, 4 for yaw
        '''
        plt.figure(1)
        plt.clf()

        ax = plt.axes(projection='3d')

        ax.set_xlabel('X pos')
        ax.set_ylabel('Y pos')
        ax.set_zlabel('other')

        colors = cycle('bgrcmykbgrcmykbgrcmykbgrcmyk')

        for k, col in zip(range(self._n_clusters), colors):
            my_members = self._labels == k
            cluster_center = self._cluster_centers[k]
            ax.plot(self._frame_data[my_members, 0], self._frame_data[my_members, 1], self._frame_data[my_members, other], col + '.')

        plt.title('estimated number of clusters: %d' % self._n_clusters)

    @property
    def frame_data(self):
        return self._frame_data