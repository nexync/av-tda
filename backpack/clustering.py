import numpy as np

import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

from sklearn.cluster import MeanShift, estimate_bandwidth, KMeans, AgglomerativeClustering

from itertools import cycle, combinations

from .distance import *

'''
This is a clusterer for a particular frame
'''
class Cluster:
    def __init__(self, frame, agents):
        self._frame = frame
        self._agents = agents  # reference to all agents
        self._aii = frame['agent_index_interval']
        
        self._embeddings = np.array([[0, 0, 0, 0, 0]])

        for agent in agents[self._aii[0]: self._aii[1]]:
            embedding = np.concatenate((agent['centroid'], agent['velocity'], [agent['yaw']]))
            self._embeddings = np.append(self._embeddings, [embedding], axis=0)

        self._embeddings = self._embeddings[1:]

        self._cluster_to_indices = None
        self._relative_motion_matrices = None

    def normalize_data(self):
        '''
        Normalize the frame data by by applying the standard score (X - mu) / sigma
        '''
        means = self._embeddings.mean(axis=0)
        stds = self._embeddings.std(axis=0)

        for i in range(len(means)):
            # ensure no nans by replacing last column with zeros if std = 0
            if stds[i] != 0:
                self._embeddings[:, i] = (self._embeddings[:, i] - means[i]) / stds[i]
            else:
                self._embeddings[:, i] = np.zeros((len(self._embeddings)))

    def meanshift_cluster(self, quantile=0.3):
        '''
        Perfoms meanshift clustering algorithm on the data. stores labels, cluster centers, unique labels, and number of clusters
        @param quantile - (default: 0.3). should be between [0, 1]. 0.5 means median of all pairwise distances is used
        '''
        bandwidth = estimate_bandwidth(self._embeddings, quantile=quantile, n_samples=len(self._embeddings))
        ms = MeanShift(bandwidth=bandwidth, bin_seeding=True)

        ms.fit(self._embeddings)

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

        km.fit(self._embeddings)

        self._labels = km.labels_
        self._cluster_centers = km.cluster_centers_
        self._labels_unique = np.unique(self._labels)
        self._n_clusters = len(self._labels_unique)

    def agglomerative_cluster(self, n_clusters=8):
        '''
        '''
        ag = AgglomerativeClustering(n_clusters=n_clusters)

        ag.fit(self._embeddings)

        self._labels = ag.labels_
        self._cluster_centers = np.asarray([0] * len(self._labels))
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
            ax.plot(self._embeddings[my_members, 0], self._embeddings[my_members, 1], self._embeddings[my_members, other], col + '.')

        plt.title('estimated number of clusters: %d' % self._n_clusters)

    def generate_distance_matrices(self, edge_distances=True):
        '''
        Generates and stores distance matricies between all of the cars within each cluster
        @param edge_distances: (bool). if two find distances between nearest edges of cars. if false use centroid
        @return distance_matrices. the list of distance matrices
        '''

        function = distance_from_edges if edge_distances else distance_from_centroids

        self._distance_matrices = self._pairwise_compute(function)

        return self._distance_matrices

    def generate_relative_motion_matrices(self):
        '''
        Generates and stores relative motion matrices between all of the cars within each cluster
        @return relative_motion_matrices - the list of relative motion matrices
        '''

        self._relative_motion_matrices = self._pairwise_compute(relative_motion)

        return self._relative_motion_matrices

    def generate_average_relative_motions(self):
        '''
        Generates list the average relative motions for each of the agents
        @return - list of average relative motion for each agent
        TODO: Try different weighting for this, maybe have different angle mean different things
        '''

        def avg(x):
            return sum(x) / len(x)

        # ensure that relative motion matrices have been made
        if not self._relative_motion_matrices:
            self.generate_relative_motion_matrices()

        self._average_relative_motions = []

        for i in range(self._aii[1] - self._aii[0]):
            # find average for each agent index
            cluster = self._labels[i]

            # edge case for an agent in its own cluster
            if i not in self._relative_motion_matrices[cluster]:
                continue

            relative_motions = list(self._relative_motion_matrices[cluster][i].values())

            average_relative_motion = avg(relative_motions)

            self._average_relative_motions.append(average_relative_motion)

        return self._average_relative_motions


    def _pairwise_compute(self, function):
        '''
        Applyies a function between all agents within each cluster
        @param function -- function to apply between each agent. function should take two agents
        @return matrices -- list of matrices generated
        '''

        if not self._cluster_to_indices:
            self._cluster_to_indices = [None] * self._n_clusters

            for i, label in enumerate(self._labels):
                # if we have not created the array at the particular index yet, do so
                if not self._cluster_to_indices[label]:
                    self._cluster_to_indices[label] = []

                self._cluster_to_indices[label].append(i)

        matrices = []

        for cluster in self._cluster_to_indices:
            matrix = dict()

            for comb in combinations(cluster, 2):
                agent1_index = self._aii[0] + comb[0]
                agent2_index = self._aii[0] + comb[1]

                agent1 = self._agents[agent1_index]
                agent2 = self._agents[agent2_index]

                # some inefficiencies could occur here
                val1 = function(agent1, agent2)
                val2 = function(agent2, agent1)

                if comb[0] not in matrix:
                    matrix[comb[0]] = {}
                if comb[1] not in matrix:
                    matrix[comb[1]] = {}

                matrix[comb[0]][comb[1]] = val1
                matrix[comb[1]][comb[0]] = val2

            matrices.append(matrix)

        return matrices

    @property
    def embeddings(self):
        return self._embeddings

    @property
    def distance_matrices(self):
        return self._distance_matrices
    
    @property
    def relative_motion_matrices(self):
        return self._relative_motion_matrices

    @property
    def average_relative_motions(self):
        return self._average_relative_motions