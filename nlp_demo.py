import pdb
import pickle
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.cluster import AgglomerativeClustering
from sklearn.decomposition import PCA
from sklearn.metrics import pairwise_distances
from sklearn.manifold import TSNE

# Comments loading
df             = pd.read_csv('ORGANISATION_DE_LETAT_ET_DES_SERVICES_PUBLICS.csv')
comments       = df.iloc[:, 25]
df       = pd.read_csv('ORGANISATION_DE_LETAT_ET_DES_SERVICES_PUBLICS.csv')
comments = df.iloc[:, 25]

# Comments cleaning and selection
empty_comments = comments.isnull()
print(empty_comments.sum() / len(comments))
comments       = comments[~empty_comments]

comments_len   = comments.str.len()
for q, length in comments_len.quantile(np.arange(0, 1.1, .1)).iteritems():
    print(f'Quantile {q:3.1f} -> length {length}')

comments = comments[comments_len < comments_len.quantile(.9)]

# Vectorizing comments using TF-IDF
tfidf = TfidfVectorizer(min_df = 10)
comment_vectors = tfidf.fit_transform(comments)
comment_vectors = comment_vectors.toarray()
print(comment_vectors.shape)

# Removing comments that correspond to empty tfidf vectors.
empty_vector_mask = np.all(comment_vectors == 0, axis = 1)
comments          = comments[~empty_vector_mask]
comment_vectors   = comment_vectors[~empty_vector_mask]
print(comment_vectors.shape)

variance_kept   = .90
pca             = PCA(n_components = variance_kept)
comment_vectors = pca.fit_transform(comment_vectors)
print(comment_vectors.shape)

n_clusters = 6
agglo      = AgglomerativeClustering(
    n_clusters = n_clusters,
    affinity   = 'cosine',
    linkage    = 'average'
)
clusters   = agglo.fit_predict(comment_vectors)
for cluster_id, cluster_size in zip(*np.unique(clusters, return_counts = True)):
    print(f'cluster {cluster_id} -> {cluster_size:4d} elements')

quantile_cutoff        = .3 # The proportion of comments closest to
                            # the center we will keep for the final
                            # figure.
cluster_reprs          = [] # The most representative comment of the
                            # each cluster.
selected_comments_list = [] # The selected comments we will use in the
                            # figure for each cluster.
selected_vectors_list  = [] # The vectors associated to the comments
                            # for each cluster.
selected_clusters_list = [] # The cluster ids for each cluster.
for cluster_id in range(n_clusters):
    # We select all the comments of the current cluster and their
    # corresponding tfidf vectors
    cluster_mask     = clusters == cluster_id
    cluster_comments = selected_comments[cluster_mask]
    cluster_vectors  = comment_vectors[cluster_mask]
    # Compute the average of the cluster vector in order to use it
    # as "center"
    cluster_center   = cluster_vectors.mean(axis = 0)
    # We then compute the cosine distance to the center of the cluster
    dist_to_center   = pairwise_distances(
        cluster_center.reshape(1, -1),
        cluster_vectors,
        metric = 'cosine'
    ).squeeze()
    # We choose the cluster most representative comment by selecting
    # the one closest to the center
    cluster_repr_idx = np.argmin(dist_to_center)
    cluster_repr     = cluster_comments.iloc[cluster_repr_idx]
    cluster_reprs.append(cluster_repr)
    # To clean the comments that we will display, we cut from each cluster
    # the comments that are the furthest away from the center
    cutoff_value = np.quantile(dist_to_center, quantile_cutoff)
    comment_mask = dist_to_center < cutoff_value
    selected_comments_list.append(cluster_comments.iloc[comment_mask])
    selected_vectors_list.append(cluster_vectors[comment_mask])
    selected_clusters_list.append(np.array([cluster_id] * len(selected_comments_list[-1])))
    print(cluster_repr, cutoff_value)
