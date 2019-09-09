import pdb
import pickle
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import requests

from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.cluster import AgglomerativeClustering
from sklearn.decomposition import PCA
from sklearn.metrics import pairwise_distances
from sklearn.manifold import TSNE

# Downloading the data
# url = 'http://opendata.auth-6f31f706db6f4a24b55f42a6a79c5086.storage.sbg5.cloud.ovh.net/2019-03-21/ORGANISATION_DE_LETAT_ET_DES_SERVICES_PUBLICS.csv'
# req = requests.get(url)
# with open('ORGANISATION_DE_LETAT_ET_DES_SERVICES_PUBLICS.csv', 'wb') as csv_file:
#     csv_file.write(req.content)

# Comments loading
df       = pd.read_csv('ORGANISATION_DE_LETAT_ET_DES_SERVICES_PUBLICS.csv')
comments = df.iloc[:, 25]
comments = comments.sample(10000)

# Comments cleaning and selection
empty_comments = comments.isnull()
print(empty_comments.sum() / len(comments))
comments       = comments[~empty_comments]

# Removing long comments
# Computes for each comment its number of character.
comments_len = comments.str.len()
# np.arange(0, 1.1, .1) creates an array with values going from 0 to 1
# by increment of .1. To compute the quantile, pandas Series already
# have a method that takes an array of quantiles as parameter. We use
# .iteritems to iterate both on the quantile and its corresponding
# value.
for q, length in comments_len.quantile(np.arange(0, 1.1, .1)).iteritems():
    # {q:3.1f} is used to avoid display floating point numbers round
    # errors like 0.30000000000000004.
    print(f'Quantile {q:3.1f} -> length {length}')

comments = comments[comments_len < comments_len.quantile(.9)]
# We sample a few comments and print them.
print(*comments.sample(10), sep = '\n\n')

# comments = comments.sample(1000)

# Vectorizing comments using TF-IDF
# First we create the vectorizer
tfidf           = TfidfVectorizer(min_df = 10, strip_accents = 'unicode')
# Then we use it to compute the array of vectors corresponding to each
# of the comments.
comment_vectors = tfidf.fit_transform(comments)
# We then convert this sparse array to a dense one because it is
# needed for later processing.
comment_vectors = comment_vectors.toarray()
print(comment_vectors.shape)

# Removing comments that correspond to empty tfidf vectors.
# First we find all the lines in which all the values are 0
empty_vector_mask = np.all(comment_vectors == 0, axis = 1)
# We remove the corresponding comments
comments          = comments[~empty_vector_mask]
# And we also remove these vectors from the vector array.
comment_vectors   = comment_vectors[~empty_vector_mask]
print(comment_vectors.shape)

# Reducing the number of dimensions using Principal Component Analysis
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
    # We are using using string formatting to align properly the
    # displays.
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
    cluster_comments = comments[cluster_mask]
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

selected_comments_final = pd.concat(selected_comments_list)
comment_vectors_final   = np.concatenate(selected_vectors_list, axis = 0)
clusters_final          = np.concatenate(selected_clusters_list, axis = 0)
cluster_reprs           = np.array(cluster_reprs)

for cluster_id, cluster_comments in enumerate(selected_comments_list):
    print(f'#################### Cluster {cluster_id} -> {len(cluster_comments):4d} Elements ####################')
    for comment in cluster_comments.sample(10, replace = True):
        print(f'\t{comment}\n')

comments_tsne_vects = TSNE(n_components = 2, metric = 'cosine').fit_transform(comment_vectors_final)
tsne_mean           = comments_tsne_vects.mean(axis = 0)
tsne_std            = comments_tsne_vects.std(axis = 0)
comments_tsne_vects = (comments_tsne_vects - tsne_mean) / tsne_std
alpha               = .3
cluster_colors      = np.array([
    'magenta',
    'darkblue',
    'darkgreen',
    'orangered',
    'sienna',
    'plum'
])

# plt.figure(figsize = (15, 15))
# plt.scatter(
#     comments_tsne_vects[:, 0],
#     comments_tsne_vects[:, 1,],
#     alpha = alpha,
#     c     = cluster_colors[clusters_final]
# )
# plt.axis('off')
# plt.show()

cluster_coords = []
for cluster_id in range(n_clusters):
    cluster_vects      = comment_vectors_final[clusters_final == cluster_id]
    cluster_vects_tsne = TSNE(n_components = 2).fit_transform(cluster_vects)
    tsne_mean          = cluster_vects_tsne.mean(axis = 0)
    tsne_std           = cluster_vects_tsne.std(axis = 0)
    cluster_vects_tsne = (cluster_vects_tsne - tsne_mean) / tsne_std
    cluster_coords.append(cluster_vects_tsne)
cluster_coords = np.concatenate(cluster_coords, axis = 0)

comment_coords = np.zeros((len(selected_comments_final), 2))
x_shift_scale  = 9
y_shift_scale  = 6
cluster_shifts = np.array([
    (x * x_shift_scale, y * y_shift_scale)
    for x in [-1, 0, 1]
    for y in [0, 1]
])
figure_comment_coords = cluster_shifts[clusters_final] + cluster_coords

plt.figure(figsize = (15, 15))
for cluster_id, (shift_x, shift_y) in zip(range(n_clusters), cluster_shifts):
    cluster_coord = figure_comment_coords[clusters_final == cluster_id]
    plt.scatter(
        x     = cluster_coord[:, 0],
        y     = cluster_coord[:, 1],
        alpha = alpha,
        c     = cluster_colors[cluster_id]
    )
plt.axis('off')
plt.show()

csv_dict = OrderedDict([
    ('comment', selected_comments_final),
    ('cluster', clusters_final),
    ('cluster_name', cluster_reprs[clusters_final]),
    ('x', 5)
])

pdb.set_trace()
