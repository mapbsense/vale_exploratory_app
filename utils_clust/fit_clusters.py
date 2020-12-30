import numpy as np
from minisom import MiniSom
from pandas import read_csv, get_dummies
from sklearn.cluster import KMeans

from utils_clust.modules.mvar import SOM, classify_data
from utils_io.parse_uploads import save_to_


def mvarSOM(data, n_clusters):
    som_inv = SOM(data, n_clusters)  # number of class. The classes are choose by user
    mapp = som_inv.som
    mapp_r = mapp.reshape(-1, mapp.shape[1])
    label = classify_data(mapp_r, data)
    # err = euclidean_distance(mapp_r[label], data)
    return label  # , err


def miniSOM(data, n_clusters):
    # Initialization and training
    som_shape = (1, n_clusters)
    som = MiniSom(som_shape[0], som_shape[1], data.shape[1], sigma=.5, learning_rate=.5,
                  neighborhood_function='gaussian')
    # each neuron represents a cluster
    winner_coordinates = np.array([som.winner(x) for x in data]).T
    # with np.ravel_multi_index we convert the bidimensional
    # coordinates to a monodimensional index
    cluster_index = np.ravel_multi_index(winner_coordinates, som_shape)
    return cluster_index


def run_kmeans(data, n_clusters):
    km = KMeans(n_clusters)
    return km.fit_predict(data)


def figure_data_w_model(data, model, **kwargs):
    data_path = f'datasets/{data}.csv'
    pred_data = read_csv(data_path)
    n_c = 4 if 'n_clusters' not in kwargs else int(kwargs["n_clusters"])

    drop_cols = None if 'drop_cols' not in kwargs else kwargs["drop_cols"].split(',')

    enter_data = pred_data.copy()
    enter_data = enter_data if drop_cols is None else enter_data.drop(columns=drop_cols)

    if 'std' in kwargs:
        enter_data = (enter_data - enter_data.mean()) / enter_data.std()

    if 'encode' in kwargs:
        enter_data = get_dummies(enter_data, drop_first=True)

    if model == 'OasisSOM':
        label = mvarSOM(np.array(enter_data), n_clusters=n_c)
    elif model == 'MiniSOM':
        label = miniSOM(np.array(enter_data), n_clusters=n_c)
    elif model == 'Kmeans':
        label = run_kmeans(np.array(enter_data), n_clusters=n_c)

    pred_data['labels'] = label
    save_to_(pred_data, data)
    return pred_data
