#!/usr/bin/env python3
from karateclub import FeatherGraph
from scipy import spatial
import numpy as np
import networkx as nx
import pandas as pd
import os
from nltk.stem.porter import PorterStemmer
from sklearn.decomposition import PCA

stemmer = PorterStemmer()


def extract_embeddings(graphs):
    """
    Compute graph embeddings by fitting a FEATHER graph level embedding
    with the standard hyperparameter settings.
    More on: https://karateclub.readthedocs.io/en/latest/notes/introduction.html#graph-embedding

    :param graphs:  list of networkx.Graph classes
    :return:        np.array, with 500-dim embeddings for each input graph
    """
    # fit model
    model = FeatherGraph(order=10)
    model.fit(graphs)

    embeddings = model.get_embedding()
    embeddings = np.around(embeddings, decimals=3)

    pca = PCA(n_components=7)
    pca.fit(embeddings)

    new_embeddings = np.around(pca.transform(embeddings), decimals=3)

    print(len(embeddings[0]))

    # get embeddings
    return embeddings


def create_matching_dict(nodes):
    """
    Given a set of nodes, it stems the node strings and creates a dict with the stemmed strings as keys
    and an index as the values.

    :param nodes:   networkx.NodeView
    :return:        dict
    """
    keys = set()
    for token in nodes:
        stemmed = stemmer.stem(token)
        keys.add(stemmed)

    return {y: x for x, y in enumerate(keys)}


def from_str_to_ids(edges: list, matching_dict: dict):
    """
    Creates a network with int node names instead of strings.

    :param edges:           networkx.EdgeView
    :param matching_dict:   dict, with <stemmed string> -> <id>
    :return:                networkx.Graph
    """

    def get_similar_id(node_str):
        """
        It matches the stemmed node string with respective id
        """
        stemmed = stemmer.stem(node_str)
        for stem in matching_dict.keys():
            if stemmed == stem:
                return matching_dict[stemmed]

        raise ValueError('Node id not found')

    stemmed_pairs = list()
    for pair in edges:
        source_id = get_similar_id(pair[0])
        target_id = get_similar_id(pair[1])

        stemmed_pairs.append((source_id, target_id))

    kg_df = pd.DataFrame(stemmed_pairs, columns=["source", "target"])
    kg = nx.from_pandas_dataframe(kg_df, "source", "target", create_using=nx.Graph())

    return kg


def cosine(emb_a, emb_b):
    """
    Cosine of angle theta between input vectors.
    Relationship to increasing similarity: Increases

    :param emb_a:   np vector
    :param emb_b:   np vector
    :return:        float
    """
    return 1 - spatial.distance.cosine(emb_a, emb_b)


def euclidean_distance(emb_a, emb_b):
    """
    Distance between ends of the input vectors.
    Relationship to increasing similarity: Decreases

    :param emb_a:   np vector
    :param emb_b:   np vector
    :return:        float
    """
    return np.linalg.norm(emb_a - emb_b)


def prepare_dataset(dataset_names):
    """
    Loads the kg datasets and transforms them to NetworkX Graph classes.
    """
    graphs = list()

    # # load ground truth
    # kg_df = pd.read_csv(os.path.join('new_results', '{}_g_kg_df.csv'.format(dataset_names[0])))[
    #     ['source', 'target', 'edge']]
    # kg = nx.from_pandas_edgelist(kg_df, "source", "target", create_using=nx.Graph())
    # graphs.append(kg)

    # load trained models
    for file_name in dataset_names:
        kg_df = pd.read_csv(os.path.join('new_results', '{}_kg_df.csv'.format(file_name)))[['source', 'target', 'edge']]
        kg = nx.from_pandas_dataframe(kg_df[["source", "target"]], "source", "target", create_using=nx.MultiDiGraph())
        graphs.append(kg)

    return graphs


def run_experiments(graphs, model_names):
    """
    Runs kg embedding pipeline and returns the results for each model compared with the ground truth.

    :param graphs:          list of networkx.Graph classes
    :param model_names:     list of str
    :return:                pd.DataFrame
    """
    embeddings = extract_embeddings(graphs)

    ground_truth_kg_embedding = embeddings[0]

    results = list()
    for model_name, model_kg_embedding in zip(model_names[1:], embeddings[1:]):
        results.append({'model': model_name,
                        'euclidean_distance': euclidean_distance(model_kg_embedding, ground_truth_kg_embedding),
                        'cosine': cosine(model_kg_embedding, ground_truth_kg_embedding)})

    return pd.DataFrame(results)


if __name__ == '__main__':
    models = ['Roberta', 'Roberta1e_custom', 'Roberta3e_custom', 'Roberta5e_custom', 'Roberta7e_custom',
              'DistilBert', 'Bert']
    datasets = ['squad', 're-date-birth', 're-place-birth', 're-place-death']

    for dataset in datasets:
        print('\nDATASET: {}'.format(dataset))
        file_names = ['{}_{}'.format(dataset, model) for model in models]
        kgs = prepare_dataset(file_names)

        kgs_with_ids = list()
        for k_g in kgs:
            m_d = create_matching_dict(k_g.nodes())
            indexed_kg = from_str_to_ids(k_g.edges(), m_d)
            kgs_with_ids.append(indexed_kg)
        results_df = run_experiments(kgs_with_ids, models)
        print(results_df)
