#!/usr/bin/env python3

import pandas as pd

from scripts.approximated_ged.VanillaAED import VanillaAED
from scripts.approximated_ged.VanillaHED import VanillaHED

from scripts.run_graph2vec_experiments import prepare_dataset, create_matching_dict, from_str_to_ids


def get_aed(kg_df, g_kg_df):
    """
    Computes Vanilla Aproximated Edit distance, implements basic costs for substitution insertion and deletion.
    """
    aed = VanillaAED()
    dist_aed, _ = aed.ged(kg_df, g_kg_df)
    return dist_aed


def get_hed(kg_df, g_kg_df):
    """
    Computes Vanilla Hausdorff Edit distance, implements basic costs for substitution insertion and deletion.
    """
    aed = VanillaHED()
    dist_hed, _ = aed.ged(kg_df, g_kg_df)
    return dist_hed


def run_kg_experiment(graphs, model_names):
    """
    For each knowledge graph of the model variants, it calculates the AED and HED distances
    from the RoBERTa pre-trained model.
    """
    ground_truth_kg = graphs[0]

    results = list()
    for model_name, model_kg in zip(model_names[1:], graphs[1:]):
        results.append({'model': model_name,
                        'aed': get_aed(model_kg, ground_truth_kg),
                        'hed': get_hed(model_kg, ground_truth_kg)})

    return pd.DataFrame(results)


if __name__ == '__main__':
    models = ['Roberta', 'Roberta1e_custom', 'Roberta3e_custom', 'Roberta5e_custom', 'Roberta7e_custom',
              'DistilBert', 'Bert']
    # datasets = ['squad', 're-date-birth', 're-place-birth', 're-place-death']
    datasets = ['squad']

    for dataset in datasets:
        print('\nDATASET: {}'.format(dataset))
        file_names = ['{}_{}'.format(dataset, model) for model in models]
        kgs = prepare_dataset(file_names)

        kgs_with_ids = list()
        for k_g in kgs:
            m_d = create_matching_dict(k_g.nodes())
            indexed_kg = from_str_to_ids(k_g.edges(), m_d)
            kgs_with_ids.append(indexed_kg)

        results_df = run_kg_experiment(kgs_with_ids, models)
        print(results_df)

