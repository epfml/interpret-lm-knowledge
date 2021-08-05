# !pip install git+https://github.com/huggingface/transformers
# !pip install textacy
import sys

experiment_data = 'squad'
experiment_model = 'Bert'
use_spacy = True
if len(sys.argv) > 1:
    experiment_data = str(sys.argv[1])
    experiment_model = str(sys.argv[2])
    use_spacy = True if len(sys.argv) >= 4 else False

import urllib.request
import zipfile
from transformers import pipeline
import json
import pandas as pd
import matplotlib.pyplot as plt
import networkx as nx
import spacy
import textacy

def preprocess_masks(df, experiment_data, experiment_model):
    good_preds_ids = []
    predicted_sentences = []
    ground_truth_sentences = []
    valid_examples = len(df)
    for i in range(len(df)):
      mask = df['masked_sentences'][i]
      if mask.count('[MASK]') > 1 or mask.count('[MASK]') == 0:
        valid_examples -= 1
        continue
      if 're' in experiment_data:
        gold_label = df['obj'][i]
      else:
        gold_label = df['obj_label'][i]
      if 'Roberta' in experiment_model:
        mask = mask.replace('[MASK]', '<mask>')
      preds = pd.DataFrame(unmasker(mask))['token_str']
      if any(preds.str.contains(gold_label)):
        good_preds_ids.append(i)
      top_pred = preds[0]
      if 'Roberta' in experiment_model:
        full_sentence = mask.replace('<mask>', top_pred)
        ground_truth_sentence = mask.replace('<mask>', gold_label)
      else:
        full_sentence = mask.replace('[MASK]', top_pred)
        ground_truth_sentence = mask.replace('[MASK]', gold_label)
      predicted_sentences.append(full_sentence)
      ground_truth_sentences.append(ground_truth_sentence)
    return good_preds_ids, predicted_sentences, ground_truth_sentences, valid_examples

def generate_df(data_type='squad'):
    if data_type == 'squad':
        file = 'data/data/Squad/test.jsonl'
    elif data_type == 're-date-birth':
        file = 'data/data/Google_RE/date_of_birth_test.jsonl'
    elif data_type == 're-place-birth':
        file = 'data/data/Google_RE/place_of_birth_test.jsonl'
    elif data_type == 're-place-death':
        file = 'data/data/Google_RE/place_of_death_test.jsonl'
    else:
        raise NameError("Data file ", data_type, "not available.")

    with open(file, 'r') as json_file:
        json_list = list(json_file)

    df = pd.DataFrame()
    for json_str in json_list:
        result = json.loads(json_str)
        if 're' in data_type:
            dfItem = pd.DataFrame.from_dict({'masked_sentences': result['masked_sentences'], 'obj': str(result['obj'][:4])})
        else:
            dfItem = pd.DataFrame.from_records(result)
        df = df.append(dfItem, ignore_index=True)
    return df

def textacy_extract_relations(text):
    nlp = spacy.load("en")
    doc = nlp(text)
    return textacy.extract.subject_verb_object_triples(doc)

def spacy_extract_relations(text):
    nlp = spacy.load("en")
    doc = nlp(text)
    triples = []

    for ent in doc.ents:
        preps = [prep for prep in ent.root.head.children if prep.dep_ == "prep"]
        for prep in preps:
            for child in prep.children:
                triples.append((ent.text, "{} {}".format(ent.root.head, prep), child.text))

    return triples

def retrieve_data():
    url = "https://dl.fbaipublicfiles.com/LAMA/data.zip"
    extract_dir = "data"

    zip_path, _ = urllib.request.urlretrieve(url)
    with zipfile.ZipFile(zip_path, "r") as f:
        f.extractall(extract_dir)

def retrive_models(experiment_model):
    if experiment_model == 'DistilBert':
        from transformers import DistilBertTokenizer, DistilBertModel
        tokenizer = DistilBertTokenizer.from_pretrained('distilbert-base-uncased')
        model = DistilBertModel.from_pretrained("distilbert-base-uncased")
        unmasker = pipeline('fill-mask', model='distilbert-base-uncased')
    elif experiment_model == 'Bert':
        from transformers import BertTokenizer, BertModel
        tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')
        model = BertModel.from_pretrained("bert-base-uncased")
        unmasker = pipeline('fill-mask', model='bert-base-uncased')
    elif experiment_model == 'Roberta':
        from transformers import RobertaTokenizer, RobertaModel
        tokenizer = RobertaTokenizer.from_pretrained('roberta-base')
        model = RobertaModel.from_pretrained("roberta-base")
        unmasker = pipeline('fill-mask', model='roberta-base')
    else:
        raise NameError("Model has not been implemented yet.")
    return tokenizer, model, unmasker

def generate_kg(predicted_sentences, use_spacy=False):
    nlp = spacy.load("en")
    label_dict = {}
    row_list = []
    if use_spacy:
        extract_relations = spacy_extract_relations
    else:
        extract_relations = textacy_extract_relations

    for text in predicted_sentences:
        relations = extract_relations(text)
        for _source, _relation, _target in relations:
          row_list.append({'source': str(_source), 'target':str(_target), 'edge': str(_relation)})
          label_dict[(str(_source), str(_target))] = str(_relation)

    return pd.DataFrame(row_list), label_dict

def plot_kg(df, label_dict, node_color='skyblue', font_color='red', save_name='img.jpg'):
    G=nx.from_pandas_edgelist(df, "source", "target",
                              edge_attr=True, create_using=nx.MultiDiGraph())

    plt.figure(figsize=(12,12))

    pos = nx.spring_layout(G)
    nx.draw(G, with_labels=True, node_color=node_color, edge_cmap=plt.cm.Blues, pos = pos)
    nx.draw_networkx_edge_labels(G,pos,edge_labels=label_dict,font_color=font_color)
    plt.savefig(save_name)

def generate_merge_df(labels):
    n_df = pd.DataFrame(labels, columns=['source', 'edge'])
    merge_df = pd.DataFrame(n_df['source'].values.tolist(), index=n_df.index, columns=['source', 'target'])
    merge_df['edge'] = n_df['edge']
    return merge_df

retrieve_data()
tokenizer, model, unmasker = retrive_models(experiment_model)

df = generate_df(experiment_data)
good_preds_ids, predicted_sentences, ground_truth_sentences, \
                valid_examples = preprocess_masks(df, experiment_data, experiment_model)

# language model predictions
kg_df, label_dict = generate_kg(predicted_sentences, use_spacy)
kg_df.to_csv(experiment_data + '_' + experiment_model + '_kg_df.csv')

plot_kg(kg_df, label_dict, 'skyblue', 'red', \
        experiment_data + '_' + experiment_model + '_kg.jpg')

# ground truth predictions
g_kg_df, g_label_dict = generate_kg(ground_truth_sentences, use_spacy)
g_kg_df.to_csv(experiment_data + '_' + experiment_model + '_g_kg_df.csv')

plot_kg(g_kg_df, g_label_dict, 'skyblue', 'red', \
        experiment_data + '_' + experiment_model + '_g_kg.jpg')

# compare ground truth and language model predictions
ground_truth_set = set(g_label_dict.items())
relations_set = set(label_dict.items())
intersection_labels = ground_truth_set.intersection(relations_set)
new_relations = relations_set.difference(ground_truth_set)
missed_ground_truth = ground_truth_set.difference(relations_set)

# relations captured by both models
intersection_df = generate_merge_df(intersection_labels)
intersection_df.to_csv(experiment_data + '_' + experiment_model + '_intersection.csv')
plot_kg(intersection_df, dict(intersection_labels), 'green', 'green', \
        experiment_data + '_' + experiment_model + '_intersection.jpg')

# relations caputured by ground truth and missed by language model
missed_ground_truth_df = generate_merge_df(missed_ground_truth)
missed_ground_truth_df.to_csv(experiment_data + '_' + experiment_model + '_missed_ground_truth.csv')
plot_kg(missed_ground_truth_df, dict(missed_ground_truth), 'red', 'pink', \
        experiment_data + '_' + experiment_model + '_missed_ground_truth.jpg')

# new relations captured by language model and missed by ground truth
new_lm_relations_df = generate_merge_df(new_relations)
new_lm_relations_df.to_csv(experiment_data + '_' + experiment_model + '_new_lm_relations.csv')
plot_kg(new_lm_relations_df, dict(new_relations), 'blue', 'blue', \
        experiment_data + '_' + experiment_model + '_new_lm_relations.jpg')

print('Done!')
