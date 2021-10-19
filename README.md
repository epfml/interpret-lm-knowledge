# Interpreting Language Models Through Knowledge Graph Extraction

**Idea**: How do we interpret what a language model learns at various stages of training? Language models have been recently described as [open knowledge bases](https://github.com/facebookresearch/LAMA). We can generate knowledge graphs by extracting relation triples from masked language models at sequential epochs or architecture variants to examine the knowledge acquisition process.  

**Dataset**: Squad, Google-RE (3 flavors)  

**Models**: BERT, RoBeRTa, DistilBert, training RoBERTa from scratch

**Authors**: [Vinitra Swamy](https://github.com/vinitra), [Angelika Romanou](https://github.com/agromanou), [Martin Jaggi](https://github.com/martinjaggi)

This repository is the official implementation of the NeurIPS 2021 [Explainable AI Workshop](https://xai4debugging.github.io/) paper titled ["Interpreting Language Models Through Knowledge Graph Extraction"](link:tba). Found this work useful? Please [cite our paper](#citations).

## Quick Start Guide
### Pretrained Model (BERT, DistilBERT, RoBERTa) -> Knowlege Graph
1. Install requirements and clone repository  
```
git clone https://github.com/epfml/interpret-lm-knowledge.git
pip install git+https://github.com/huggingface/transformers   
pip install textacy
cd interpret-lm-knowledge/scripts
```
2. Generate knowledge graphs and dataframes
`python run_knowledge_graph_experiments.py <dataset> <model> <use_spacy>`  
e.g. `squad Bert spacy`  
e.g. `re-place-birth Roberta`    

options:  
```
dataset=squad - "squad", "re-place-birth", "re-date-birth", "re-place-death"  
model=Roberta - "Bert", "Roberta", "DistilBert"  
extractor=spacy - "spacy", "textacy", "custom"
```
See [`run_lm_experiments notebook`](scripts/run_lm_experiments.ipynb) for examples.

### Train LM model from scratch -> Knowledge Graph
1. Install requirements and clone repository
```
!pip install git+https://github.com/huggingface/transformers
!pip list | grep -E 'transformers|tokenizers'
!pip install textacy
```
2. Run [`wikipedia_train_from_scratch_lm.ipynb`](scripts/wikipedia_train_from_scratch_lm.ipynb).
3. As included in the last cell of the notebook, you can run the KG generation experiments by:
```
from run_training_kg_experiments import *
run_experiments(tokenizer, model, unmasker, "Roberta3e")
```

## Citations
```bibtex
@inproceedings{swamy2021interpreting,
 author = {Swamy, Vinitra and Romanou, Angelika and Jaggi, Martin},
 booktitle = {Advances in Neural Information Processing Systems, Workshop on eXplainable AI Approaches for Debugging and Diagnosis},
 title={Interpreting Language Models Through Knowledge Graph Extraction},
 volume = {34},
 year = {2021}
}
```
