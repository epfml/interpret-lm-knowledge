# interpret-lm-knowledge

**Idea**: How do we understand what a language model is learning at various stages of training? Language models have been recently described as [open knowledge bases](https://github.com/facebookresearch/LAMA). We can generate knowledge graphs by extracting relation triples from masked language models at various stages of training to examine the knowledge acquisition process.  

**Dataset**: Squad, Google-RE (from LAMA paper)  

**Models**: BERT, RoBeRTa, DistilBert, training RoBERTa from scratch

**Contributors**: [Vinitra Swamy](https://github.com/vinitra), [Angelika Romanou](https://github.com/agromanou), Karl Aberer, [Martin Jaggi](https://github.com/martinjaggi)

## Quick Start Guide
### Pretrained Model (BERT, DistilBERT, RoBERTa) -> KG 
1. Install requirements and clone repository  
```
!pip install git+https://github.com/huggingface/transformers   
!pip install textacy
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
See [`run_lm_experiments notebook`](run_lm_experiments.ipynb) for examples.

## Train LM model from scratch -> KG
1. Install requirements and clone repository
```
!pip install git+https://github.com/huggingface/transformers
!pip list | grep -E 'transformers|tokenizers'
!pip install textacy
```
2. Run [`wikipedia_train_from_scratch_lm.ipynb`](wikipedia_train_from_scratch_lm.ipynb).
3. As included in the last cell of the notebook, you can run the KG generation experiments by:
```
from run_training_kg_experiments import *
run_experiments(tokenizer, model, unmasker, "Roberta3e")
```
