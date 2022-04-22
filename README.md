# SpatialAssociationsInLM
This repository will contain all data, scripts and results for the paper:

> Alexander Henlein, Alexander Mehler, _"What do Toothbrushes do in the Kitchen? How Transformers Think our World is Structured"._ NAACL 2022.
[__preprint__](https://arxiv.org/abs/2204.05673)


# Instruction
* **A detailed manual and documentation will be added soon.**
* **Additionally, another generalized tool is in the works that can be applied more dynamically to custom data.**


* The data used for evaluation are located in the folder: _data_
* The generated scores between concepts and instances, over which the correlation is calculated later, are stored in the folder: _results_
* The scores themselves were generated using the _CreateAll...Results.py_ scripts.
* The tables with the final distance correlation scores were finally generated with _GenerateTable.py_
* The heatmaps with _GenerateGraph.py_
* The correlation against googleNmaps with _GenerateTable.py_


# BibTeX
```

@InProceedings{Henlein:Mehler:2022,
  Author         = {Henlein, Alexander and Mehler, Alexander},
  Title          = {What do Toothbrushes do in the Kitchen? How Transformers Think our World is Structured},
  BookTitle      = {Proceedings of the 2022 Annual Conference of the North American Chapter of the Association for Computational Linguistics (NAACL 2022)},
  location       = {Seattle, Washington},
  year           = 2022,
  note           = {accepted}
}
```
