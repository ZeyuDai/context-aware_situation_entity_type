# context_aware_situation_entity
Code and Data for the EMNLP2018 paper ["Building Context-aware Clause Representations for Situation Entity Type Classification"](https://arxiv.org/abs/1809.07483)
```
@inproceedings{dai2018situation,
  title={Building Context-aware Clause Representations for Situation Entity Type Classification},
  author={Dai, Zeyu and Huang, Ruihong},
  booktitle={Proceedings of the 2018 Conference on Empirical Methods in Natural Language Processing},
  year={2018}
}
```

To run the code:
1. Download preprocessed [MASC+Wiki data](https://drive.google.com/open?id=1_GlGplj4cEGeGqLN0EXhTlVX3vLAZ7uJ) in .pt format (All the Words/POS/NER/label and entity boundary information are already transformed to Pytorch vector format) and put it in folder ./data/ <br/>
2. For the model without CRF, run ```python run_situation_entity.py``` <br/>
3. For the model with CRF, run ```python run_CRF_situation_entity.py``` <br/>
4. For cross-validation (on train set), cross-genre and learning curve experiments, modify the Line 381-387 and then run ```python run_CRF_situation_entity_bygenre.py``` <br/>
