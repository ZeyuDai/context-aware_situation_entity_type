# context_aware_situation_entity
Code and Data for the EMNLP 2018 paper ["Building Context-aware Clause Representations for Situation Entity Type Classification"](http://www.aclweb.org/anthology/D18-1368)
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
5. You can change the hyperparameters in .py file before the main() function (I am sorry that I didn't write code for config).

--------------------------------------------------------------------
About Preprocessing:
1. Download both Google [word2vec](https://drive.google.com/file/d/0B7XkCwpI5KDYNlNUTTlSS21pQmM/edit) and preprocessed [POS/NER file](https://drive.google.com/open?id=1tEhQghyVF7qeIbuwgkAYkbsGpTItPp64) (You can also generate them by yourself by downloading Standford [CoreNLP toolkit](https://stanfordnlp.github.io/CoreNLP/) and put them in ./resource); put them in folder ./resource/ <br/>
2. The MASC+Wiki dataset raw files are already in the ./dataset/MASC_Wikipedia (copied from this [repo](https://github.com/annefried/sitent/tree/master/annotated_corpus))<br/>
3. run ```python preprocess.py``` <br/> 


--------------------------------------------------------------------
Package Version:<br/>
python == 2.7.10<br/>
torch == 0.3.0<br/>
nltk >= 3.2.2<br/>
gensim >= 0.13.2<br/>
numpy >= 1.13.1<br/>
beautifulsoup4 >= 4.6.0<br/>
