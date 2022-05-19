# Extraction of arguments from Thai article <br/> (การสกัดข้อความโต้แย้งจากบทความภาษาไทย)
  In this project is intended for the use of machine learning (Machine learning) Create a model to identify claim and premise of the disputed text from the Thai article. In the preparation of this project, 500 Thai-language articles archives were used

### Web Application (argument-targer)
https://github.com/pitiwatMIKE/project_web_argument_targer

## Project structure

```
|__ Summarize
        |__ summarize.ipynb/ --> Used for plot graphs to compare the accuracy of models.
|__ create_dateset
        |__ create_data_comment.ipynb/ -->  create a Dataset in the format of CoNLL
|__ dataset
        |__ CoNLL2002-dataset/ --> data in format CoNll
               |__ comment-pos.conll
               |__ comment.conll
               |__ test_CONLL_BERT.txt
               |__ train_CONLL_BERT.txt
        |__ data/ --> Data in CoNll format saved with pickle
               |__ comment-pos.data
               |__ comment.data
        |__ LAW/ --> Raw data for tagging claim and premise
            |__ ...
|__ thai2vec
        |__ thai2vecNoSym.bin/ --> word_embeding for model lstm, bilstm, bilstm-crf
|__ train_model/ --> Used for training each model.
        |__ mode_CRF
              |__ Model_CRF.ipynb
        |__ model_BiLSTM
              |__ model_BiLSTM.ipynb
        |__ model_BiLSTM-CRF
              |__ model_BiLSTM-CRF.ipynb
        |__ model_LSTM
              |__ model_LSTM.ipynb
        |__ wangchanbert_hugginface/ --> train use libraly tranformers
              |__ colab_train_model.txt/ --> Train on colab to use Colab GPUs.
              |__ use_model.ipynb
              |__ wangchanberta.ipynb
        |__ wangchanbert_simpletranformers/ --> train use libraly simpletranformers (recommend)
              |__ colab_train_model.txt/ --> Train on colab to use Colab GPUs.
              |__ evaluation.ipynb
              |__ train_bert_lazy.ipynb
              |__ use_pipline.ipynb
|__ trained_model/ --> trained model
        |__ model_BiLSTM
        |__ model_BiLSTM-CRF
        |__ CRF
        |__ LSTM
|__ use_model/ --> How to use a trained model
        |__ use_model_BiLSTM-CRF.ipynb
        |__ use_model_BiLSTM.ipynb
        |__ use_model_CRF.ipynb
        |__ use_model_LSTM.ipynb
|__ .gitignore
|__ requirements.txt --> file for managing packages requirements
```

# Installation
Creating virtual environments
```
sudo apt update
sudo apt install python3.8
python3.8 -m venv env_argument
```
Activating a virtual environment
```
source env_argument/bin/activate
```
clone project
```
git clone https://github.com/pitiwatMIKE/project_argument.git
```
install packages requirements
```
cd project_argument
pip install -r requirements.txt
```
    
# Summarize
| Model| F1(Claim) | F1(Premise) | Accuracy |
|-------| ------- | :---------: | :------: |
|CRF| 0.37 |0.32 | 0.59 |
|LSTM| 0.15 |0.16 | 0.60 |
|BiLSTM| 0.17 |0.17 | 0.63 |
|BiLSTM-CRF| 0.23 |0.13 | 0.41 |
|Wangchanberta| 0.51 |0.39 | 0.65 |

| Model| F1(B-c) | F1(B-p) | F1(I-c) | F1(I-p) | Accuracy |
|-------| ------- | :---------: | :------: | :------: | :------: |
|CRF| 0.66 | 0.56 | 0.50 | 0.72 | 0.59 |
|LSTM| 0.32 | 0.59 | 0.54 | 0.69 | 0.60 |
|BiLSTM| 0.76 | 0 .58| 0.59 | 0.67 | 0.63 |
|BiLSTM-CRF| 0.41 | 0.34 | 0.32 | 0.48 | 0.41 |
|Wangchanberta| 0.75 | 0.64 | 0.65 | 0.72 | 0.65 |

## Bar Plot
![sentence](https://user-images.githubusercontent.com/68042822/163938892-591039d1-f0da-4a3d-bf82-a65549525b1e.png)
![word](https://user-images.githubusercontent.com/68042822/163939766-3c48e769-49a8-41ed-9cb9-e283b5665b79.png)

## CRF
| Accuracy | 0.59 |
| :------: | :------: |

|  | Precision | Recall | F1 | support |
|-------| :---------: | :------: | :------: | :------: |
| B-c | 0.74 | 0.59 | **0.66** | 176  |
| B-p | 0.65 | 0.56 | **0.56** |  176 |
| I-c | 0.52 | 0.50 | **0.50** | 1910  |
| I-p | 0.71 | 0.73 | **0.72** |  5083 |
|micor avg| 0.66 | 0.66 | 0.66 | 7336 |
|macro avg| 0.65 | 0.57 | 0.61 | 7336 |
|weighted avg| 0.66 | 0.66 | 0.66 | 7336 |


|  | Precision | Recall | F1 | support |
|-------| :---------: | :------: | :------: | :------: |
| C | 0.41 | 0.33 | **0.37** | 176  |
| P | 0.38 | 0.28 | **0.32** |  176 |
|micor avg| 0.40 | 0.31 | 0.35 | 352 |
|macro avg| 0.40 | 0.31 | 0.35 | 352 |
|weighted avg| 0.40 | 0.31 | 0.35 | 352 |

## LSTM
| Accuracy | 0.60 |
| :------: | :------: |

|  | Precision | Recall | F1 | support |
|-------| :---------: | :------: | :------: | :------: |
| B-c | 0.82 | 0.20 | **0.32** | 157  |
| B-p | 0.75 | 0.48 | **0.59** |  158 |
| I-c | 0.58 | 0.51 | **0.54** | 1735  |
| I-p | 0.60 | 0.81 | **0.69** |  3727 |
|micor avg| 0.60 | 0.70 | 0.64 | 5777 |
|macro avg| 0.69 | 0.50 | 0.53 | 5777 |
|weighted avg| 0.60 | 0.70 | 0.63 | 5777 |


|  | Precision | Recall | F1 | support |
|-------| :---------: | :------: | :------: | :------: |
| C | 0.13 | 0.17 | **0.15** | 157  |
| P | 0.11 | 0.25 | **0.16** |  158 |
|micor avg| 0.12 | 0.21 | 0.15 | 315 |
|macro avg| 0.12 | 0.21 | 0.15 | 315 |
|weighted avg| 0.12 | 0.21 | 0.15 | 315 |


## BiLSTM
| Accuracy | 0.63 |
| :------: | :------: |

|  | Precision | Recall | F1 | support |
|-------| :---------: | :------: | :------: | :------: |
| B-c | 0.84 | 0.70 | **0.76** | 157  |
| B-p | 0.77 | 0.47 | **0.58** |  158 |
| I-c | 0.75 | 0.48 | **0.59** | 1735  |
| I-p | 0.65 | 0.68 | **0.67** |  3727 |
|micor avg| 0.68 | 0.62 | 0.65 | 5777 |
|macro avg| 0.75 | 0.58 | 0.65 | 5777 |
|weighted avg| 0.69 | 0.62 | 0.64 | 5777 |


|  | Precision | Recall | F1 | support |
|-------| :---------: | :------: | :------: | :------: |
| C | 0.15 | 0.20 | **0.17** | 157  |
| P | 0.12 | 0.27 | **0.17** |  158 |
|micor avg| 0.14 | 0.23 | 0.17 | 315 |
|macro avg| 0.14 | 0.23 | 0.17 | 315 |
|weighted avg| 0.14 | 0.23 | 0.17 | 315 |

## BiLSTM-CRF
| Accuracy | 0.41 |
| :------: | :------: |

|  | Precision | Recall | F1 | support |
|-------| :---------: | :------: | :------: | :------: |
| B-c | 0.38 | 0.45 | **0.41** | 161  |
| B-p | 0.29 | 0.40 | **0.34** |  170 |
| I-c | 0.32 | 0.31 | **0.32** | 1845  |
| I-p | 0.34 | 0.78 | **0.48** |  3518 |
|micor avg| 0.34 | 0.60 | 0.43 | 5694 |
|macro avg| 0.33 | 0.48 | 0.38 | 5694 |
|weighted avg| 0.34 | 0.60 | 0.42 | 5694 |


|  | Precision | Recall | F1 | support |
|-------| :---------: | :------: | :------: | :------: |
| C | 0.21 | 0.25 | **0.23** | 161  |
| P | 0.11 | 0.16 | **0.13** |  170 |
|micor avg| 0.15 | 0.21 | 0.18 | 331 |
|macro avg| 0.16 | 0.21 | 0.18 | 331 |
|weighted avg| 0.16 | 0.212 | 0.18 | 331 |

## Wangchanberta
| Accuracy | 0.65 |
| :------: | :------: |

|  | Precision | Recall | F1 | support |
|-------| :---------: | :------: | :------: | :------: |
| B-c | 0.73 | 0.77 | **0.75** | 154  |
| B-p | 0.50 | 0.70 | **0.64** |  155 |
| I-c | 0.70 | 0.61 | **0.65** | 1675  |
| I-p | 0.65 | 0.81 | **0.72** |  3637 |
|micor avg| 0.66 | 0.75 | 0.70 | 5621 |
|macro avg| 0.67 | 0.72 | 0.69 | 56221 |
|weighted avg| 0.67 | 0.75 | 0.70 | 5621 |


|  | Precision | Recall | F1 | support |
|-------| :---------: | :------: | :------: | :------: |
| C | 0.45 | 0.59 | **0.51** | 154  |
| P | 0.31 | 0.54 | **0.39** |  155 |
|micor avg| 0.37 | 0.56 | 0.44 | 309 |
|macro avg| 0.38 | 0.56 | 0.45 | 309 |
|weighted avg| 0.38 | 0.56 | 0.45 | 309 |

# Reference
https://python3.wannaphong.com/2018/12/named-entity-recognition-ner-pythainlp.html
https://sklearn-crfsuite.readthedocs.io/en/latest/index.html
https://github.com/wannaphong/thai-ner
https://pytorch.org/tutorials/beginner/nlp/advanced_tutorial.html
https://github.com/achernodub/targer
https://colab.research.google.com/drive/1CWamaQH1Lgd7mSZ0UZ4jx2AUMAGDpsfq?usp=sharing
https://huggingface.co/airesearch/wangchanberta-base-att-spm-uncased


