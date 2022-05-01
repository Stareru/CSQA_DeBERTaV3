# Solution of DeBERTaV3 on CommonsenseQA

The implementation of DeBERTaV3-based commonsense question answering on CommonsenseQA. 

[DeBERTa](https://arxiv.org/abs/2111.09543) is a pre-trained language model with an enhanced decoding procedure. [DeBERTaV3](https://openreview.net/forum?id=XPZIaotutsD) refines the training process by replacing the initial training objective, masked language modeling, with replaced token detection. 

We formalize the question selection as a text classification by transforming a question-answer pair into the following prompt. 

```Q [SEP] A.```

where Q, A, \[SEP\] refer to the question, answer and separation token. For instance,

```Where can you find all of space? [SEP] Universe.```

We use a text classifier with DebertaV3 as the backbone to score the prompt. We score the five answer candidates of a question and use cross-entropy loss with the correct label as the objective to train the model.

# Performance
**CommonsenseQA 1.0**
| Method            |  Single | Ensemble |
| ----------------- | :-----: | :------: |
| `deberta-base`    |  60.3   |   62.2   |
| `deberta-large`   |  76.5   |   78.8   |
| `debertav3-base`  |  78.7   |   79.6   |
| `debertav3-large` |  84.1   |   85.3   |

**CommonsenseQA 2.0**
| Method            |  Single |
| ----------------- | :-----: |
| `deberta-base`    |  52.0   |
| `deberta-large`   |  53.3   |
| `debertav3-base`  |  56.8   |
| `debertav3-large` |  68.2   |

# Environment
python=3.8.5\
numpy=1.20.1\
torch=1.9.1+cu102\
transformers=4.10.0\
tqdm=4.62.2

# Source and Citation
[Source](https://www.tau-nlp.org/commonsenseqa) of CommonsenseQA 1.0\
[Source](https://allenai.github.io/csqa2/) of CommonsenseQA 2.0

The initial DeBERTaV3:
```bib
@misc{he2021debertav3,
      title={DeBERTaV3: Improving DeBERTa using ELECTRA-Style Pre-Training with Gradient-Disentangled Embedding Sharing}, 
      author={Pengcheng He and Jianfeng Gao and Weizhu Chen},
      year={2021},
      eprint={2111.09543},
      archivePrefix={arXiv},
      primaryClass={cs.CL}
}
```
