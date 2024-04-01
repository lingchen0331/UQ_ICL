# UQ_ICL
This is the official implementation of our *NAACL 2024* paper **Uncertainty Quantification for In-Context Learning of Large Language Models**, the paper can be found [here](https://arxiv.org/abs/2402.10189).

## Dependencies
This code is written in Python. To use it you will need:
- Numpy - 1.16.2
- Scipy - 1.2.1
- pandas - 0.23.4
- Transformers - 4.35.0
- PyTorch 1.10.0+
- datasets - 2.15.0

## Usage
### Data
The data can be downloaded from the file by datasets Python library.

### Test Models
There are five datasets, you can test the results of different datasets with using the executable files (*cola.sh, ag_news.sh, financial.sh, ssh.sh, sentiment.sh*) provided.

Note that the parameter value ranges are hyper-parameters, and different ranges may result in different performances in different datasets, be sure to tune hyper-parameters carefully. 


If you find our paper and implementation are useful in your project, please consider citing our work:
```
@inproceedings{
ling2024uncertainty,
title={Uncertainty Decomposition and Quantification for In-Context Learning of Large Language Models},
author={Chen Ling and Xujiang Zhao and Wei Cheng and Yanchi Liu and Yiyou Sun and Xuchao Zhang and Mika Oishi and Takao Osaki and Katsushi Matsuda and Jie Ji and Guangji Bai and Liang Zhao and Haifeng Chen},
booktitle={2024 Annual Conference of the North American Chapter of the Association for Computational Linguistics},
year={2024},
url={https://openreview.net/forum?id=Oq1b1DnUOP}
}
```
