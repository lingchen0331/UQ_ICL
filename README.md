# UQ_ICL

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

Note that the parameter value ranges are hyper-parameters, and different range 
may result different performance in different dataset, be sure to tune hyper-parameters carefully. 
