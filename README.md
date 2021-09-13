# Fairness and Robustness in Invariant Learning: A Case Study in Toxicity classification

This code accompanies the paper [Fairness and Robustness in Invariant Learning: A Case Study in Toxicity Classification](https://arxiv.org/abs/2011.06485), which was featured in a spotlight talk at the 2020 Neurips Workshop on Algorithmic Fairness through the Lens of Causality and Interpretability. 

Thanks to my incredible co-authors [Elliot Creager](https://github.com/ecreager/), [David Madras](https://github.com/dmadras) and [Rich Zemel](https://www.cs.toronto.edu/~zemel/inquiry/home.php). 

## Setup 
Download anaconda, and new conda environment using python 3.8.3. Then, install the required dependencies from the requirements.txt file. 
```
pip install -r requirements.txt
``` 

Next, two different data files must be downloaded for code to function. They can be stored in arbitrary directories.  

1. The civil comments dataset (all_data.csv) can be downloaded at https://www.kaggle.com/c/jigsaw-unintended-bias-in-toxicity-classification/data

2. The FastText word embeddings (crawl-300d-2M.vec) can be downloaded at https://fasttext.cc/docs/en/english-vectors.html


## Reproducing Paper Results 

### Step 1: 
Generate the list of python commands which must be run. Run with -h parameter for more info on the parameters 
```
python launchfiles/setup_params.py RESULTS_DIRECTORY DATA_FILEPATH WORDVEC_PATH
```

### Step 2: 
Run each of the previously generated python commands from the project's home directory. The list is generated in a file called cmdfile.sh, located in the previosuly specified RESULTS_DIRECTORY. To manually specify hyperparameters for a given run, include them as flags to each of the generated python commands (run with -h flag for more details).

### Step 3:
To analyse results and generate tables from paper. 
```
analysis/results.ipynb 
```

## Citing This Work 
Please cite this paper using the following bibtex entry 
```
@inproceedings{adragna20fairness,
  title={Fairness and Robustness in Invariant Learning: A Case Study in Toxicity Classification},
  author={Adragna, Robert and Creager, Elliot and Madras, David and Zemel, Richard},
  booktitle={2020 Neurips Workshop on Algorithmic Fairness through the Lens of Causality and Interpretability},
  year={2020},
}
```
