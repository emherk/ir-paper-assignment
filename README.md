# DSAIT4050: Information retrieval project

This repository contains code for running experiments on the [2021 TREC Health Misinformation Track dataset](https://trec-health-misinfo.github.io/2021.html) to compare how different IR methods compare in terms of returning misinformative results.

Here we provide the scripts we used to subsample the dataset (i.e. choosing the topics, getting qrels related to those topics, downloading and the documents), as well as the scripts for running the evaluation experiments with different IR methods.

## Repository structure

This is our repository structure. Some folders are not in the repository and are ignored by git, but are created when running the setup bash script. Those folders will be specified as `(ignored)`. Everything within them is also implied to be ignored.

```
├── c4/                      # The c4 repository (ignored)
│   ├── en.noclean/         # Data variant used in our experiments
│   └── ...                 # Other repository folders/files
├── data/                    # Downloaded documents (ignored)
├── eval/                    # Topics and qrels used for evaluation (ignored)
│   └── misinfo-resources-2021/  
|       ├── qrels/          # Qrels folder
|       ├── topics/         # Topics folder
|       └── ...             # Other folders/files
├── index/                   # Downloaded document index location (ignored)
├── .gitignore              # Files and folders to ignore in Git
├── topics.py               # Functions for parsing topics
├── qrels.py                # Functions for parsing qrels
├── docnos.py               # Script for downloading documents
├── index.py                # Functions for indexing documents
├── labels.py               # Only contains a dictionary for mapping qrel labels
├── main.py                 # Main code for running experiments
├── README.md               # Project documentation
├── setup.sh          # Bash script to setup the experiment as we did
└── requirements.txt        # Required dependencies
```

## Setup

We recommend runing the bash script to fully setup the repository for the experiments in order to use the exact same data as we did in our experiments. In a Linux shell:

```
./setup.sh
```

On Windows the script can be executed in a Git Bash shell.

Optionally, the setup can also be customized. Currently the script downloads all documents that have qrels for the first 5 topics (that have at least one qrel) with both a *helpful* and *unhelpful* stances. The number of topics for which you wish to download documents can be adjusted.

The experiments can also be setup manually, by:

1. Installing dependencies:

```
pip install -r requirements.txt
```

2. Cloning the c4 repository (make sure git-lfs is installed on your system):

```
GIT_LFS_SKIP_SMUDGE=1 git clone https://huggingface.co/datasets/allenai/c4
```

3. Downloading and extracting the topics and qrels:

```
curl --progress-bar -L https://trec.nist.gov/data/misinfo/misinfo-resources-2021.tar.gz -o <path-to-eval-dir>/misinfo-resources-2021.tar.gz
```

```
tar -xzf <path-to-eval-dir>/misinfo-resources-2021.tar.gz -C <path-to-eval-dir>
```

4. Downloading the documents:

```
python docnos.py --c4-dir <path-to-c4-repo> --topics-dir <path-to-topics-xml> --qrels-dir <path-to-qrels-file> --n <number-of-topics-to-consider>
```

5. Indexing the documents:

```
python index.py --data-dir <path-to-documents-folder>
```

## Performing experiments

To perform the experiments, simply run the main code file:

```
python main.py
```

The contents of the file can be adjusted to gather different results, e.g. compare other methods, evaluate using different measures.

## Collaboration

While this project was developed primarily for academic purposes, contributions are welcome for further improvements or extensions.

If you'd like to contribute:

1. Fork the repository to your own GitHub account.

2. Create a new branch (feature-branch or bugfix-branch).

3. Make your changes and ensure the code follows best practices.

4. Submit a pull request with a clear description of your changes.

## Contributors

<table>
  <tr>
<!-- TODO: Add Yifan's github stuff -->
    <td align="center"><a href="https://github.com/sun12fff4n"><img src="https://avatars.githubusercontent.com/sun12fff4n?v=4" width="100px;" alt=""/><br /><sub><b>Yifan Sun</b></sub></a></td>
    <td align="center"><a href="https://github.com/emherk"><img src="https://avatars.githubusercontent.com/emherk?v=4" width="100px;" alt=""/><br /><sub><b>Kristóf Sándor</b></sub></a></td>
    <td align="center"><a href="https://github.com/AdoBag"><img src="https://avatars.githubusercontent.com/AdoBag?v=4" width="100px;" alt=""/><br /><sub><b>Adomas Bagdonas</b></sub></a></td>
    <td align="center"><a href="https://github.com/zygis009"><img src="https://avatars.githubusercontent.com/zygis009?v=4" width="100px;" alt=""/><br /><sub><b>Žygimantas Liutkus</b></sub></a></td>
  </tr>
</table>
