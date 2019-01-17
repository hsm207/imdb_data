# Introduction

This repository contains code to create a tsv file of the [IMDb dataset](http://ai.stanford.edu/~amaas/data/sentiment/) using
the [tensor2tensor](https://github.com/tensorflow/tensor2tensor) library.

# Usage

1. Create and switch to a new Python 3.6+ environment.
2. Navigate to the project's root directory.
3. Execute:
    ```bash
    pip install -r requirements.txt
    ```
4. Execute:
    ```bash
    python create_imdb_dataset.py --output_dir OUTPUT_DIR
    ```
    where `OUTPUT_DIR` is the path to where you want to save the training
    and test files.
