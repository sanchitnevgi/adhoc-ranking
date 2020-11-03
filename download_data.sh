#!/bin/bash

# Create data folder
mkdir -p ./data

cd data
# Paralleize the data download, 4 processes
cat ../data_files.txt | xargs -n 1 -P 4 wget

cd ../