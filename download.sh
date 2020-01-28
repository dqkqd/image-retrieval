#!/bin/bash

# download dataset for training
wget https://www.robots.ox.ac.uk/~vgg/data/oxbuildings/oxbuild_images.tgz

# Download groundtruth file for evaluatation
wget https://www.robots.ox.ac.uk/~vgg/data/oxbuildings/gt_files_170407.tgz

# unzip files
mkdir dataset
mkdir groundtruth
tar -xf oxbuild_images.tgz -C dataset
tar -xf gt_files_170407.tgz -C groundtruth
