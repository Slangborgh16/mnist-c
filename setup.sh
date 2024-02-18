#!/bin/bash

URL=http://yann.lecun.com/exdb/mnist/

TRAIN_LABELS=train-labels-idx1-ubyte
TRAIN_IMAGES=train-images-idx3-ubyte
TEST_LABELS=t10k-labels-idx1-ubyte
TEST_IMAGES=t10k-images-idx3-ubyte

echo Downloading MNIST dataset from: $URL
mkdir dataset
cd dataset

for FILE in $TRAIN_LABELS $TRAIN_IMAGES $TEST_LABELS $TEST_IMAGES; do
    echo Downloading $FILE
    wget -q $URL$FILE.gz
    gunzip $FILE.gz
done

cd ..
echo Done
