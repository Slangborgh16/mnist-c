#ifndef MNIST_H
#define MNIST_H

#include <stdio.h>
#include <stdlib.h>
#include <errno.h>
#include <stdint.h>
#include <endian.h>
#include <string.h>

#include "matrix.h"

typedef struct Dataset {
    uint32_t numImages;
    uint32_t imageRows;
    uint32_t imageCols;

    uint8_t* labels;
    uint8_t* pixelData;
} Dataset;

void loadDataset(Dataset* dataset, const char* labelsPath, const char* imagesPath);
int getLabel(Dataset* dataset, const int index);
Matrix* oneHotEncode(Dataset* dataset, const int index);
Matrix* imgToMatrix(Dataset* dataset, const int index);
void printImage(Dataset* dataset, const int index);

#endif
