#ifndef MNIST_H
#define MNIST_H

#include <stdio.h>
#include <stdlib.h>
#include <errno.h>
#include <stdint.h>
#include <endian.h>

typedef struct LabelData {
    uint32_t magicNumber;
    uint32_t numLabels;
    uint8_t* labels;
} LabelData;

typedef struct ImageData {
    uint32_t magicNumber;
    uint32_t numImages;
    uint32_t numRows;
    uint32_t numCols;
} ImageData;

int loadLabels(const char* labelsPath, struct LabelData* labelData);
int loadImages(const char* imagesPath, struct ImageData* imageData);

#endif
