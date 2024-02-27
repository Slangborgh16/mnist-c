#ifndef MNIST_H
#define MNIST_H

#include <stdio.h>
#include <stdlib.h>
#include <errno.h>
#include <stdint.h>
#include <endian.h>
#include <string.h>

#include "matrix.h"

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
    uint8_t* pixelData;
} ImageData;

void loadLabels(const char* labelsPath, LabelData* labelData);
void loadImages(const char* imagesPath, ImageData* imageData);
uint8_t* readImage(ImageData* imageData, const int index);
Matrix* imgToMatrix(ImageData* imageData, const int index);
void pgmExport(ImageData* imageData, const int index, const char* outputPath);
Matrix* oneHotEncode(LabelData* labelData, const int index);

#endif
