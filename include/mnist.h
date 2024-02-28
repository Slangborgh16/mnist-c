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
    uint32_t numPixels;
    uint8_t* pixelData;
} ImageData;

void loadLabels(const char* labelsPath, LabelData* labelData);
void loadImages(const char* imagesPath, ImageData* imageData);
Matrix* imgToMatrix(ImageData* imageData, const int index);
Matrix* oneHotEncode(LabelData* labelData, const int index);
void printImage(Matrix* matrix);

#endif
