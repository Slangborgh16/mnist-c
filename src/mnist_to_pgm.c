#include <stdio.h>
#include <stdlib.h>
#include <stdint.h>
#include <errno.h>
#include "mnist.h"

int main(int argc, char* argv[]) {
    if (argc != 3) {
        printf("Usage: %s <image index> <output file>\n", argv[0]);
        exit(EXIT_FAILURE);
    }

    int index = atoi(argv[1]);

    const char trainingLabelsPath[] = "dataset/train-labels-idx1-ubyte";
    const char trainingImagesPath[] = "dataset/train-images-idx3-ubyte";
    const char testingLabelsPath[] = "dataset/t10k-labels-idx1-ubyte";
    const char testingImagesPath[] = "dataset/t10k-images-idx3-ubyte";

    LabelData trainingLabels;
    ImageData trainingImages;
    LabelData testingLabels;
    ImageData testingImages;

    
    if (loadImages(trainingImagesPath, &trainingImages) == -1) {
        printf("Error reading training images\n");
        exit(EXIT_FAILURE);
    }

    uint8_t* image = readImage(&trainingImages, index);
    if (image == NULL) {
        printf("Index %d out of range. Max value: %d\n", index, trainingImages.numImages - 1);
        free(trainingImages.pixelData);
        exit(EXIT_FAILURE);
    }

    uint32_t rows = trainingImages.numRows;
    uint32_t cols = trainingImages.numCols;
    printf("Rows: %u, Cols: %u\n", rows, cols);

    FILE* fd = fopen(argv[2], "wb");
    if (fd == NULL) {
        printf("Error opening file: %s\nerrno: %d\n", argv[2], errno);
        free(trainingImages.pixelData);
        exit(EXIT_FAILURE);
    }

    fprintf(fd, "P2\n%u %u\n255\n", cols, rows);
    for (uint32_t i = 0; i < rows * cols; ++i) {
        fprintf(fd, "%d ", image[i]);
    }

    if (fclose(fd) == EOF) {
        printf("Error closing file.\nerrno: %d\n", errno);
        free(trainingImages.pixelData);
        exit(EXIT_FAILURE);
    }

    free(trainingImages.pixelData);
    exit(EXIT_SUCCESS);
}
