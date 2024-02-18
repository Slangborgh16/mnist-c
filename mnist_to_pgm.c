#include <stdio.h>
#include <stdlib.h>
#include <stdint.h>
#include "mnist.h"

int main() {
    const char trainingLabelsPath[] = "dataset/train-labels-idx1-ubyte";
    const char trainingImagesPath[] = "dataset/train-images-idx3-ubyte";
    const char testingLabelsPath[] = "dataset/t10k-labels-idx1-ubyte";
    const char testingImagesPath[] = "dataset/t10k-images-idx3-ubyte";

    const char outputFile[] = "test.pgm";

    LabelData trainingLabels;
    ImageData trainingImages;
    LabelData testingLabels;
    ImageData testingImages;

    
    if (loadImages(trainingImagesPath, &trainingImages) == -1) {
        printf("Error reading training images\n");
        exit(EXIT_FAILURE);
    }

    uint32_t x = trainingImages.numCols;
    uint32_t y = trainingImages.numRows;
    printf("Cols: %u, Rows: %u\n", x, y);
    FILE* fd = fopen(outputFile, "wb");
    fprintf(fd, "P2\n%u %u\n255\n", x, y);
    for (uint32_t j = 0; j < y; ++j) {
        for (uint32_t i = 0; i < x; ++i) {
            fprintf(fd, "%d ", 128);
        }
    }
    fclose(fd);
    exit(EXIT_SUCCESS);
}
