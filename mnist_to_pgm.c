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

    uint8_t* image = readImage(&trainingImages, 0);

    uint32_t rows = trainingImages.numRows;
    uint32_t cols = trainingImages.numCols;
    printf("Rows: %u, Cols: %u\n", rows, cols);

    FILE* fd = fopen(outputFile, "wb");
    fprintf(fd, "P2\n%u %u\n255\n", cols, rows);
    for (uint32_t i = 0; i < rows * cols; ++i) {
        fprintf(fd, "%d ", image[i]);
    }
    fclose(fd);

    free(trainingImages.pixelData);
    free(image);
    exit(EXIT_SUCCESS);
}
