#include "mnist.h"


int loadLabels(const char* labelsPath, struct LabelData* labelData) {
    FILE* fd = fopen(labelsPath, "r");
    if (fd == NULL) {
        printf("Error opening file: %s\nerrno: %d\n", labelsPath, errno);
        return -1;
    }

    uint32_t magicNumber;
    uint32_t numLabels;

    fread(&magicNumber, sizeof(uint32_t), 1, fd);
    fread(&numLabels, sizeof(uint32_t), 1, fd);

    labelData->magicNumber = be32toh(magicNumber);
    labelData->numLabels = be32toh(numLabels);
    labelData->labels = (uint8_t*)malloc(sizeof(uint8_t) * labelData->numLabels);

    fread(labelData->labels, sizeof(uint8_t), labelData->numLabels, fd);

    if (fclose(fd) == EOF) {
        printf("Error closing file.\nerrno: %d\n", errno);
        return -1;
    }

    return 0;
}


int loadImages(const char* imagesPath, struct ImageData* imageData) {
    FILE* fd = fopen(imagesPath, "r");
    if (fd == NULL) {
        printf("Error opening file: %s\nerrno: %d\n", imagesPath, errno);
        return -1;
    }

    uint32_t magicNumber;
    uint32_t numImages;
    uint32_t numRows;
    uint32_t numCols;

    fread(&magicNumber, sizeof(uint32_t), 1, fd);
    fread(&numImages, sizeof(uint32_t), 1, fd);
    fread(&numRows, sizeof(uint32_t), 1, fd);
    fread(&numCols, sizeof(uint32_t), 1, fd);

    imageData->magicNumber = be32toh(magicNumber);
    imageData->numImages = be32toh(numImages);
    imageData->numRows = be32toh(numRows);
    imageData->numCols = be32toh(numCols);

    if (fclose(fd) == EOF) {
        printf("Error closing file.\nerrno: %d\n", errno);
        return -1;
    }

    return 0;
}
