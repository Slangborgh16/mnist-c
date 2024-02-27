# Based off of the Makefile found here: https://stackoverflow.com/a/30602701

SRC_DIR := src
OBJ_DIR := obj
BIN_DIR := .

EXE := $(BIN_DIR)/mnistc
SRC := $(wildcard $(SRC_DIR)/*.c)
OBJ := $(SRC:$(SRC_DIR)/%.c=$(OBJ_DIR)/%.o)

CPPFLAGS := -Iinclude -MMD -MP
CFLAGS := -Wall -ggdb
LDFLAGS :=
LDLIBS := -lm

URL := http://yann.lecun.com/exdb/mnist/
DATASET_DIR := dataset
DATASET_FILES := train-labels-idx1-ubyte train-images-idx3-ubyte t10k-labels-idx1-ubyte t10k-images-idx3-ubyte

.PHONY: all download clean

all: $(EXE) download

# Build the executable
$(EXE): $(OBJ) | $(BIN_DIR)
	$(CC) $(LDFLAGS) $^ $(LDLIBS) -o $@

$(OBJ_DIR)/%.o: $(SRC_DIR)/%.c | $(OBJ_DIR)
	$(CC) $(CPPFLAGS) $(CFLAGS) -c $< -o $@

# Download the MNIST dataset
download: $(addprefix $(DATASET_DIR)/,$(DATASET_FILES))

$(DATASET_DIR)/%: $(DATASET_DIR)/%.gz | $(DATASET_DIR)
	@echo "Extracting $@"
	@gunzip -c $< > $@

$(DATASET_DIR)/%.gz: | $(DATASET_DIR)
	@echo "Downloading $(@F)"
	@wget -q -P $(DATASET_DIR) $(URL)$(@F)

# Make required directories
$(BIN_DIR) $(OBJ_DIR) $(DATASET_DIR):
	mkdir -p $@

# Remove build files
clean: 
	@$(RM) -rv $(EXE) $(OBJ_DIR)

-include $(OBJ:.o=.d)
