SRC_DIR=src
TESTS_DIR=tests
OBJ_DIR=obj
OUT_DIR=out
EXAMPLES_DIR=examples

CC=g++
CFLAGS=-fopenmp -lpthread -g -lm -Wall -I$(SRC_DIR)

SRC_FILES=$(wildcard $(SRC_DIR)/*.cpp)
EXAMPLES_FILE=$(wildcard $(EXAMPLES_DIR)/*.cpp)
HEADER_FILES=$(wildcard $(SRC_DIR)/*.h)
OBJ_FILES=$(patsubst $(SRC_DIR)/%.cpp,$(OBJ_DIR)/%.o,$(SRC_FILES))

UNIT_TESTS=$(wildcard $(TESTS_DIR)/*.h)

$(OBJ_DIR)/%.o: $(SRC_DIR)/%.cpp $(HEADER_FILES)
	$(CC) -c -o $@ $< $(CFLAGS)

$(OUT_DIR)/yaann_tests : $(OBJ_FILES) $(TESTS_DIR)/yaann_tests.cpp $(UNIT_TESTS)
	$(CC) -o $@ $^ $(CFLAGS) -lgtest -lgmock

$(OUT_DIR)/%: $(EXAMPLES_DIR)/%.cpp $(OBJ_FILES)
	$(CC) -o $@ $^ $(CFLAGS)

.PHONY: all

all:
	$(OUT_DIR)/yaann_tests

.PHONY: clean

clean:
	rm -f $(OBJ_DIR)/*.o rm -f $(OUT_DIR)/*
