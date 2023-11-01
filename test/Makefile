all: test
CC_FLAGS = -O2 -Wall --std=c++17 -fPIC
INC = -I$(ROCM_HOME)/include
extra_link = -lrocsparse -lhipsparse

test:
	hipcc spmm_test.hip ${extra_link} ${CC_FLAGS} ${INC} -o test

.PHONY: clean
clean:
	-rm test
