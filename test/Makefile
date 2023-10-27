all: test
CC_FLAGS = -O2 -Wall --std=c++14 -fPIC 
INC = -I$(ROCM_HOME)/include

test:
	hipcc spmm_test.hip ${CC_FLAGS} ${INC}  -o test

.PHONY: clean
clean:
	-rm ./test