all: test
CC_FLAGS = -O2 -Wall --std=c++17 -fPIC
INC = -I$(ROCM_HOME)/include
extra_link = -lrocsparse

test:
	hipcc spmm_test.hip ${extra_link} ${CC_FLAGS} ${INC}

.PHONY: clean
clean:
	-rm a.out
