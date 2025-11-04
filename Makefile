ifeq ($(shell uname -s)-$(shell uname -m), Darwin-arm64)
	INCS = -I$(shell brew --prefix libomp)/include
endif

HDRS = incs/InverseCumulativeNormal.h

SRCS = srcs/main.cpp

BIN_DIR = bin/

all: baseline fast_core fast_core_no_halley parallelize
	@echo "Compilation complete;"
	@echo "  Baseline test binary: ./$(BIN_DIR)test_baseline"
	@echo "  Rational approximation with Halley test binary: ./$(BIN_DIR)test_new_fast"
	@echo "  Rational approximation (no Halley) test binary: ./$(BIN_DIR)test_new_fast_no_halley"
	@echo "  Parallelized rational approximation test binary: ./$(BIN_DIR)test_new_omp"

baseline: mk_dir
	c++ -O3 -DBASELINE $(INCS) -o $(BIN_DIR)test_baseline $(SRCS)

fast_core: mk_dir
	c++ -O3 -DICN_ENABLE_HALLEY_REFINEMENT $(INCS) -o $(BIN_DIR)test_new_fast $(SRCS)

fast_core_no_halley: mk_dir
	c++ -O3 $(INCS) -o $(BIN_DIR)test_new_fast_no_halley $(SRCS)

parallelize: mk_dir
	c++ -O3 -DICN_ENABLE_HALLEY_REFINEMENT  $(INCS) -Xpreprocessor -fopenmp -L$(shell brew --prefix libomp)/lib -lomp -o $(BIN_DIR)test_new_omp $(SRCS)

mk_dir:
	@mkdir -p $(BIN_DIR)

fclean:
	@rm -rf $(BIN_DIR)
	@echo "Binaries removed."
