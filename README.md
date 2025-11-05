Implementation details are in [DESIGN.md](/DESIGN.md)

## How to build test main

### Prerequisites
- `OpenMP` library for parallelization.

**Build** the testing main simply by running `make` at project root. The Makefile builds four binaries;
```bash
./test_baseline					# Baseline implementation that was supplied in the .h file
./test_new_fast_no_halley		# Rational approximation WITHOUT Halley refinement
./test_new_fast					# Rational approximation with one Halley step
./test_new_omp					# Rational approximation with OpenMP parallelization on the vector overload.
```

To **remove** binaries, run `make fclean`.
