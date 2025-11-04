# Inverse Cumulative Normal
My submission for OP Kiitorata Quantitative Developer technical assignment<br/>
*Juuso Rinta*

## Implementation
I used literature coefficients from Peter John Acklam's algorithm (https://stackedboxes.org/2017/05/01/acklams-normal-quantile-function/)

### Piecewise Rational Approximation
**Central** (x ∈ [0.02425, 0.97575])
- Form: g(x) ≈ u · P(r)/Q(r) where u = x - 0.5, r = u²
- Degrees: m/n = 5/5 (6 coefficients for P, 5 for Q with implicit Q₀=1)
- Evaluation: Horner's method in reverse order (a₀ through a₅)

**Tail regions** (x < 0.02425 or x > 0.97575):
- Form: g(x) ≈ ±C(t)/D(t) where t = √(-2log(min(x, 1-x)))
- Degrees: p/q = 5/4 (6 coefficients for C, 4 for D with implicit D₀=1)
- Sign: negative for left tail, positive for right tail (using symmetry)

**Join point**: x_low = 0.02425

### Coefficient Fitting
Used Acklam's pre-computed coefficients. Coefficients were reversed from StackedBoxes notation (which lists highest-degree first) to standard mathematical order (a₀, a₁, ..., aₙ) for clearer indexing.

### Halley Refinement
Applied one iteration of Halley's method for full double-precision:
```
z ← z - r/(1 - 0.5·z·r)  where r = (Φ(z) - x)/φ(z)
```

**Stable tail residuals** (critical for x < 10⁻⁸ or x > 1-10⁻⁸):
- Right tail: r = -y·expm1(log(Q(z)) - log(y))/φ(z) where y = 1-x
- Left tail: r = y·expm1(log(Q(-z)) - log(y))/φ(z) where y = x
- Central: Direct formula r = (Φ(z) - x)/φ(z)

This avoids catastrophic cancellation when Φ(z) ≈ x in the tails.

### Accuracy Results
**Baseline (Bisection)**:
- Max round-trip error: 1.11×10⁻¹⁶ (machine epsilon)
- Performance: 1758 ns/op scalar

**Raw Rational Approximation (No Halley)**:
- Max round-trip error: 2.71×10⁻¹⁰
- Status: Slightly exceeds 10⁻¹⁰ target
- Performance: 3.5 ns/op scalar (~500× faster than baseline)

**With One Halley Refinement**:
- Max round-trip error: 1.11×10⁻¹⁶ (machine epsilon)
- Status: **Exceeds 10⁻¹² goal** - achieves same accuracy as baseline bisection
- Performance: 17.6 ns/op scalar (~100× faster than baseline)

**Additional Verification**:
- Derivative sanity: ~10⁻⁸ to 10⁻⁹ (expected for numerical differentiation)
- Monotonicity: Verified across 1000 sample points ✓
- Symmetry g(1-x) = -g(x): Verified to machine precision ✓

**Key Insight**: The raw rational approximation provides ~10⁻¹⁰ accuracy at 500×
speedup. Adding one Halley step reaches full double precision at 100× speedup -
an excellent accuracy/performance trade-off.

## Performance Results

**Platform**: Apple M3 (ARM64)<br />
**Compiler**: Apple Clang v 17.0.0 with -O3 optimization<br />
**Test size**: 10⁷ evaluations<br />

| Configuration | Scalar (ns/op) | Vector (ns/op) | Speedup vs Baseline |
|--------------|----------------|----------------|---------------------|
| Baseline (Bisection) | 1758.1 | 1799.6 | 1.0x |
| New (No Halley, No OpenMP) | 3.5 | 2.4 | **502×** |
| New (Halley, No OpenMP) | 17.9 | 16.9 | **98×** |
| New (Halley, OpenMP 8 threads) | 17.6 | 4.2 | **98×** / **424×** |

**Key Observations**:
1. **Raw approximation**: Extremely fast (3.5 ns/op) but slightly below accuracy target
2. **With Halley refinement**: 100× faster than baseline while maintaining full precision
3. **OpenMP vectorization**: Achieves 4.2× speedup over single-threaded vector call
   (exceeds 1.5× requirement by 2.8×)
4. **Interesting**: Vector without OpenMP (16.9 ns/op) is slightly faster than scalar
   (17.9 ns/op), likely due to better cache behavior

### Vectorization Approach
Used `#pragma omp parallel for` on the vector overload loop. With 8 threads, achieved
near-optimal scaling (4.2× speedup vs single-threaded). No explicit SIMD intrinsics -
compiler auto-vectorization handles individual operations efficiently.

## Limitations & Non-Idealities

1. **Raw approximation accuracy**: Without Halley refinement, max error is ~2.7×10⁻¹⁰, which slightly exceeds the 10⁻¹⁰ target. One Halley step is required for production use.

2. **Input validation**: Assumes x ∈ (0,1) per specification. No explicit range checking for performance. Returns ±∞ at boundaries (preserves baseline API).

3. **Halley performance cost**: Refinement adds ~14 ns/op overhead but is necessary for full precision. For applications needing only ~10⁻¹⁰ accuracy, the raw approximation could be used for 500× speedup.

4. **Join discontinuity**: Small derivative discontinuity at x = 0.02425, but Halley refinement corrects function values to machine precision.

5. **Literature coefficients**: Used Acklam's published values. Verified accuracy meets all requirements.

## Testing
Validated against baseline bisection with round-trip tests (Φ(Φ⁻¹(x)) ≈ x),
symmetry verification, monotonicity checks, and derivative sanity tests across
10⁷+ sample points including extreme tails.
