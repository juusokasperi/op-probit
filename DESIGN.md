# Inverse Cumulative Normal
*Juuso Rinta*

## Implementation & Refinement

- **Core Method**: A piecewise rational function.
	- **Central Region** ($x \in[0.02425,0.97575]$): $g(x) \approx u ⋅ P(r)/Q(r)$ where $u = x − 0.5$, $r = u²$. Degrees: P/Q = 5/5.
	- **Tails** ($x \notin [0.02425, 0.97575]$): $g(x) \approx \pm C(t)/D(t)$ where $t = \sqrt{-2\log(\min(x, 1-x))}$. Degrees: C/D = 5/4.

- **Coefficients**: Used [Acklam's pre-computed values](https://stackedboxes.org/2017/05/01/acklams-normal-quantile-function/), which were stored in standard ascending-power order ({a₀, a₁, ... aₙ}) to match a reverse-loop Horner's method.
- **Refinement**: One Halley step is applied to the raw approximation to achieve full double precision.
	- $z ← z - r/(1 - 0.5 · z · r)$
	- **Stable residuals**: For central region, direct formula $r = (\phi(z) - x) / \varphi(z).$ To prevent catastrophic cancellation in the far tails ($x < 10^{−8}$ or $x > 1-10^{-8}$), the residual $r$ is calculated using the ${expm1}$ forms as specified in the brief;

$$
r =
\begin{cases}
 -y \cdot \mathrm{expm1}\!\left( \log(Q(z)) - \log(y) \right) / \varphi(z), & \text{for the right tail, where } y = 1 - x, \\\\[6pt]
 y \cdot \mathrm{expm1}\!\left( \log(Q(-z)) - \log(y) \right) / \varphi(z), & \text{for the left tail, where } y = x.
\end{cases}
$$

- **Join point**: x_low = 0.02425

- **Evaluation**: Horner's method in reverse order (a₀ ... a₅)

### Accuracy Results
**Error Statistics** ($10^{6}$ test points over [10⁻¹², 1-10⁻¹²]):

| Configuration | Max Error | Mean Error | p99 Error |
|--------------|-----------|------------|-----------|
| Baseline (Bisection) | 2.22×10⁻¹⁶ | 4.33×10⁻¹⁷ | 1.11×10⁻¹⁶ |
| Raw (No Halley) | 2.73×10⁻¹⁰ | 1.14×10⁻¹⁰ | 2.70×10⁻¹⁰ |
| **With Halley (Final)** | 2.22×10⁻¹⁶ | 2.31×10⁻¹⁷ | 1.11×10⁻¹⁶ |

**Status**: Exceeds 10⁻¹² target; achieves full double precision. Passed all round-trip, symmetry, monotonicity, and numerical derivative checks. Correctness verified on both random (10⁶) and dense (≈2000) grids, covering [10⁻¹², 1−10⁻¹²]. Symmetry and monotonicity hold. Derivative sanity confirmed to <5×10⁻⁵ relative error.

**Key Insight**: The rational approximation with one Halley step provides more consistant results than bisection.

## Performance Results

**Platform**: Apple M3 (ARM64), Apple Clang v 17.0.0 with -O3 optimization. **Test size**: $10^{6}$ evaluations.

| Configuration | Scalar (ns/op) | Vector (ns/op) | Speedup vs Baseline |
|--------------|----------------|----------------|---------------------|
| Baseline (Bisection) | 1741.93 | 1749.89 | 1.0× |
| New (No Halley, No OpenMP) | 5.68 | 4.78 | **306×** / **366×** |
| **New (Halley, No OpenMP)** | 27.78 | 20.5 | **62×** / **85×** |
| **New (Halley, OpenMP 8 threads)** | 27.69 | 5.34 | **63×** / **327×** |

- **Scalar**: The final implementation without parallelization is **85×** faster than the baseline, expeeding the 10× target.
- **Vector**: Using `#pragma omp parallel for` (8 threads) on the vector overload gives a **3.8×** speedup over the single-threaded vector call, exceeding the 1.5× target.

## Limitations & Non-Idealities

1. **Raw approximation accuracy**: Without Halley refinement, max error is ~2.7×10⁻¹⁰, which slightly exceeds the 10⁻¹⁰ target. One Halley step is required for production use. Refinement adds ~14 ns/op overhead but is necessary for full precision.

2. **Input validation**: Assumes x ∈ (0,1) per specification. No explicit range checking, returns ±∞ at boundaries (preserves baseline API).

3. **Join discontinuity**: Small derivative discontinuity at x = 0.02425, but Halley refinement corrects function values to machine precision.
