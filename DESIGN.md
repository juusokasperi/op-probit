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
**Error Statistics** (10⁷ test points over [10⁻¹², 1-10⁻¹²]):

| Configuration | Max Error | Mean Error | p99 Error |
|--------------|-----------|------------|-----------|
| Baseline (Bisection) | 1.11×10⁻¹⁶ | 3.33×⁻¹⁷ | 1.11×⁻¹⁶ |
| Raw (No Halley) | 2.71×10⁻¹⁰ | 5.88×⁻¹⁰ | 2.62×⁻¹⁰ |
| **With Halley (Final)** | 1.11×10⁻¹⁶ | 1.02×10⁻¹⁷ | 1.11×10⁻¹⁶ |

**Status**: Exceeds 10⁻¹² goal, achieves machine epsilon precision. Passed all round-trip, symmetry, monotonicity, and numerical derivative checks.

**Key Insight**: The rational approximation with one Halley step provides more consistant results than bisection.

## Performance Results

**Platform**: Apple M3 (ARM64), Apple Clang v 17.0.0 with -O3 optimization. **Test size**: 10⁷ evaluations.

| Configuration | Scalar (ns/op) | Vector (ns/op) | Speedup vs Baseline |
|--------------|----------------|----------------|---------------------|
| Baseline (Bisection) | 1758.1 | 1799.6 | 1.0× |
| New (No Halley, No OpenMP) | 3.5 | 2.4 | **502×** |
| **New (Halley, No OpenMP)** | 17.9 | 16.9 | **98×** |
| **New (Halley, OpenMP 8 threads)** | 17.6 | 4.2 | **98×** / **424×** |

- **Scalar**: The final implementation is **98×** faster than the baseline, expeeding the 10× target.
- **Vector**: Using `#pragma omp parallel for` (8 threads) on the vector overload gives a **4.2×** speedup over the single-threaded vector call, exceeding the 1.5× target.

## Limitations & Non-Idealities

1. **Raw approximation accuracy**: Without Halley refinement, max error is ~2.7×10⁻¹⁰, which slightly exceeds the 10⁻¹⁰ target. One Halley step is required for production use. Refinement adds ~14 ns/op overhead but is necessary for full precision. For applications needing only ~10⁻¹⁰ accuracy, the raw approximation could be used for 500× speedup.

2. **Input validation**: Assumes x ∈ (0,1) per specification. No explicit range checking, returns ±∞ at boundaries (preserves baseline API).

3. **Join discontinuity**: Small derivative discontinuity at x = 0.02425, but Halley refinement corrects function values to machine precision.
