#include "../incs/InverseCumulativeNormal.h"
#include <iostream>
#include <vector>
#include <cmath>
#include <chrono>
#include <iomanip>
#include <limits>
#include <cfloat>
#include <omp.h>
#include <random>
#include <numeric>
#include <algorithm>

namespace {
	static inline double Phi_ref(double z) {
		constexpr double INV_SQRT_2 = 0.707106781186547524400844362104849039284835937688474036588;
		return 0.5 * std::erfc(-z * INV_SQRT_2);
	}
}

class Timer {
public:
	Timer(const std::string& name)
		: name_(name), start_time_(std::chrono::high_resolution_clock::now()) {}

	void stop_and_report(size_t num_ops) {
		auto end_time = std::chrono::high_resolution_clock::now();
		auto duration_ns = std::chrono::duration_cast<std::chrono::nanoseconds>(end_time - start_time_).count();
		double ms = duration_ns / 1e6;
		double ns_per_op = (double)duration_ns / (double)num_ops;
		std::cout << "  " << std::setw(30) << name_ << ": "
				  << std::setw(10) << std::fixed << std::setprecision(3) << ms << " ms total,"
				  << std::setw(10) << std::fixed << std::setprecision(3) << ns_per_op << " ns/op\n";
	}

private:
	std::string name_;
	std::chrono::time_point<std::chrono::high_resolution_clock> start_time_;
};

int main() {
	const size_t N_PERF_TEST = 1'000'000;
	quant::InverseCumulativeNormal icn(0.0, 1.0);

	std::cout << "--- Probit Implementation Test ---\n";
	std::cout << "Test Configuration:\n";
#ifdef BASELINE
	std::cout << "  [MODE]: BASELINE (Bisection)\n";
#else
	std::cout << "  [MODE]: NEW (Rational Approx)\n";
#endif
#ifdef ICN_ENABLE_HALLEY_REFINEMENT
	std::cout << "  [HALLEY]: ENABLED\n";
#else
	std::cout << "  [HALLEY]: DISABLED\n";
#endif
#ifdef _OPENMP
	std::cout << "  [OPENMP]: ENABLED (Threads: " << omp_get_max_threads() << ")\n";
#else
	std::cout << "  [OPENMP]: DISABLED\n";
#endif
	std::cout << "-----------------------------------\n\n";

	{
		std::cout << "--- Specific Value Tests ---\n";
		double test_vals[] = {0.001, 0.01, 0.1, 0.25, 0.5, 0.75, 0.9, 0.99, 0.999};
		for (double x : test_vals) {
			double z = icn(x);
			double x_rt = Phi_ref(z);
			double err = std::abs(x_rt - x);
			std::cout << "  x=" << std::setw(6) << x
			<< "  z=" << std::setw(10) << std::setprecision(6) << z
			<< "  Phi(z)=" << std::setw(10) << x_rt
			<< "  err=" << std::scientific << err << std::defaultfloat << "\n";
		}
		std::cout << "\n";
	}

	{
		std::cout << "--- Symmetry Tests ---\n";
		double sym_test[] = {0.1, 0.2, 0.3, 0.4};
		for (double x : sym_test) {
			double z1 = icn(x);
			double z2 = icn(1.0 - x);
			double sym_err = std::abs(z1 + z2);
			std::cout << "  g(" << x << ") + g(" << (1.0-x) << ") = "
			<< z1 << " + " << z2 << " = " << (z1+z2)
			<< "  err=" << std::scientific << sym_err << std::defaultfloat << "\n";
		}
		std::cout << "\n";
	}

	{
		std::cout << "\n--- Large-Sample Accuracy Test (" << N_PERF_TEST << " points) ---\n";

		std::mt19937_64 rng(12345);
		std::uniform_real_distribution<double> dist(1e-12, 1.0 - 1e-12);

		std::vector<double> rt_errors;
		rt_errors.reserve(N_PERF_TEST);
		std::vector<double> sym_errors;
		sym_errors.reserve(N_PERF_TEST);

		double max_sym_err = 0.0;
		double max_rt_err = 0.0;

		for (size_t i = 0; i < N_PERF_TEST; ++i)
		{
			double x = dist(rng);
			double z = icn(x);
			if (!std::isfinite(z)) continue;
			if (x < 0.5 && x > 1e-10)
			{
				double z_sym = icn(1.0 - x);
				if (std::isfinite(z_sym))
				{
					double sym_err = std::abs(z + z_sym);
					max_sym_err = std::max(max_sym_err, sym_err);
					sym_errors.push_back(sym_err);
				}
			}
			double back = Phi_ref(z);
			double rt_err = std::abs(back - x);
			max_rt_err = std::max(max_rt_err, rt_err);
			rt_errors.push_back(rt_err);
		}

		std::sort(rt_errors.begin(), rt_errors.end());
		std::sort(sym_errors.begin(), sym_errors.end());

		double mean_rt_err = std::accumulate(rt_errors.begin(), rt_errors.end(), 0.0) / rt_errors.size();
		double p99_rt_err = rt_errors[static_cast<size_t>(rt_errors.size() * 0.99)];

		double mean_sym_err = sym_errors.empty() ? 0.0
			: std::accumulate(sym_errors.begin(), sym_errors.end(), 0.0) / sym_errors.size();
		double p99_sym_err = sym_errors.empty() ? 0.0
			: sym_errors[static_cast<size_t>(sym_errors.size() * 0.99)];

		std::cout << "  Roundtrip Errors:\n";
		std::cout << "    Max:  " << std::scientific << max_rt_err << "\n";
		std::cout << "    Mean: " << std::scientific << mean_rt_err << "\n";
		std::cout << "    p99:  " << std::scientific << p99_rt_err << "\n\n";

		std::cout << "  Symmetry Errors:\n";
		std::cout << "    Max:  " << std::scientific << max_sym_err << "\n";
		std::cout << "    Mean: " << std::scientific << mean_sym_err << "\n";
		std::cout << "    p99:  " << std::scientific << p99_sym_err << "\n\n";

		std::cout << "  Target: <= 1e-10 (Goal: 1e-12)\n";
		std::cout << "  Status: ";
		if (max_rt_err <= 1e-12) std::cout << "EXCELLENT! (≤ 1e-12)\n\n";
		else if (max_rt_err <= 1e-10) std::cout << "PASS! (≤ 1e-10)\n\n";
		else std::cout << "FAIL! (>1e-10)\n\n";
	}

	{
		std::cout << "--- Full Accuracy Test ---\n";
		double max_sym_err = 0.0;
		double max_rt_err = 0.0;

		std::vector<double> test_xs;
		std::vector<double> rt_errors;
		std::vector<double> sym_errors;

		for (double x = 1e-12; x < 0.5; x *= 1.5) test_xs.push_back(x);
		for (double x = 0.01; x < 1.0; x += 0.01) test_xs.push_back(x);
		for (double x = 1.0 - 1e-12; x > 0.5; x = 1.0 - (1.0 - x) * 1.5) test_xs.push_back(x);

		for (double x : test_xs) {
			if (x <= 1e-12 || x >= 1.0 - 1e-12) continue;

			double z = icn(x);
			if (!std::isfinite(z)) continue;

			if (x < 0.5 && x > 1e-10) {
				double z_sym = icn(1.0 - x);
				if (std::isfinite(z_sym)) {
					double sym_err = std::abs(z + z_sym);
					max_sym_err = std::max(max_sym_err, sym_err);
					sym_errors.push_back(sym_err);
				}
			}

			double x_rt = Phi_ref(z);
			double rt_err = std::abs(x_rt - x);
			max_rt_err = std::max(max_rt_err, rt_err);
			rt_errors.push_back(rt_err);
		}

		std::sort(rt_errors.begin(), rt_errors.end());
		std::sort(sym_errors.begin(), sym_errors.end());

		double mean_rt_err = std::accumulate(rt_errors.begin(), rt_errors.end(), 0.0) / rt_errors.size();
		double p99_rt_err = rt_errors[static_cast<size_t>(rt_errors.size() * 0.99)];

		double mean_sym_err = sym_errors.empty() ? 0.0
			: std::accumulate(sym_errors.begin(), sym_errors.end(), 0.0) / sym_errors.size();
		double p99_sym_err = sym_errors.empty() ? 0.0
			: sym_errors[static_cast<size_t>(sym_errors.size() * 0.99)];

		std::cout << "  Rountrip Errors:\n";
		std::cout << "    Max:  " << std::scientific << max_rt_err << "\n";
		std::cout << "    Mean: " << std::scientific << mean_rt_err << "\n";
		std::cout << "    p99:  " << std::scientific << p99_rt_err << "\n";

		std::cout << "  Symmetry Errors:\n";
		std::cout << "    Max:  " << std::scientific << max_sym_err << "\n";
		std::cout << "    Mean: " << std::scientific << mean_sym_err << "\n";
		std::cout << "    p99:  " << std::scientific << p99_sym_err << "\n\n";

		std::cout << "  Target: <= 1e-10 (Goal: 1e-12)\n";
		std::cout << "  Status: ";
		if (max_rt_err <= 1e-12) std::cout << "EXCELLENT! (≤ 1e-12)\n\n";
		else if (max_rt_err <= 1e-10) std::cout << "PASS! (≤ 1e-10)\n\n";
		else std::cout << "FAIL! (>1e-10)\n\n";
	}

	{
		std::cout << "--- Monotonicity Test ---\n";
		bool monotonic = true;
		double prev_x = 0.0001, prev_z = icn(prev_x);
		for (double x = 0.0002; x < 1.0; x += 0.001)
		{
			double z = icn(x);
			if (z <= prev_z)
			{
				std::cout << "  FAIL: Non-monotonic at x=" << x
				<< " (z=" << z << " <= prev_z=" << prev_z << ")\n";
				monotonic = false;
				break;
			}
			prev_x = x; prev_z = z;
		}
		if (monotonic) std::cout << "  PASS: Strictly monotonic ✓\n";
		std::cout << "\n";
	}

	{
		std::cout << "--- Horner Evaluation Test ---\n";
		double test_coeffs[] = {1.0, -54.476, 161.586, -155.699, 66.801, -13.281};
		double r = 0.0;
		double result = 0.0;
		for (std::size_t i = 6; i-- > 0; )
		result = result * r + test_coeffs[i];
		std::cout << "  Horner with r=0:    " << result << " (should be 1.0)\n";

		result = 0.0;
		r = 0.25;
		for (std::size_t i = 6; i-- > 0; )
		result = result * r + test_coeffs[i];
		std::cout << "  Horner with r=0.25:    " << result << "\n";
	}

	{
		std::cout << "--- Derivative Sanity (dense) ---\n";
		const double eps = DBL_EPSILON;
		const double h_base = std::cbrt(eps);

		auto phi_pdf = [](double z) {
			static const double INV_SQRT_2PI = 0.3989422804014327;
			return INV_SQRT_2PI * std::exp(-0.5 * z * z);
		};

		std::vector<double> pts;
		pts.reserve(2000);
		for (double x = 1e-12; x <= 1e-6; x *= std::pow(1e-6/1e-12, 1.0/500.0)) pts.push_back(x);
		for (int i = 0; i < 1000; ++i) pts.push_back( (i+0.5) / 1000.0 );
		size_t lower_count = pts.size();
		for (size_t i = 0; i < lower_count && pts[i] < 0.5; ++i) {
			pts.push_back(1.0 - pts[i]);
		}

		std::vector<double> rel_errs;
		rel_errs.reserve(pts.size());
		double max_rel = 0.0;
		long double sum_rel = 0.0L;

		for (double x : pts) {
			if (x <= 1e-12 || x >= 1.0 - 1e-12) continue;
			double scale = std::max(1.0, std::fabs(x));
			double h = h_base * scale;
			if (x - h <= 0.0 || x + h >= 1.0) continue;

			double g_plus = icn(x + h);
			double g_minus = icn(x - h);
			if (!std::isfinite(g_plus) || !std::isfinite(g_minus)) continue;

			double dnum = (g_plus - g_minus) / (2.0 * h);
			double gx = icn(x);
			double phi_gx = phi_pdf(gx);
			double dth = 1.0 / phi_gx;
			double rel = std::abs((dnum - dth) / dth);

			rel_errs.push_back(rel);
			sum_rel += rel;
			if (rel > max_rel) max_rel = rel;
		}

		std::sort(rel_errs.begin(), rel_errs.end());
		double mean_rel = rel_errs.empty() ? 0.0 : double(sum_rel / (long double)rel_errs.size());
		double p99_rel = rel_errs.empty() ? 0.0 : rel_errs[static_cast<size_t>(0.99 * rel_errs.size())];

		std::cout << "  Relative derivative error: max=" << std::scientific << max_rel
				<< " mean=" << mean_rel << " p99=" << p99_rel << "\n\n";
	}

	{
		std::cout << "--- Performance (N = " << N_PERF_TEST << ") ---\n";
		std::vector<double> inputs(N_PERF_TEST), outputs(N_PERF_TEST);
		for (size_t i = 0; i < N_PERF_TEST ; ++i) {
			inputs[i] = (i + 0.5) / (double)N_PERF_TEST;
		}

		{
			Timer t("Scalar Loop");
			for (size_t i = 0; i < N_PERF_TEST; ++i) {
				outputs[i] = icn(inputs[i]);
			}
			t.stop_and_report(N_PERF_TEST);
		}
		{
			Timer t("Vector Overload");
			icn(inputs.data(), outputs.data(), N_PERF_TEST);
			t.stop_and_report(N_PERF_TEST);
		}
		std::cout << "\n  (Sanity check, g(0.5) = " << outputs[N_PERF_TEST / 2] << ")\n";
		std::cout << "-----------------------------------\n";
	}

	return 0;
}
