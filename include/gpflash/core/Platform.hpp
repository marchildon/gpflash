/**
 * Platform.hpp -- portability shims and the OpenMP PERF_ tuning constants.
 *
 * Where unsupported, a shim expands to nothing, and a thread query to its one-thread answer. Put a pragma macro on
 * its own line, immediately above the loop it applies to.
 */
#pragma once

// C++14 for generic lambdas and return-type deduction. It is an interface requirement, not just a build one:
// these headers compile in the consumer's translation unit at the consumer's -std. MSVC is exempt because
// __cplusplus is only accurate there with /Zc:__cplusplus.
#if !defined(_MSC_VER) && (!defined(__cplusplus) || __cplusplus < 201402L)
#error "GpFlash requires C++14 or later."
#endif

// Every input check rests on std::isfinite, which -ffinite-math-only (part of -ffast-math and -Ofast) folds to a
// constant true, leaving the validation compiled but inert.
#if defined(__FINITE_MATH_ONLY__) && __FINITE_MATH_ONLY__
#error "GpFlash: -ffast-math/-Ofast silently disable input validation. Use -fno-math-errno instead (README)."
#endif

// ── Vector math (GCC, glibc, x86-64) ─────────────────────────────────────────
// glibc declares its SIMD exp, log, pow, sin and cos only under __FAST_MATH__; without them GCC cannot vectorize the
// kernel loops. Needs -fno-math-errno. GPFLASH_NO_VECTOR_MATH keeps the scalar versions.
#include <cmath>
#if defined(__GNUC__) && !defined(__clang__) && defined(__x86_64__) && defined(__GLIBC__) \
        && !defined(__FAST_MATH__) && !defined(GPFLASH_NO_VECTOR_MATH)
extern "C" {
__attribute__((__simd__("notinbranch"))) double exp(double) noexcept;
__attribute__((__simd__("notinbranch"))) double log(double) noexcept;
__attribute__((__simd__("notinbranch"))) double pow(double, double) noexcept;
__attribute__((__simd__("notinbranch"))) double sin(double) noexcept;
__attribute__((__simd__("notinbranch"))) double cos(double) noexcept;
}
#endif

// ── Postcondition checks ─────────────────────────────────────────────────────
// Wraps a check on an intermediate calculation to help debug if GPFLASH_DEBUG_CHECKS is defined.
// tst/CMakeLists.txt defines it for the test targets. Input validation is never wrapped: it always runs.
#if defined(GPFLASH_DEBUG_CHECKS)
#define GPFLASH_CHECK(expr) expr
#else
#define GPFLASH_CHECK(expr) ((void)0)
#endif

// ── restrict qualifier ──────────────────────────────────────────────────────
#if defined(__GNUC__) || defined(__clang__)
#define GPFLASH_RESTRICT __restrict__
#elif defined(_MSC_VER)
#define GPFLASH_RESTRICT __restrict
#else
#define GPFLASH_RESTRICT
#endif

// ── printf-style format checking ────────────────────────────────────────────
// Has the compiler check a variadic function's arguments against its format string, as it does printf's.
#if defined(__GNUC__) || defined(__clang__)
#define GPFLASH_PRINTF_FORMAT(idx_format_arg, idx_first_variadic_arg) \
    __attribute__((format(printf, idx_format_arg, idx_first_variadic_arg)))
#else
#define GPFLASH_PRINTF_FORMAT(idx_format_arg, idx_first_variadic_arg)
#endif

// ── Stringize ────────────────────────────────────────────────────────────────
// Outside the compiler branch below, because Version.hpp's version string needs it on every compiler.
#define GPFLASH_STR_(x) #x
#define GPFLASH_STR(x) GPFLASH_STR_(x)

// ── OpenMP `simd` pragmas (GCC/Clang only) ──────────────────────────────────
#if defined(__GNUC__) || defined(__clang__)
#define GPFLASH_OMP_SIMD _Pragma("omp simd")
#define GPFLASH_OMP_SIMD_REDUCTION(...) _Pragma(GPFLASH_STR(omp simd reduction(__VA_ARGS__)))
#else
#define GPFLASH_OMP_SIMD
#define GPFLASH_OMP_SIMD_REDUCTION(...)
#endif

// ── OpenMP thread queries ───────────────────────────────────────────────────
// For sizing a per-thread buffer before a parallel region opens. Without -fopenmp there is one thread.
#if defined(_OPENMP)
#include <omp.h>
#define GPFLASH_OMP_MAX_THREADS() omp_get_max_threads()
#define GPFLASH_OMP_THREAD_NUM() omp_get_thread_num()
#else
#define GPFLASH_OMP_MAX_THREADS() 1
#define GPFLASH_OMP_THREAD_NUM() 0
#endif

// ── Performance constants ───────────────────────────────────────────────────
// Every PERF_ constant selects between paths computing the same quantity, so none changes the model or the
// fitted hyperparameters, though a parallel reduction sums in a different order and is not bitwise
// reproducible. Tuned for one machine and BLAS; starting points elsewhere.
namespace gpflash {
// Below this point count, entering a parallel region costs more than splitting the loop saves.
constexpr int PERF_OMP_MIN_N_POINT = 256;
constexpr int PERF_OMP_COL_CHUNK = 16;      // kernel-matrix / MLL-gradient column loops
constexpr int PERF_OMP_ROW_SUM_CHUNK = 64;  // absolute-row-sum reduction for the Gershgorin bound
// A hyperparameter-gradient column with fewer pairs per dimension than this loops over pairs outside and
// dimensions inside, where the inner loop is the longer one.
constexpr int PERF_GRAD_MIN_PAIR_PER_DIM = 4;
}  // namespace gpflash

// ══ Platform capabilities ═══════════════════════════════════════════════════
// Every platform probe lives here. Library headers elsewhere test the named capability, never the OS or compiler.
// Each can be forced with -D to opt an unprobed platform in.

// Some BLAS scale a transposed tall-narrow DGEMV super-linearly, so the single-query cross-covariance streams
// an inline SIMD dot instead. Only Accelerate has been found to need this. The CMake target sets it from the BLAS
// it found; without that, __APPLE__ stands in for Accelerate.
#if !defined(GPFLASH_BLAS_SKINNY_GEMV_IS_SLOW)
#if defined(__APPLE__)
#define GPFLASH_BLAS_SKINNY_GEMV_IS_SLOW 1
#else
#define GPFLASH_BLAS_SKINNY_GEMV_IS_SLOW 0
#endif
#endif

// Whether std::rename replaces an existing destination. Windows runtimes fail instead, MinGW's included.
#if !defined(GPFLASH_RENAME_REPLACES_EXISTING)
#if defined(_WIN32)
#define GPFLASH_RENAME_REPLACES_EXISTING 0
#else
#define GPFLASH_RENAME_REPLACES_EXISTING 1
#endif
#endif

// The reentrant gmtime: gmtime_s, with C11's arguments reversed, on Windows; POSIX gmtime_r elsewhere. MinGW counts
// as Windows: it declares gmtime_s always and gmtime_r only under _POSIX_C_SOURCE.
#if !defined(GPFLASH_HAVE_GMTIME_S)
#if defined(_WIN32)
#define GPFLASH_HAVE_GMTIME_S 1
#else
#define GPFLASH_HAVE_GMTIME_S 0
#endif
#endif

// When 1, a multistart fit spreads its independent optimizations over MPI processes. Off by default: turning it
// on requires the consumer to link MPI and to initialize it, at MPI_THREAD_FUNNELED or above (optimization/README.md).
#if !defined(GPFLASH_HAVE_MPI)
#define GPFLASH_HAVE_MPI 0
#endif
