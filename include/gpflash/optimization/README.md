# Numerical optimization in GpFlash

The `optimization/` headers select kernel hyperparameters by maximizing the marginal log likelihood (MLL) with
analytic gradients. A **fit** is one `select_hp_with_max_mll` call; it may run several local solves. This note
covers what spans several headers. The package `README.md` shows the workflow; each struct documents its fields.

## Public API

| Entry point | Kernel from |
| --- | --- |
| `select_hp_with_max_mll(train_data, optz_opts)` | `optz_opts.model_options.kernel_id`, resolved once per fit |
| `make_posterior_from_fit(train_data, mll_optz_result)` | the result's `ModelOptions`, so the model matches the fit's |
| `fit_and_make_posterior(train_data, optz_opts)` | as the fit; the whole path in one call |

## Layout of the tier

Each directory depends only on the ones below it.

| directory | holds | tested by |
| --- | --- | --- |
| `optimization/` | `HpOptimizer`, `MllOptzObjective`, `MllOptzResult`, `MultistartReport` | `test_gpflash_optimizer`, `_optimizer_periodic`, `_optimizer_mpi`, `_multistart_report` |
| `multistart/` | `RunMultistart`, `MultistartResult`, `OptimumClusters`, `LatinHypercube` | `test_gpflash_multistart`, `_multistart_report` |
| `local_optz/` | `LocalOptzResult` (the backend seam), `OptimizerNlopt` | `test_gpflash_local_optz` |
| `search_space/` | `HpOptzLayout`, `OptzOptions`, `OptzRegion`, `ActiveBounds`, `HpAtBound` | `test_gpflash_search_space`, `_local_optz` |

`search_space/` depends on `core/`, `inputs/` and `kernels/OptzSeeding.hpp`, which lives in `kernels/` because
`KernelBase` names it.

## What is searched

`hp_gamma` and `hp_kern_extra` are always searched; which variances are follows the sign of
`TrainingData::sig2_f_val`, as the package README's [Model conventions](../../../README.md#model-conventions)
tabulates. Every searched parameter is positive and is searched in log space.

`HpOptzLayout` lays the flat vector out as one contiguous block per `HpBlock`; an unsearched block has width 0.
Append to `HpBlock`, never insert: a recorded fit is sliced back against these offsets.

`hp_kern_extra` is bounded, seeded and reported as one block of whatever width the kernel declares, so a new
kernel hyperparameter needs no optimizer change.

**Spaces.** `OptzOptions`, `OptzSeeding` and every error message speak natural space. `build_optz_region` takes the
logarithm once, after checking the bounds in natural space; from there the search is in log space, and
`MllOptzObjective` exponentiates per evaluation. `MultistartReport` exponentiates again, for display only.

## The three steps of a fit

`select_hp_with_max_mll` validates its inputs, then:

1. builds `MllOptzObjective`, whose `HpOptzLayout` says what the vector holds (`MllOptzObjective.hpp`);
2. builds `OptzRegion` — constraint box, sampling box and start, one entry per searched hyperparameter, with the
   kernel's `OptzSeeding` applied and clamped into the box (`search_space/OptzRegion.hpp`);
3. runs the search and builds the fit from its record (`multistart/RunMultistart.hpp`, `HpOptimizer.hpp`).

`MultistartResult` carries the layout and `MllOptzResult` carries the region, so a reader can slice any
optimization's point back into named hyperparameters and derive its at-bound flags without a kernel.

## Heuristics

Every number below is a judgment, not a derivation. The user-tunable ones and their defaults are in
`OptzOptions.hpp`; the rest sit next to their use.

| what | value | where |
| --- | --- | --- |
| first start, `hp_gamma` | `1 / stddev(X_d)`; a constant dimension gives 1 | `OptzRegion.hpp` |
| first start, other blocks | the `_init` options; a variance is a multiple of the residual variance of `f_train` about the mean model's least-squares fit while `scale_variance_bounds_by_residual_var` is set (the default) and that residual is positive, and absolute otherwise | `OptzRegion.hpp`, `MeanModel.hpp` |
| remaining `n_ind_optz - 1` starts | Latin-hypercube points over the sampling box, which is the constraint box unless the kernel narrows it | `RunMultistart.hpp`, `LatinHypercube.hpp` |
| a start that raises a numerical failure, in its MLL or in the backend | recorded with objective `+inf` as `OptzStopReason::Failed`; the fit stands on the rest, and only every start failing raises | `RunMultistart.hpp` |
| two solves reached the same optimum | log-space distance below `SAME_OPTIMUM_LN_DIST` and objectives within `SAME_OPTIMUM_REL_OBJECTIVE` | `OptimumClusters.hpp` |
| a hyperparameter is at a bound | within `AT_BOUND_REL_TOL` of the bound's span | `ActiveBounds.hpp` |
| kernel seeding | a kernel overrides `KernelBase::calc_optz_seeding` to move the start and narrow the sampling box; both are clamped into the constraint box | `kernels/OptzSeeding.hpp` |

`use_multistart` with `use_bounds = false` is rejected: the sampling box is the constraint box, which an
unconstrained fit does not enforce. An unconstrained single-start fit is fine; it starts from the heuristic point
and never samples.

## The local solve

`local_optz/OptimizerNlopt.hpp` is the only optimizer-specific header. `LocalObjective` and `LocalOptzResult` are the
neutral half of the seam; `LocalOptzResult` carries only what the backend sees.

| `NumericalOptzAlgorithm` | NLopt algorithm | |
| --- | --- | --- |
| `Sqp` | `LD_SLSQP` | default: fewest MLL evaluations to the same optimum on a sweep over `n_dim` and `n_train` |
| `Ccsaq` | `LD_CCSAQ` | a different family, to try if `Sqp` struggles |

Both take the analytic gradient and enforce the box natively. `OptzStopReason` says why a solve stopped; only
`EvalBudgetSpent` is a reason to raise `max_evals`.

## Reading the result

`MllOptzResult` stores the fit and derives the rest through accessors:

- `optimality` — distance from the first-order conditions, bounds accounted for.
- `build_best_local_optz_result().stop_reason` — why the winning solve stopped.
- `calc_hp_at_bound_struct()` — which hyperparameters a bound is holding.
- `calc_n_failed_start()` — how many starts failed.

`to_summary_string` is the short report. `MultistartReport.hpp` has the long one: a summary of the fit, counting
distinct optima through `OptimumClusters.hpp`, then a table each of the optimizations, where each landed and where
each started, then a glossary of the terms.

## Running the independent optimizations in parallel

The `n_ind_optz` optimizations share nothing until their results are combined, so they can run as MPI processes.

1. Compile with `-DGPFLASH_HAVE_MPI=1` and link an MPI library; without it the build has no MPI code. The program
   initializes MPI; GpFlash asks `MPI_Initialized` and runs single-process if it was not called. Initialize with
   `MPI_Init_thread` at `MPI_THREAD_FUNNELED` or above, not `MPI_Init`: GpFlash's OpenMP regions run threads
   beside the one calling MPI, which plain `MPI_Init` promises not to. Check the level it provides. `FUNNELED` holds
   when the fit is called from the thread that initialized MPI; call it from any other thread and ask for
   `MPI_THREAD_SERIALIZED`.
2. Budget the threads. Each process sizes OpenMP and BLAS to the whole machine, so under `mpirun` they
   oversubscribe and the fit gets slower as processes are added. Processes times threads should not exceed the
   cores; the `4 x 6` below assumes 24:

```sh
mpirun -n 4 -genv OMP_NUM_THREADS 6 -genv OPENBLAS_NUM_THREADS 6 ./my_fit
```

The third argument of a fit is a `ParallelContext`, which names the processes to spread over:

```cpp
select_hp_with_max_mll(train_data, optz_opts);                                  // every process the program has
select_hp_with_max_mll(train_data, optz_opts, ParallelContext(my_comm));        // a sub-communicator
select_hp_with_max_mll(train_data, optz_opts, ParallelContext(MPI_COMM_SELF));  // serial on every process
```

A fit is a collective over that context: every process must call it, none may skip it, and all must pass the same
training data and options. The training data, the model options, the counts that size the collectives and
`hypercube_sampling_seed` are compared, and a disagreement raises on every process; the other options are not. A
process whose own inputs fail a check, or whose start raises anything but a numerical failure, rethrows its error,
and every other process raises `ErrorPeerFailure`.

The Latin-hypercube design is built from the generator's raw output, which the C++ standard fixes, so every process
builds the same start points from the seed on any platform and standard library.

`max_parallel_ind_optz` caps the processes optimizing at once, since each holds its own `n_train` by `n_train`
factor. The effective count is `min(n_process, max_parallel_ind_optz, n_ind_optz)`, so a single-optimization fit
optimizes on one process while the rest join its collectives.

`MultistartResult` holds arrays with one column per optimization. Each process zeroes them, writes its own columns,
and one `MPI_Allreduce(MPI_SUM)` per array reconstructs the rest, so a new `StartField` travels automatically.
`core/ParallelContext.hpp` is the only file that names MPI; `test_gpflash_optimizer_mpi` is the only test built with
`GPFLASH_HAVE_MPI=1`.
