# `posterior_cache/` -- persisting a fitted posterior

Building a posterior costs an `O(n_train^3)` Cholesky factorization; evaluating one is cheap. This directory saves
a built posterior to a file so a later run can load it and skip the factorization.

## What a file holds

`save_precompute` writes one file, and only to a path the caller names. It holds:

- what queries need: the fitted hyperparameters, the mean model and its weights, the scaled training points and the
  center subtracted from them, the solved residual behind the posterior mean, and a symmetry model's mirror and
  parity;
- optionally the lower triangle of the Cholesky factor, which the variance needs and which dominates the file's
  size. A posterior loaded from a mean-only file serves the mean and throws if asked for a variance;
- a record of the fit: its kernel, input hash and MLL, the noise variance, `cond_max` and both `EigMaxUbMethod`
  choices. The record never enters a query; `print_cache_info` reports it, and a re-save keeps it;
- an optional free-text `note`, for recording where the training data came from. The data itself is not stored.

The format is HDF5: a small header and one named array per quantity. `CacheLayout.hpp` defines the names and the
header, `Hdf5Store.hpp` does the I/O. Read from Python, matrices appear transposed, since Armadillo is column-major
and HDF5 row-major, and the factor is one vector, its lower triangle column by column. `calc_default_cache_path`
names a file after the kernel and the training set's shape.

## Matching a file to a fit

Each file carries a hash of everything the factorization depends on: the arguments of `calc_input_hash` in
`CacheInputHash.hpp`, which the note is not among. A loader holding the fit's inputs recomputes the hash and refuses
the file if it differs.

The hash catches accidental staleness, not a crafted collision. It is taken over in-memory bytes, so a verified load
on a machine of the opposite endianness refuses every file: safe, but cache files do not travel between such
machines.

## Using a cache

The package `README.md` shows the calls. `load_posterior(path)` checks the file is a valid GpFlash cache.
`load_posterior_verified(path, train_data, hp_val_struct, model_opts)` also checks it came from these inputs; prefer
it whenever they are in hand. A prediction-only build has `load_posterior` alone. To replace a file, save over it.
The format layer itself is `save_precompute` and `load_precompute` in `PosteriorCache.hpp`.

No two cache calls, reads or saves, may run at once, even on different files: HDF5 is linked without thread safety.

## When errors are raised

Reading never creates, modifies or removes a file. A read raises `ErrorInvalidInput` when:

- the file is missing or unreadable;
- another format version, or another kernel, wrote it;
- a stored value is impossible, such as a non-finite likelihood or a non-positive count;
- the stored parts disagree with one another;
- the inputs were supplied and their hash does not match the stored one.

`peek_info` reads the header alone, so it raises only for a missing or unreadable file, another format version, or
an impossible stored value.

A write raises `ErrorInvalidInput` when the destination holds a file that is not a GpFlash cache (checked before
anything is written), when the header would fail a load's checks, or when a write or the final rename fails. It
writes to a temporary beside the destination and renames it over, so a failed save leaves any previous file intact
and removes its temporary; nothing else is ever removed.

## Changing the format

A file records its format version, and a reader accepts only its own, so there is no compatibility in either
direction. The version is read before anything else, so another version's file is refused by name whatever else
changed. A refusal costs one refit, far cheaper than misreading a file.

Bump the version for any change to what a file stores or what a stored value means, including a numerics change that
makes the stored factor differ at the same inputs. That case is easy to miss: the file still loads, and serves wrong
variances. When unsure, bump.
