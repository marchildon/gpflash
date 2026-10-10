/**
 * ParallelContext.hpp -- the processes a computation is spread over, and the collectives over them.
 *
 * The only file in the library that names MPI. It holds two definitions of the class, one per build, rather than
 * a macro inside each method. Every method the library calls has to exist in both, and the serial one is the
 * default build, so a method added to one and not the other fails to compile.
 */
#pragma once

#include <armadillo>  // arma::mat, arma::u64
#include <string>     // the broadcast message

#include "gpflash/core/Platform.hpp"  // GPFLASH_HAVE_MPI

#if GPFLASH_HAVE_MPI
#include <mpi.h>    // the program owns MPI_Init_thread
#include <cstddef>  // std::size_t, for a broadcast message length
#include <limits>   // std::numeric_limits, for MPI's element count

#include "gpflash/core/Errors.hpp"  // ErrorInvalidInput

namespace gpflash {

// The processes sharing one computation, built once by whoever starts it and passed down.
//
// A context is serial when the program has not initialized MPI or has already finalized it, and when its
// communicator holds one process. Every collective is then a no-op, so no caller branches on which it has.
class ParallelContext {
   public:
    // Every process the program has.
    ParallelContext() { adopt_comm(MPI_COMM_WORLD); }

    // The processes of comm. MPI_COMM_SELF asks for a serial context, each process carrying the whole
    // computation, whatever the program launched.
    explicit ParallelContext(const MPI_Comm comm) { adopt_comm(comm); }

    int n_process() const { return n_process_; }
    int i_process() const { return i_process_; }

    // ── The collectives ──────────────────────────────────────────────────────

    // The lowest, and the highest, value any process holds.
    int calc_min(const int val) const { return calc_reduce(val, ReduceKind::Min); }
    int calc_max(const int val) const { return calc_reduce(val, ReduceKind::Max); }

    // Sum an array across the processes, in place on each.
    void calc_sum_inplace(arma::mat& mat_inout, const char* field_name) const {
        if (!is_collective()) return;
        if (mat_inout.n_elem > static_cast<std::size_t>(std::numeric_limits<int>::max()))
            throw ErrorInvalidInput(
                    std::string("ParallelContext::calc_sum_inplace: summing ") + field_name + " needs "
                    + std::to_string(mat_inout.n_elem) + " elements, more than MPI's int count can carry.");

        MPI_Allreduce(MPI_IN_PLACE, mat_inout.memptr(), static_cast<int>(mat_inout.n_elem), MPI_DOUBLE, MPI_SUM, comm_);
    }

    // Every process holds the same value, or all of them throw. One fixed-width element, so it is valid to call
    // even when the processes disagree about the sizes of the collectives that follow.
    void check_same_on_every_process(const arma::u64 local_val, const char* field_name, const char* caller) const {
        if (!is_collective()) return;

        arma::u64 lowest_val = local_val;
        arma::u64 highest_val = local_val;
        MPI_Allreduce(MPI_IN_PLACE, &lowest_val, 1, MPI_UINT64_T, MPI_MIN, comm_);
        MPI_Allreduce(MPI_IN_PLACE, &highest_val, 1, MPI_UINT64_T, MPI_MAX, comm_);
        if (lowest_val == highest_val) return;

        throw ErrorInvalidInput(
                std::string(caller) + ": " + field_name + " differs across the processes of this ParallelContext; "
                + "process " + std::to_string(i_process_) + " holds " + std::to_string(local_val)
                + ". Every process must be given the same inputs.");
    }

    // Give every process the string i_root holds.
    void broadcast_inplace(std::string& str_inout, const int i_root) const {
        if (!is_collective()) return;

        // The length first, so a receiver can size its buffer.
        int n_char = (i_process_ == i_root) ? static_cast<int>(str_inout.size()) : 0;
        MPI_Bcast(&n_char, 1, MPI_INT, i_root, comm_);
        str_inout.resize(static_cast<std::size_t>(n_char));
        if (n_char > 0) MPI_Bcast(&str_inout[0], n_char, MPI_CHAR, i_root, comm_);
    }

   private:
    MPI_Comm comm_ = MPI_COMM_SELF;  // what adopt_comm accepted; left here, and unread, on a serial context
    int n_process_ = 1;              // 1 until adopt_comm finds a live MPI runtime
    int i_process_ = 0;              // this process's number within comm_

    enum class ReduceKind { Min, Max };

    // MPI_Initialized is callable before MPI_Init, so a consumer that links MPI without initializing it gets a
    // serial context rather than an abort. It stays true after MPI_Finalize, hence the second question.
    void adopt_comm(const MPI_Comm comm) {
        int is_mpi_initialized = 0;
        int is_mpi_finalized = 0;
        MPI_Initialized(&is_mpi_initialized);
        MPI_Finalized(&is_mpi_finalized);
        if (!is_mpi_initialized || is_mpi_finalized) return;

        comm_ = comm;
        MPI_Comm_size(comm_, &n_process_);
        MPI_Comm_rank(comm_, &i_process_);
    }

    bool is_collective() const { return n_process_ > 1; }

    int calc_reduce(const int val, const ReduceKind reduce_kind) const {
        if (!is_collective()) return val;

        int reduced_val = val;
        MPI_Allreduce(
                MPI_IN_PLACE, &reduced_val, 1, MPI_INT, (reduce_kind == ReduceKind::Min) ? MPI_MIN : MPI_MAX, comm_);

        return reduced_val;
    }
};

}  // namespace gpflash
#else

namespace gpflash {

// One process, the build having no MPI, so every collective leaves its argument as it found it.
class ParallelContext {
   public:
    int n_process() const { return 1; }
    int i_process() const { return 0; }

    int calc_min(const int val) const { return val; }
    int calc_max(const int val) const { return val; }
    void calc_sum_inplace(arma::mat&, const char*) const {}
    void check_same_on_every_process(const arma::u64, const char*, const char*) const {}
    void broadcast_inplace(std::string&, const int) const {}
};

}  // namespace gpflash
#endif
