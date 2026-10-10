/**
 * EigMaxUbMethod.hpp -- how the upper bound on lambda_max of a unit-diagonal M_precon is taken, which sizes its
 * nugget eta.
 *
 * Append, never insert: the posterior cache stores the method as its integer.
 */
#pragma once

#include "gpflash/core/Errors.hpp"  // ErrorInvalidInput

namespace gpflash {

// End is one past the last method, for check_enum_val.
enum class EigMaxUbMethod {
    Gershgorin,  // 1 + the largest off-diagonal absolute row sum: reads the matrix, and moves with it
    Trace,       // n_row: free, constant, and never tighter than Gershgorin's
    End
};

// The method's name, for summaries.
inline const char* to_eig_max_ub_method_name(const EigMaxUbMethod eig_max_ub_method) {
    switch (eig_max_ub_method) {
        case EigMaxUbMethod::Gershgorin:
            return "Gershgorin";
        case EigMaxUbMethod::Trace:
            return "trace";
        case EigMaxUbMethod::End:
            break;
    }
    // Only End, or a value cast in from outside the enum, which check_model_options and
    // check_posterior_model_info reject first.
    throw ErrorInvalidInput("to_eig_max_ub_method_name: unknown EigMaxUbMethod.");
}

}  // namespace gpflash
