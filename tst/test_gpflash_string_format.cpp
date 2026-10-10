// StringFormat.hpp's printf wrapper. What this file asserts:
//
//   1) formatted_string_is_exactly_the_text  the returned string is the formatted text and nothing else, the
//                                            terminating null vsnprintf writes dropped again
#include <cstdio>
#include <string>

#include "GpFlashTestSupport.hpp"
#include "gpflash/core/StringFormat.hpp"

using gpflash_test::kDetailLen;
using gpflash_test::report;

namespace {

// The length is the assertion: a null left at the end survives every substring check the report tests make, but
// fputs stops at it, so a printed report would end at its first fragment.
bool test_formatted_string_is_exactly_the_text(const char* name) {
    const int n_long_char = 4096;  // past any buffer a fixed-size formatter would use
    const std::string short_str = gpflash::detail::to_formatted_string("%s %d", "n", 7);
    const std::string long_str = gpflash::detail::to_formatted_string("%s", std::string(n_long_char, 'n').c_str());
    const std::string empty_str = gpflash::detail::to_formatted_string("%s", "");

    const bool is_short_exact = short_str == "n 7";
    const bool is_long_exact = long_str == std::string(n_long_char, 'n');
    const bool is_empty_exact = empty_str.empty();

    char detail[kDetailLen];
    std::snprintf(
            detail,
            sizeof(detail),
            "short_len=%d long_len=%d empty_len=%d",
            static_cast<int>(short_str.size()),
            static_cast<int>(long_str.size()),
            static_cast<int>(empty_str.size()));
    return report(name, is_short_exact && is_long_exact && is_empty_exact, detail);
}

}  // namespace

int main() {
    std::printf("== GpFlash string format ==\n");
    bool ok = true;
    ok &= test_formatted_string_is_exactly_the_text("formatted_string_is_exactly_the_text");
    std::printf("\n%s\n", ok ? "ALL TESTS PASSED" : "SOME TESTS FAILED");
    return ok ? 0 : 1;
}
