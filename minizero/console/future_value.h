#pragma once

namespace minizero::console {

// How much does a position's value depend on the hidden future? For positions
// replayed from game records, draw K hidden futures (upcoming pieces beyond the
// preview and future garbage), run one search per future that sees that future,
// and write the root values. See future_value.cpp for the parameters.
class FutureValue {
public:
    void run();
};

} // namespace minizero::console
