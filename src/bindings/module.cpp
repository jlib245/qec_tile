#include <nanobind/nanobind.h>
#include <nanobind/stl/vector.h>

#include "core.hpp"

namespace nb = nanobind;
using namespace nb::literals;

NB_MODULE(_core, m) {
    m.doc() = "qec_tile native core (nanobind)";
    m.def("add", &qec_tile::add, "a"_a, "b"_a);
    m.def("parity", &qec_tile::parity, "bits"_a,
          "Parity (XOR-fold) of a 0/1 vector.");
}
