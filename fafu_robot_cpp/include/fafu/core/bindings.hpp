#pragma once

#include <pybind11/pybind11.h>

namespace fafu::core {

void bind_core(pybind11::module_& module);

}  // namespace fafu::core
