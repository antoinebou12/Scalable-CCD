#pragma once

#include <vector>
#include <scalable_ccd/stq/cpu/aabb.hpp>

namespace stq::cpu {

void parseMesh(
    const char* filet0, const char* filet1, std::vector<Aabb>& boxes);

} // namespace stq::cpu