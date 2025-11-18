/* _____________________________________________________________________ */
//! \file Managers.hpp

//! \brief Management of the operators

/* _____________________________________________________________________ */

#pragma once

#include <vector>

#include "ElectroMagn.hpp"
#include "Particles.hpp"
#include "Params.hpp"

namespace managers {

void initialize(const Params &params, ElectroMagn &em, std::vector<Particles> &particles);

void iterate(const Params &params, ElectroMagn &em, std::vector<Particles> &particles, int it);

} // namespace managers
