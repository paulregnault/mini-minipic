/* _____________________________________________________________________ */
//! \file Operators.hpp

//! \brief contains generic kernels for the particle pusher

/* _____________________________________________________________________ */

#pragma once

#include "ElectroMagn.hpp"
#include "Headers.hpp"
#include "Particles.hpp"

namespace operators {

double sum_host(typename Particles::hostview_t view);

double sum_device(typename Particles::view_t view);

double sum_power(ElectroMagn::view_t v, const int power);

double sum_power(ElectroMagn::hostview_t v, const int power);

void interpolate(ElectroMagn &em, std::vector<Particles> &particles);

void push(std::vector<Particles> &particles, double dt);

void push_momentum(std::vector<Particles> &particles, double dt);

void pushBC(const Params &params, std::vector<Particles> &particles);

void project(const Params &params, ElectroMagn &em, std::vector<Particles> &particles);

void solve_maxwell(const Params &params, ElectroMagn &em);

void currentBC(const Params &params, ElectroMagn &em);

void solveBC(const Params &params, ElectroMagn &em);

void antenna(const Params &params,
             ElectroMagn &em,
             std::function<double(double, double, double)> profile,
             double x,
             double t);

} // namespace operators
