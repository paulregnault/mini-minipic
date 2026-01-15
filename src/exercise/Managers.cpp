/* _____________________________________________________________________ */
//! \file Managers.cpp

//! \brief Management of the operators

/* _____________________________________________________________________ */

#include "Managers.hpp"
#include "Operators.hpp"

namespace managers {

void initialize(const Params &params, ElectroMagn &em,
                std::vector<Particles> &particles) {
  // Momentum correction (to respect the leap frog scheme)
  if (params.momentum_correction) {

    std::cout << " > Apply momentum correction "
              << "\n"
              << std::endl;

    operators::interpolate(em, particles);

    operators::push_momentum(particles, -0.5 * params.dt);
  }
}

void iterate(const Params &params, ElectroMagn &em,
             std::vector<Particles> &particles, int it) {
  if (params.current_projection || params.n_particles > 0) {

    DEBUG("  -> start reset current");

    em.reset_currents(minipic::device);

    DEBUG("  -> stop reset current");
  }

  // Interpolate from global field to particles
  DEBUG("  -> start interpolate ");

  operators::interpolate(em, particles);

  DEBUG("  -> stop interpolate");

  // Push all particles
  DEBUG("  -> start push ");

  operators::push(particles, params.dt);

  DEBUG("  -> stop push");

  // Do boundary conditions on global domain
  DEBUG("  -> Patch 0: start pushBC");
  Kokkos::Profiling::pushRegion("pushBC");
  operators::pushBC(params, particles);
  Kokkos::Profiling::pushRegion("pushBC");
  DEBUG("  -> stop pushBC");

#if defined(MINI_MINIPIC_DEBUG)
  // check particles
  for (std::size_t is = 0; is < particles.size(); ++is) {
    particles[is].check(inf_m[0], sup_m[0], inf_m[1], sup_m[1], inf_m[2],
                        sup_m[2]);
  }
#endif

  // Projection in local field
  if (params.current_projection) {

    // Projection directly in the global grid
    DEBUG("  ->  start projection");

    operators::project(params, em, particles);

    DEBUG("  ->  stop projection");
  }

  // __________________________________________________________________
  // Sum all species contribution in the local and global current grids

  if (params.current_projection || params.n_particles > 0) {

    // Perform the boundary conditions for current
    DEBUG("  -> start current BC")
    Kokkos::Profiling::pushRegion("currentBC");
    operators::currentBC(params, em);
    Kokkos::Profiling::pushRegion("currentBC");
    DEBUG("  -> stop current BC")

  } // end if current projection

  // __________________________________________________________________
  // Maxwell solver

  if (params.maxwell_solver) {

    // Generate a laser field with an antenna
    for (std::size_t iantenna = 0; iantenna < params.antenna_profiles_m.size();
         iantenna++) {
      operators::antenna(params, em, params.antenna_profiles_m[iantenna],
                         params.antenna_positions_m[iantenna], it * params.dt);
    }

    // Solve the Maxwell equation
    DEBUG("  -> start solve Maxwell")

    operators::solve_maxwell(params, em);

    DEBUG("  -> stop solve Maxwell")

    // Boundary conditions on EM fields
    DEBUG("  -> start solve BC")
    Kokkos::Profiling::pushRegion("solveBC");
    operators::solveBC(params, em);
    Kokkos::Profiling::pushRegion("solveBC");
    DEBUG("  -> end solve BC")

  } // end test params.maxwell_solver
}

} // namespace managers
