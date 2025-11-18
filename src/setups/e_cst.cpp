/* _____________________________________________________________________ */
//! \file default.hpp

//! \brief This is the default setup used to initialize MiniPic
//! CASE: homogeneous uniform thermalized plasma with periodic boundary conditions

/* _____________________________________________________________________ */

#include "Setup.hpp"

//! \brief Functiun to setup all input parameters
void setup(Params &params) {

  // Simulation name
  params.name = "Constant electric field";

  // Space
  params.inf_x = 0.;
  params.inf_y = 0.;
  params.inf_z = 0.;
  params.sup_x = 1.;
  params.sup_y = 1.;
  params.sup_z = 1.;

  // Cells per patch per direction
  params.nx_cells = 32;
  params.ny_cells = 32;
  params.nz_cells = 32;

  // Time

  // const double dx = (params.sup_x - params.inf_x) / (params.nx_cells);
  // const double dy = (params.sup_y - params.inf_y) / (params.ny_cells);
  // const double dz = (params.sup_z - params.inf_z) / (params.nz_cells);

  params.dt = 0.9; // Fraction of the CFL

  params.simulation_time = 50000 * params.dt;

  // Species

  // custom density profile
  auto profile = [](double x, double y, double z) -> double { return 1e-5; };

  // name, mass, charge, density, temperature, density profile, drift velocity, particles per cell,
  // position initialization
  params.add_species("electron", 1, -1, 1e-2, profile, {0, 0, 0}, 0, "random", "cell");

  // Ad a single particle
  params.add_particle(0, 1e-3, 0.1, 0.1, 0.1, 0., 0., 0.);

  // Initiale electric field values
  params.initialize_electric_field(-0.05, -0.01, -0.08);

  // Bourndary conditions
  params.boundary_condition = "periodic";

  // Display
  params.print_period = 10000;

  // Random seed
  params.seed = 0;

  // Field Diagnostics
  params.particle_cloud_period = 1000;

  // Scalar Diagnostics
  params.scalar_diagnostics_period = 100;

  // Field Diagnostics
  params.field_diagnostics_period = 0;

  // Momentum correction at init
  params.momentum_correction = true;

  // Current projection
  params.current_projection = false;

  // Maxwell solver
  params.maxwell_solver = false;
}
