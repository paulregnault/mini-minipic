# Python tools (`libminipic`)

MiniPIC uses several tools gathered in the `libminipic` library.
These tools are used for full execution, validation, printing, verification, and plotting.

Two commands and several scripts are available.

## Python loading

On a supercomputer, you may have to load a specific version of Python:

```sh
module load python<x.y>
```

with `<x.y>` the version of Python.
The `libminipic` requires Python ≥ 3.11.

You can list the available versions with:

```sh
module avail python
```

## Installation

Install the Python tools with:

```sh
pip install --user .
```

<details>

<summary>
Even better, you can use a virtual environment.
</summary>

```sh
pip install --user virtualenv
virtualenv --python python<x.y> $PWD/venv
source venv/activate
pip install .
```

with `<x.y>` the version of Python.
By instance, `python3.12`.
Note you have to source the activation script if you are in a new terminal (and in a Slurm job).

</details>

## `mini-run` command

The `mini-run` command is used to build miniPIC for a selection of setups, to execute it and to validate its results.
For validation, `mini-run` calls the same functions as `mini-validate` internally.

You can get some help with:

```bash
mini-run -h
```

### Available options

Here is a list of the available options:

| Option | Long Option | Description |
| --- | --- | --- |
| `-h` | `--help` | Show a help message and exit |
| `-g CONFIG` | `--config CONFIG` | Configuration choice: cpu (default), gpu |
| `-c COMPILER` | `--compiler COMPILER` | Custom compiler choice |
| `-s SETUP ` | `--setups SETUPS` | Specific setup, you can specify several setups with a coma. For instance "default,beam" |
| | `--build-dir` | Build directory to use, default to `build` |
| | `--implementation` | Which implementation to use, default to `exercise` |
| `-a ARGUMENTS` | `--arguments ARGUMENTS` | Default arguments |
| | `--fresh` | Whether to delete or not already existing files (clean before build) |
| | `--clean` | Whether to delete or not the generated files (clean after run) |
| | `--no-evaluate` | If used, do not evaluate against the reference |
| | `--compile-only` | If used, only compile the tests |
| | `--threshold THRESHOLD` | Threshold for the validation |
| | `--save-timers` | Save the timers for each setup |
| | `--env` | Custom environment variables for the execution |
| | `--cmake-args` | Set CMake arguments |
| | `--cmake-args-add` | Append custom CMake arguments |

### Configurations

Here is a list of all configurations:

| Configuration | Description |
| --- | --- |
| cpu-serial | CPU serial |
| cpu-openmp | CPU with OpenMP, 8 threads and `OMP_PROC_BIND` set to `spread` |
| gpu-v100 | GPU on V100 |
| gpu-a100 | GPU on A100 |
| gpu-h100 | GPU on H100 |
| gpu-mi250 | GPU on MI250 |
| gpu-mi300a | GPU on MI300A, with `HSA_XNACK` set |

### Usage examples

#### Default run

```bash
mini-run
```

#### Specific configuration

```bash
mini-run -g gpu
```

#### Custom compiler

```bash
mini-run -c clang++
```

## `mini-validate` command

A validation mechanism can be used to validate simulation results.
Output files are checked against reference values.

You can access the help page by doing:

```bash
mini-validate -h
```

### Available options

Here is a list of the available options:

| Option | Long Option | Description |
| --- | --- | --- |
| `-s` | `--setup` | Name of the setup (autodetected by default) |
| `-p` | `--path` | Path of the execution directory (default to curent directory) |
| | `--threshold THRESHOLD` | Threshold for the validation |

### Setups

The possible setups are:

- `antenna`
- `b_cst`
- `beam`
- `e_cst`
- `thermal`

Validation take place if, for a given setup, a Python file of the same name is present in the directory `libminipic/validation`.
For instance, the `thermal` setup (which corresponds to the `src/setups/thermal.cpp`) is checked using the `libminipic/validation/thermal.py` file.

## Scripts directory

The `scripts` directory contains programs for plotting, printing, and verifying simulation outputs.
The complete list is:

- `plot_energy_balance.py`;
- `plot_field.py`;
- `plot_particle_binning.py`;
- `plot_particle_cloud.py`;
- `print_all_fields.py`;
- `print_momentum_intime.py`;
- `print_particle_intime.py`;
- `print_particles.py`;
- `verif_Bfield_cst.py`;
- `verif_EB_cst.py`;
- `verif_Efield_cst.py`;
- `verif_kinetic_energy.py`.

Please refer to each script for its use.
