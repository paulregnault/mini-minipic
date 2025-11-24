# Compilation

## Prerequisites

miniPIC uses CMake (version ≥ 3.16) as a build system. For using Kokkos, you have three options (by order of recommendation):

1. Using CMake FetchContent
1. Using a Git submodule
2. Using an installed instance of Kokkos

### Method 1: Using CMake FetchContent

CMake would download an archive of Kokkos and decompress it in `external/kokkos` by default.

```bash
git clone <url of the repo>
```

### Method 2: Using a Git submodule

Git would clone the Kokkos repo in `external/kokkos` and switch to a stable branch.

```bash
git clone --recurse-submodules <url of the repo>
```

If you have cloned without the submodules (i.e. `git clone <url of the repo>`):

```sh
git submodule update --init
```

### Method 4: Using an installed instance of Kokkos

Kokkos would be already installed, either by an administrator or by yourself.

<details>

<summary>Kokkos installation</summary>

```sh
git clone --branch <kokkos version> https://github.com/kokkos/kokkos.git
cd kokkos
cmake -B build -DCMAKE_INSTALL_PREFIX=/path/to/kokkos/install <kokkos extra flags>
cmake --build build --parallel
cmake --install build
```

Please check [the documentation](https://kokkos.org/kokkos-core-wiki/get-started/building-from-source.html) for the Kokkos flags.
Note that you need one build (and one installation directory) per backend and architecture.

</details>

```bash
git clone <url of the repo>
```

Then, you would need to tell CMake where to find Kokkos with `-DKokkos_ROOT=/path/to/kokkos/install`.

## Build

### Basic compilation (sequential mode)

```bash
cmake -B build
cmake --build build
```

<img title="Warning" alt="Warning" src="./images/warning.png" height="20"> Building in the root directory is not supported.

<img title="Warning" alt="Warning" src="./images/warning.png" height="20"> By default, the code is compiled with Kokkos serial backend (sequential mode).

Note that you have to add `-DKokkos_ROOT=</path/to/kokkos/install>` if you use an already installed instance of Kokkos.

### Options

CMake generic options:

- `-DCMAKE_CXX_COMPILER=<compiler choice>`: specify the compiler to use;
- `-DCMAKE_BUILD_TYPE=<build type>`: specify the build (most commons are `Debug` and `Release`).

Project specific options:

- `-DMINIPIC_DEBUG=<ON/OFF>`: enable/disable debug messages (`OFF` by default);
- `-DMINIPIC_WARNING=<ON/OFF>`: enable/disable compiler warnings (add `-Wall`, `-Wextra`, and `-Wpedantic`, `OFF` by default);
- `-DMINIPIC_UNIFIED_MEMORY=<ON/OFF>`: enable/disable unified memory views (`OFF` by default);
- `-DMINIPIC_IMPLEMENTATION=<implementation>`: which implementation to use (`exercise` or `kokkos`, default to the former);
- `-DMINIPIC_SETUP=<setup>`: which setup to build and run with (`antenna`, `b_cst`, `beam`, `e_cst`, `thermal`, default to the former).

## Examples

### Using a Git submodule

When using the submodule you can pass directly Kokkos options. See the [Kokkos CMake options documentation](https://kokkos.org/kokkos-core-wiki/get-started/configuration-guide.html).

#### Basic compilation (defaults to serial backend)

```bash
cmake -B build
cmake --build build
```

#### OpenMP compilation using GCC and OpenMP

```bash
cmake -B build -DCMAKE_CXX_COMPILER=g++ -DKokkos_ENABLE_OPENMP=ON
cmake --build build
```

#### Kokkos compilation using Clang for CPU

```bash
cmake -B build -DCMAKE_CXX_COMPILER=clang++ -DKokkos_ENABLE_SERIAL=ON
cmake --build build
```

#### Kokkos compilation using `nvcc` for Nvidia V100

```bash
cmake -B build -DKokkos_ENABLE_CUDA=ON -DKokkos_ARCH_VOLTA70=ON
cmake --build build
```

### Using and installed Kokkos

When using a pre-installed Kokkos, you need to specify its location and the configuration should already be set in the compiled and installed library:

- Using Kokkos installed in a custom location

```bash
cmake -B build -DKokkos_ROOT=/path/to/kokkos/install
cmake --build build
```
