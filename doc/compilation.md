# Compilation

## Prerequisites

(mini)miniPIC uses CMake (version ≥ 3.22) as a build system. For using Kokkos, you have three options (by order of recommendation):

1. Using CMake FetchContent
2. Using a Git submodule
3. Using an installed instance of Kokkos

### Method 1: Using CMake FetchContent

CMake would download an archive of Kokkos and decompress it in `external/kokkos` by default.

```bash
git clone https://github.com/CExA-project/mini-minipic.git
```

### Method 2: Using a Git submodule

Git would clone the Kokkos repo in `external/kokkos` and switch to a stable branch.

```bash
git clone --recurse-submodules https://github.com/CExA-project/mini-minipic.git
```

If you have cloned without the submodules (i.e. `git clone <url of the repo>`):

```sh
git submodule update --init
```

### Method 3: Using an installed instance of Kokkos

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
git clone https://github.com/CExA-project/mini-minipic.git
```

Then, you would need to tell CMake where to find Kokkos with `-DKokkos_ROOT=/path/to/kokkos/install`.

## Build

### Basic compilation

Without any option provided, the sequential backend should be used.

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

- `-DMINI_MINIPIC_DEBUG=<ON/OFF>`: enable/disable debug messages (`OFF` by default);
- `-DMINI_MINIPIC_WARNING=<ON/OFF>`: enable/disable compiler warnings (add `-Wall`, `-Wextra`, and `-Wpedantic`, `OFF` by default);
- `-DMINI_MINIPIC_KOKKOS_SCATTER_VIEW=<ON/OFF>`: use Kokkos scatter views for projection operator (`OFF` by default);
- `-DMINI_MINIPIC_IMPLEMENTATION=<implementation>`: which implementation to use (`exercise`,`kokkos`, or any directory with valid files, default to the former);
- `-DMINI_MINIPIC_SETUP=<setup>`: which setup to build and run with (`antenna`, `b_cst`, `beam`, `e_cst`, `thermal`, default to the former);
- `-DMINI_MINIPIC_KOKKOS_SOURCE_DIRECTORY=</path/to/kokkos/sources>`: Path to the local source directory of Kokkos (default to `./external/kokkos`).

## Examples

### Using FetchContent or a Git submodule

When getting Kokkos with CMake FetchContent or with a Git submodule, you should pass Kokkos options directly.
See the [Kokkos CMake options documentation](https://kokkos.org/kokkos-core-wiki/get-started/configuration-guide.html).

#### Basic compilation (defaults to serial backend)

```bash
cmake -B build
cmake --build build --parallel 10
```

#### OpenMP compilation using GCC and OpenMP

```bash
cmake -B build -DCMAKE_CXX_COMPILER=g++ -DKokkos_ENABLE_OPENMP=ON
cmake --build build --parallel 10
```

#### Kokkos compilation using Clang for CPU

```bash
cmake -B build -DCMAKE_CXX_COMPILER=clang++
cmake --build build --parallel 10
```

#### Kokkos compilation using Cuda for Nvidia A100

```bash
cmake -B build -DKokkos_ENABLE_CUDA=ON -DKokkos_ARCH_AMPERE80=ON
cmake --build build --parallel 10
```

### Using and installed instance of Kokkos

When using an already installed instance of Kokkos, you should specify its location to CMake at configuration time.
Kokkos options would be automatically transferred.

```bash
cmake -B build -DKokkos_ROOT=/path/to/kokkos/install
cmake --build build --parallel 10
```
