# Nonlinear Wave Solvers

A unified framework for solving nonlinear wave equations in 2D and 3D.

## Supported Equations

### Real-valued Wave Equations

The general form of real-valued wave equations is:

```
u_tt = div(c(x)grad(u)) - m(x)f(u)
```

where:
- `u` is the wave field
- `c(x)` is the anisotropy coefficient
- `m(x)` is the focusing coefficient
- `f(u)` is the nonlinear term

Supported equations:
- Klein-Gordon equation: `f(u) = u`
- Sine-Gordon equation: `f(u) = sin(u)`

### Complex-valued Wave Equations (NLSE)

The general form of nonlinear Schrödinger equations is:

```
i u_t + div(c(x) grad(u)) + m(x) V(|u|²) u = 0
```

where:
- `u` is the complex wave field
- `c(x)` is the anisotropy coefficient
- `m(x)` is the focusing coefficient
- `V(|u|²)` is the nonlinear potential

Supported equations:
- Cubic NLSE: `V(|u|²) = |u|²`

## Building

### Prerequisites

- C++17 compatible compiler
- CMake 3.10 or higher
- Eigen 3.3 or higher
- cnpy library

### Build Instructions

```bash
mkdir build
cd build
cmake ..
make
```

To enable CUDA support:

```bash
cmake -DWITH_CUDA=ON ..
make
```

## Usage

### Real-valued Wave Equations

```bash
./nlwave_bin_real --device host --system-type klein-gordon --dim 3 --anisotropy c_file.npy --focussing m_file.npy --L 10. --n 200 --T 10. --nt 1000 --snapshots 32 --method gautschi --initial-u initial.npy --initial-v initial_velocity.npy --trajectory-file traj.npy --velocity-file vel.npy
```

### Complex-valued Wave Equations (NLSE)

```bash
./nlwave_bin_nlse --device host --system-type nlse-cubic --dim 2 --L 10. --n 200 --T 10. --nt 1000 --snapshots 32 --method strang --initial-u initial.npy --trajectory-file traj.npy
```

## Command-line Arguments

### Common Arguments

- `--device`: Device type (`host` or `cuda`)
- `--dim`: Dimension (2 or 3)
- `--anisotropy`: Anisotropy coefficient file (optional)
- `--focussing`: Focusing coefficient file (optional)
- `--L`: Domain half-length (default: 10.0)
- `--n`: Number of grid points per dimension (default: 200)
- `--T`: Total simulation time (default: 10.0)
- `--nt`: Number of time steps (default: 1000)
- `--snapshots`: Number of snapshots to save (default: 32)
- `--initial-u`: Initial condition file
- `--trajectory-file`: Output trajectory file

### Real-valued Wave Arguments

- `--system-type`: System type (`klein-gordon` or `sine-gordon`)
- `--method`: Integration method (`gautschi` or `stormer-verlet`)
- `--initial-v`: Initial velocity file (optional)
- `--velocity-file`: Output velocity file (optional)

### Complex-valued Wave Arguments

- `--system-type`: System type (`nlse-cubic`)
- `--method`: Integration method (`strang`)

## Method Comparison

To compare different methods and devices, run:

```bash
./compare_methods.py
```

This script will:
1. Generate initial conditions
2. Run simulations with different methods and devices
3. Compare energy conservation
4. Generate plots in the `comparison_results` directory

## Architecture

The code is organized in a hierarchical structure:

```
BaseSolver (abstract)
├── RealWaveSolver (abstract)
│   ├── KleinGordonSolver
│   └── SineGordonSolver
└── ComplexWaveSolver (abstract)
    └── NLSECubicSolver
```

Execution is abstracted through the `ExecutionContext` interface:

```
ExecutionContext (abstract)
├── HostExecutionContext
└── DeviceExecutionContext (planned)
```

This design allows for:
- Code reuse between similar solvers
- Easy addition of new equation types
- Separation of algorithm from execution target
- Unified parameter handling
