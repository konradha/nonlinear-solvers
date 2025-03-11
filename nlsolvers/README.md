#### Refactoring of existing code

##### Design decisions
For now we'll truncate the possible trajectory space to not incorporate anything more than
no-flux boundary conditions which might be restrictive but getting a handle on the phase
space is more important.

Almost all post-processing is going to be done in Python. Python is the glue code that connects
binaries, job management and thus data generation, postprocessing and cleanup. Not only SLURM
jobs but also local integrations should preferably be launched using Python.

TODO

- [x] animation styles: 2d
- [ ] animation styles: 3d
- [x] implementing GPU pipeline for all systems
- [ ] test GPU pipelines
- [ ] porting all systems into 3d mode (CPU)
- [ ] porting all systems into 3d mode (GPU)
- [ ] testing all systems in 3d (including benchmarking)
- [ ] finalize pipelines and write global SLURM scripts
- [ ] complete analysis framework for 2d
- [ ] complete analysis framework for 3d

