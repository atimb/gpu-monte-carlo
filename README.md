# GPU Monte Carlo Scintillator Simulation

The goal of this project is to simulate the energy spectrum observed with a scintillator (http://en.wikipedia.org/wiki/Scintillator) 
placed next to a point gamma source (http://en.wikipedia.org/wiki/Gamma_ray) by Monte Carlo simulation (By following the path and 
interactions of a large number of individual photons).
The GPU enables the simulator to run on average 10x faster than on a CPU.

## How it works

The project builds upon the nVidia CUDA API.
The simulation consists of a point gamma source and a box shaped scintillator. Most physical properties, like positioning, cross sections, 
as well as simulation parameters are configurable via command line or the `input-data.txt` configuration file.
The simulation then runs on the GPU (and optionally on CPU with fewer number of photons, to be able to verify the GPU results).

## Structure

The three source files include the `scintillator.cu` which defines the main entry point, and `scintillator_gold.cpp` which runs the simulation
on CPU, and `scintillator_kernel.cu` which runs the simulation on GPU.
You can either build the source with the included `Makefile` or import the project in Visual Studio with the included `sln` project file.

## Docs

This was actually my thesis, and you can find the (unfortunately only hungarian) paper under `docs`.

## License

(C) 2009 Attila Incze <attila.incze@gmail.com>

http://atimb.me

This work is licensed under the Creative Commons Attribution-NonCommercial-ShareAlike 3.0 Unported License. To view a copy of this license, visit
http://creativecommons.org/licenses/by-nc-sa/3.0/ or send a letter to Creative Commons, 444 Castro Street, Suite 900, Mountain View, California, 94041, USA.