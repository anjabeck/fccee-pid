# fccee-pid
This repository contains the particle identification tool developed in [arxiv:2511.17447](https://doi.org/10.48550/arXiv.2511.17447).

## Notes on training data
* Particles where either the Silicone dE/dx or timing information where NaN were removed from training.
* In cases where there are hits in Silicone but no hits in the drift chamber, the dN/dx value in the training data was set to $-1$. This also includes cases where the momentum of a particle is below $\beta\gamma<1$ as the extrapolation as implemented in the Delphes framework does not reach further.

## Input and output data
The code can deal with the standard ROOT file format where each row represents an array of arbitrary length. The output file retains this structure so that it can be concatenated or added as a friend to the original file without having to modify the input file. The output contains four branches, one for the proton/kaon/pion probability and another one for the prediction corresponding to the hypothesis with the highest probability. In order to verify that the variables have the correct units, compare them to Figures 1 or 2 in the publication.

## Arguments
`--input` (str): Name of the input file (this won't be modified).

`--output` (str): Name of the output file (this contains the PID information).

`--treename` (str): Name of the ROOT tree in the input file.

`--detector` (str, defaults to IDEA): Name of the detector concept used to train the classifiers. The current options are CLD and IDEA (which is identical to ALLEGRO for PID purposes at the time of training).

`--dndx_val` (float, defaults to 0.8): Efficiency of counting clusters in the drift-chamber. The options are 0.5, 0.8, and 1.0.

`--tof_val` (float, defaults to 30): Resolution of the time of flight in picoseconds. The options are 0, 1, 3, 5, 10, 30, 50, 100, 300, and 500. If speed should not be used in the classification, set this to -1.

`--tof_var` (str, defaults to None): Variable containing the time of flight in nanoseconds.

`--flight_var` (str, defaults to None): Variable containing the flight distance in meters.

`--speed_var` (str, defaults to None): Variable containing the speed in units of $c$ (cannot be given in addition to flight_var and tof_var).

`--dndx_var` (str, defaults to None): Variable containing the dN/dx information from the drift chamber in units of 1/mm.

`--dedx_var` (str, defaults to None): Variable containing the dE/dx information from the Silicon trackers in units of MeV/cm.

`--momentum_var` (str, defaults to momentum): Variable containing the momentum in GeV/c.

Note that the variables that are provided will be used. This means for example that if dE/dx should not be used in the classification, the variable `dedx_var` should not be set but remain `None`.
For example, the PID information for IDEA, using only dN/dx with a cluster-counting efficiency of 50% where the dN/dx variable is called dndx and the momentum is called p, could be obtained by running
```bash
python get_pid.py --input myinput.root --output myoutput.root --treename data --detector IDEA --dndx_val 0.5 --dndx_var dndx --momentum_var p
```

## Environment
This repository requires `awkward`, `uproot`, and `xgboost`. In principle, these packages are available in the edm4hep cvmfs build. However the classifiers used here were produced using `xgboost` 3 and cannot be used with the lower version 2 included in the most recent build. This may change in future releases.
This repository provides a minimal custom `conda` environment that allows to run the script. It can be built with
```bash
conda env create -f environment.yml`
```
and activated using
```
conda activate pid_package
```