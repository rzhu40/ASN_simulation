# ASN_simulation
 Efficiency improved, multi-electodes enabled code for Zdenka's group.

# To Start
 * If one hopes to use single pair of source and drain, just imporve the efficiency of Ido's code, please go here:
 https://github.com/rzhu40/ASN_simulation/blob/master/Improve%20Old%20Code/README.md.
 * MatLab user guide: https://github.com/rzhu40/ASN_simulation/blob/master/MatLab/README.md.
 * Python user guide:

# TO-DOs
 * Enable network visualization in MatLab code(Left the snapshot API usable from Ido's code).
 * Write new network generator in MatLab and Python.
 * Enable time-dependent plotting in python code (with a slider for time points).
 * Enable movie compiling in python.
 * Re-pacakage Ido's code to make a lite version with multi-electrodes enabled.
 * At some stage, figure out a way to enable other circuit components like capacitors and inductors.

# DO notice
 * The improvement of simulation is done based on Nodal analysis. https://www.swarthmore.edu/NatSci/echeeve1/Ref/mna/MNA3.html
 * Still need to test more for multi-electrodes simulations. Don't hesitate to contact me if there are any bugs or requirements.
 * The output dynamics are recording things slightly different from Ido's code (message me or feel free to record what ever data one needs from simulation).
 * Most of my codes are written in python first then MatLab. Please CONSIDER python :). 
 
   I will upload a bench mark for comparison between python and MatLab. Most importantly, python code is plotting with plot.ly https://plot.ly/python/, which makes beautiful and **Interactive** snapshots of the network.
 
