# Improve Ido's Code
  * TO start with, one should have Ido's ASN simulation code.
  * Download this piece of simulateNetwork.m, go to ~/asn/simulator.
  * Replace simulateNetwork.m with the one here.
  * When calling the function for simulation, remember that the input of simulation requires Connectivity instead of Equations:
  ```matlab
  [Output, SimulationOptions, snapshots] = simulateNetwork(Connectivity, Components, Signals, SimulationOptions);
  ```
