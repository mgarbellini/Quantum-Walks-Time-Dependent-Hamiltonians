# Quantum Walks with Time-Dependent Hamiltonian

The repository is structured in the following way

  0. **report**. Contains a latex and pdf report of the work done so far. Constantly updated and not yet up to current date.
  Quick link to pdf (https://github.com/matteogarbellini/Adiabatic-Quantum-Walk/blob/AdiabaticQuantumWalk/0.%20report/thesis_report.pdf)
  1. **non-adiabatic-implementation**. Search with the usual Grover's QW algorithm. Provides benchmark for the adiabatic-QW implementation
  2. **adiabatic-implementation**. Search with the adiabatic-QW implementation. Contains an N-dimensional schroedinger solver, and a routine to check if the adiabati c theorem is satisfied.
  3. **eigenvalues-crossing**. Contains a simply routine to check the no-crossing needed for the adiabati theorem
  4. **RK-error**. Standalone study on the normalization error due to the Runge-Kutta integrator. Contains a python routine and pdf report of the results
  5. **probability-heatmap**. Routine for the probability heatmap needed in order to save computational time. Lots of results (.npy and .pdf), including heatmap plots.
  6. **comparing-results**. Comparison of the static (std QW) and dynamic (Adiab QW) search implementation. Combines the results of the original benchmark (Sec 1) and results from the probability heatmaps (Sec 5)
  7. **ignoring-adiabatic-theorem**. Contains some results (heatmap plots and data) up to N=35 without the adiabatic time constrain**
  8. **step-function**. Some test with different (time) step functions. Will be later deleted, as it's already implemented in the other routines and results are redundant
  9. **complete-graph**. Probability heatmap routine for the complete graph. Includes plots and numerical results (for different step functions) and some initial test following the step function s(t) proposed by Roland and Cerf (*Quantum Search by Local Adiabatic Evolution*, quant-ph/0107015)
  10. **plots**. Some 'somewhat' organized plots and useful gnuplot snippets
  11. **meeting-results**. Some 'somewhat' organized results for the weekly meetings
