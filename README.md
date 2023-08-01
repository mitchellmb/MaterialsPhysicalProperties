# MaterialsPhysicalProperties

Modules for fitting common temperature-dependent physical properties of condensed matter materials. 
Current iteration includes: Curie Weiss model for magnetization and Debye model for specific heat lattice vibrations. 

To use:
1. import PhysicalProperties.py as a module
2. load and preprocess magnetization or specific heat data (e.g., np.genfromtxt() )
3. create CurieWeiss() or HeatCapacityDebye() instances with initial guesses for parameters
4. call .fit(), parameters are stored in class attributes .fitparams and .fitcov
5. plot with .plotter()
