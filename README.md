Scripts and data for studying scaling of PyClaw on KAUST's Cray XC40

To investigate:
- Dynamic loading
- Separate costs of various parallel parts
- Weak scaling
  - 2D/3D
  - fixed # of steps/adaptive stepping
  - Classic/SharpClaw solvers (and influence of larger WENO stencils)
- Strong scaling
  - All the same as above
- Single-node efficiency?
