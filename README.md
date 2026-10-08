================================================================================
 Space-time energy/entropy-stable method for nonlinear wave systems
================================================================================

Code accompanying the paper

  M. Dumbser, I. Perugia and E. Zampa,
  "An arbitrarily high-order and locally energy/entropy-stable space-time method for a class of nonlinear wave systems".

Author of the code: Enrico Zampa


--------------------------------------------------------------------------------
 1. What this code does
--------------------------------------------------------------------------------

The script StructurePreservingDG.py solves, in two space dimensions, the nonlinear wave system

    d_t rho + div v   = 0
    d_t v   + grad s  = 0,        s = E'(rho) = rho^gamma / gamma,

with total energy

    E = 1/2 |v|^2 + rho^(gamma+1) / (gamma (gamma+1)).

The scheme is an arbitrary-order space-time method:

  * space : discontinuous (DG-type) discretization, v in a broken H(div) space
            and rho in L2, both of polynomial degree k (command-line option --k);
  * time  : Legendre basis of degree m on each time step, integrated with
            10-point Gauss-Legendre quadrature (command-line option --m);
  * each time step leads to a nonlinear space-time system that is solved with
    Newton's method; the linear systems are solved with block-Jacobi
    preconditioned GMRES (one block per element).

Besides the solution, the script monitors the global energy, the L2 norm of
curl(v), and the element-wise (local) energy conservation error.


--------------------------------------------------------------------------------
 2. Requirements
--------------------------------------------------------------------------------

  * Python 3
  * NGSolve (including Netgen with OpenCascade/OCC support)
        pip install ngsolve
  * NumPy

Note: the adaptive epsilon computation (compute_epsilon in StructurePreservingDG.py) uses a
PARDISO inverse. If your NGSolve build does not include PARDISO, replace
inverse = "pardiso" in that function with another sparse direct solver, for
example "umfpack".


--------------------------------------------------------------------------------
 3. Running the code
--------------------------------------------------------------------------------

    python StructurePreservingDG.py [--ictype I] [--NMAX N] [--k K] [--m M] [--Ctime C]

All arguments are optional:

  --ictype   test case, see Section 4                           (default: 2)
  --NMAX     mesh parameter; the maximum mesh size is
             hmax = scaleFactor / NMAX, where scaleFactor is the
             length scale of the domain of the test case         (default: 40)
  --k        polynomial degree in space                          (default: 1)
  --m        polynomial degree in time                           (default: 0)
             (m + 1 basis functions in time are used per step)
  --Ctime    time step factor, dt = Ctime * hmax                 (default: 0.1)
             (test case 5 overrides this value internally)

Examples:

    # Riemann problem on a thin strip, degree 1 in space, degree 0 in time
    python StructurePreservingDG.py --ictype 2 --NMAX 40 --k 1 --m 0

    # Smooth linear test with exact solution, degree 2 in space and time
    python StructurePreservingDG.py --ictype 4 --NMAX 40 --k 2 --m 2

    # Simple convergence study for test case 5 (errors are appended to a file)
    for N in 10 20 40 80; do python StructurePreservingDG.py --ictype 5 --NMAX $N --k 2 --m 2; done

Other settings (tolerances, number of quadrature points, bonus integration
order, switching iterative/direct solvers via use_gmres, adaptive epsilon, ...)
are defined at the top of StructurePreservingDG.py.


--------------------------------------------------------------------------------
 4. Test cases (--ictype)
--------------------------------------------------------------------------------

  ictype | name                | domain               | gamma | final time | boundaries
  -------+---------------------+----------------------+-------+------------+-----------
    2    | RP2d                | [0,1] x [0,0.025]    |  2    |  0.1       | periodic
    3    | SmoothNonlinear2d   | [-0.5,0.5]^2         |  1.4  |  0.25      | periodic
    4    | SmoothLinear2d      | [0,1]^2              |  1    |  0.4       | periodic
    5    | SmoothNonlinearEx2D | [0,2*pi]^2           |  3    |  0.5       | periodic
    6    | RP2d_BC             | [0.5,1] x [0,0.025]  |  2    |  0.1       | periodic in y
    7    | Explosion           | [-1,1]^2             |  2    |  0.3       | periodic

  2, 6   Riemann problem (rho = 2 / 0.1, v = 0), quasi-1D on a thin strip.
         Case 6 uses non-periodic boundary conditions in x.
  3      Smooth nonlinear case: Gaussian bump on top of a constant density.
  4      Linear case with known exact solution (L2 errors are computed).
  5      Nonlinear case with known exact solution, obtained by solving the
         characteristic equation with Newton iterations (L2 errors computed).
  7      Radial Riemann problem (explosion).


--------------------------------------------------------------------------------
 5. Output files
--------------------------------------------------------------------------------

All files are written to the current directory. In the names below,
<icname> is the name from Section 4, <k> the spatial degree, <m> the temporal
degree and <N> the value of --NMAX.

  <icname>_k<k>_m<m>_errors.txt                          (test cases 4 and 5)
      L2 errors of rho and v at the final time. One line is appended per run,
      with columns:  h  N  k  m  errrho  errv

  rhoh<icname>_k<k>_m<m>_N<N>.dat                        (test cases 2 and 6)
      Cut of rho along the line y = 0.0125, columns:  gamma  rho

  rhoh<icname>_k<k>_m<m>_N<N>_theta<theta>.dat           (test case 7)
      Cuts of rho and v along rays from the origin with angles
      theta = 0, pi/4, pi/2, 3*pi/4, columns:  gamma  rho  u  v

  <icname>_k<k>_m<m>_N<N>_cons.dat                       (all test cases)
      Total energy over time, columns:  t  E

  <icname>_k<k>_m<m>_N<N>_curl.dat                       (all test cases)
      L2 norm of curl(v) over time, columns:  t  curlerr

  <icname>_k<k>_m<m>_N<N>_LocCons.txt                    (test case 3 only)
      Local (element-wise) energy conservation error; one row per time step,
      one column per element.

  <icname>_k<k>_m<m>_N<N>*.vtu                           (all test cases)
      VTK output of rho and v, can be opened with ParaView.

Note: the "gamma" column in the cut files is the coordinate along the cut,
not the adiabatic exponent.


--------------------------------------------------------------------------------
 6. Citation
--------------------------------------------------------------------------------

If you use this code, please cite the paper:

  M. Dumbser, I. Perugia and E. Zampa, "An arbitrarily high-order accurate and
  locally energy/entropy-stable space-time method for a class of
  nonlinear wave systems".


