# =============================================================================
# Code for the paper:
#   M. Dumbser, I. Perugia and E. Zampa,
#   "An arbitrarily high-order locally energy/entropy-stable space--time method for a class of nonlinear wave systems"
# Author: Enrico Zampa
# =============================================================================
#
# Overview
# --------
# NGSolve implementation of a space-time (DG in space, Legendre/Gauss-Legendre
# in time) scheme for the nonlinear wave system
#
#       d_t rho + div v     = 0
#       d_t v   + grad s    = 0,        s = E'(rho) = rho^gamma / gamma,
#
# with total energy  0.5*|v|^2 + rho^(gamma+1) / (gamma*(gamma+1)).
#
# Each time step solves a nonlinear space-time system with Newton's method.
# The linearized systems are solved with block-Jacobi preconditioned GMRES.
# Several test cases are selected through `--ictype`
# (see the "Initial conditions" section).
#
# Command-line usage (all arguments optional):
#   python FINAL.py --ictype 2 --NMAX 40 --k 1 --m 0 --Ctime 0.1
#
# Quantities written to disk at the end of the run:
#   * L2 errors (smooth test cases only)
#   * line cuts of rho (and v) for the Riemann-problem-type test cases
#   * global energy history, norm of curl(v), local energy-conservation error
#   * a VTK file with rho and v
# =============================================================================


# =============================================================================
# 1. Command-line arguments
# =============================================================================
import argparse

parser = argparse.ArgumentParser(description="NGSolve Simulation with Overwrite Params")

# Optional arguments. Defaults are None so that we can detect whether they were
# actually passed on the command line.
parser.add_argument("--ictype", type=int, default=None, help="Initial Condition Type")
parser.add_argument("--NMAX", type=int, default=None, help="Max Mesh Scale Factor")
parser.add_argument("--k", type=int, default=None, help="Polynomial order in space")
parser.add_argument("--m", type=int, default=None, help="Polynomial order in time")
parser.add_argument("--Ctime", type=float, default=None, help="Ctime")

try:
    args = parser.parse_args()
except SystemExit:
    # Fallback for Jupyter notebooks if args are empty or conflict
    args = parser.parse_args(args=[])


# =============================================================================
# 2. Default parameters
# =============================================================================
default_order = 1

use_gmres = True  # True: iterative solvers (GMRES for Newton steps, CG for the
                  # curl projection); False: direct inverses

order = args.k if args.k is not None else default_order  # polynomial order in space
ictype = args.ictype if args.ictype is not None else 2   # test case, see section 5
nel = args.NMAX if args.NMAX is not None else 40         # mesh parameter: hmax = scaleFactor / nel
k = args.m+1 if args.m is not None else 1                # number of basis functions in time (= m + 1)
Ctime = args.Ctime if args.Ctime is not None else 0.1    # time step: dt = Ctime * hmax

tolNewton = 1e-14  # stopping tolerance (residual norm) of the Newton iteration

smax = 1.          # smax factor in the dissipation (jump penalty) terms
scaleFactor = 1    # side length scale of the domain (overridden per test case)
bio = 10           # bonus integration order

epsilon_factor = 1       # multiplier of the element-wise epsilon (see compute_epsilon)
adaptive_epsilon = True  # True: epsilon is recomputed each step; False: constant


# =============================================================================
# 3. Imports and global setup
# =============================================================================
from ngsolve import *
from netgen.occ import *
from netgen.geom2d import SplineGeometry
from netgen.meshing import IdentificationType
import numpy as np
SetHeapSize(1000000000)

time = Parameter(0.0)  # symbolic time, used by the exact solutions (ictype 4 and 5)


# -----------------------------------------------------------------------------
# Gauss-Legendre quadrature in time, rescaled from [-1, 1] to [0, 1]
# -----------------------------------------------------------------------------
nq = 10  # number of quadrature points
xi, wq = np.polynomial.legendre.leggauss(nq)
xi = xi/2 + 0.5
wq = 0.5*wq

print(xi, wq)

type_basis = 'GaussLegendre'  # time basis: 'GaussLegendre' or 'monomial'

# Boundary conditions: periodic in x and/or y (changed per test case)
periodic_x = True
periodic_y = True


# =============================================================================
# 4. Initial conditions / test cases
# =============================================================================
# Every case defines (at least): icname, v0, rho0, gamma, tend, plus the
# computational domain `shape` 

if ictype == 2:
    icname = "RP2d"  # Riemann problem on a thin strip
    shape = Rectangle(1, 0.025).Face().Move((0, 0, 0))
    v0 = CF((0, 0))
    gamma = 2
    rhomin = 0.1
    rhomax = 2
    rho0 = IfPos((x - 0.75)*(x - 0.25), rhomin, rhomax)  # 0.1 outside [0.25, 0.75], 2 inside
    tend = 0.1

    epsilon_factor = 2

elif ictype == 3:
    icname = "SmoothNonlinear2d"
    shape = Rectangle(1, 1).Face().Move((-0.5, -0.5, 0))
    v0 = CF((0, 0))
    gamma = 1.4
    sigma = 0.05
    p0 = 1
    rho0 = CF(4. + p0 * exp(-0.5*(x**2 + y**2)/(sigma ** 2)))  # smooth Gaussian bump
    tend = 0.25

    smax = 0
    epsilon_factor = 0

elif ictype == 4:
    icname = "SmoothLinear2d"  # linear case (gamma = 1) with a known exact solution
    shape = Rectangle(1, 1).Face().Move((0, 0, 0))

    gamma = 1
    tend = 0.4
    rho0 = (1/(2*sqrt(2)*pi)) * sin(2*pi*x) * sin(2*pi*y) * cos(2*sqrt(2)*pi*time)

    v0 = CoefficientFunction((
        -(1/(4*pi)) * cos(2*pi*x) * sin(2*pi*y) * sin(2*sqrt(2)*pi*time),
        -(1/(4*pi)) * sin(2*pi*x) * cos(2*pi*y) * sin(2*sqrt(2)*pi*time)
    ))

    smax = 1
    epsilon_factor = 2

elif ictype == 5:
    icname = "SmoothNonlinearEx2D"  # nonlinear simple wave with exact solution (via Newton on characteristics)
    shape = Rectangle(2*pi, 2*pi).Face().Move((0, 0, 0))

    gamma = 3
    L = 2*np.pi
    p_int, q_int = 1, 1  # oblique wave direction (integers, for periodicity)
    rho_bar = 1.0
    delta = 0.3
    kappa = 2*np.pi/L
    K = kappa*np.sqrt(p_int**2 + q_int**2)
    _norm = np.sqrt(p_int**2 + q_int**2)
    nx_hat, ny_hat = p_int/_norm, q_int/_norm
    t_star = 1.0/(K*delta)
    tend = 0.5  # 0.5*t_star

    # Exact solution: rho is constant along characteristics xi + K*rho*t = xi_exsol.
    # The implicit relation is solved with a fixed number of Newton iterations,
    # written out symbolically as a CoefficientFunction expression.
    xi_exsol = kappa*(p_int*x + q_int*y)
    xi0 = xi_exsol
    for _ in range(10):  # unrolled Newton iteration
        R0 = rho_bar + delta*sin(xi0)
        R0p = delta*cos(xi0)
        Fval = xi0 + K*R0*time - xi_exsol
        Fprim = 1 + K*R0p*time
        xi0 = xi0 - Fval/Fprim

    # .Compile(): the unrolled Newton loop makes a deep expression tree;
    # Compile() speeds up evaluation a lot.
    rho0 = (rho_bar + delta*sin(xi0)).Compile()
    v0 = CF((0.5*rho0**2*nx_hat,
             0.5*rho0**2*ny_hat)).Compile()

    smax = 1
    epsilon_factor = 2
    scaleFactor = 2 * pi
    Ctime = 0.5 / (2 * pi)

elif ictype == 6:
    icname = "RP2d_BC"  # Riemann problem with non-periodic boundary in x
    shape = Rectangle(0.5, 0.025).Face().Move((0.5, 0, 0))
    v0 = CF((0, 0))
    gamma = 2
    rhomin = 0.1
    rhomax = 2
    rho0 = IfPos((x - 0.75), rhomin, rhomax)  # 0.1 for x > 0.75, 2 otherwise
    tend = 0.1
    periodic_x = False

    smax = 1
    epsilon_factor = 2
    scaleFactor = 0.5

elif ictype == 7:
    icname = "Explosion"  # 2D radial Riemann problem (explosion)
    shape = Rectangle(2, 2).Face().Move((-1., -1., 0))
    v0 = CF((0, 0))
    gamma = 2
    rhomin = 0.1
    rhomax = 2
    r = sqrt(x**2 + y**2)
    rho0 = IfPos(r - 0.6, rhomin, rhomax)  # 0.1 outside radius 0.6, 2 inside
    tend = 0.3
    scaleFactor = 2
    epsilon_factor = 2

# =============================================================================
# 5. Mesh generation
# =============================================================================
hmax = scaleFactor / nel
dt = Ctime * hmax  # time step (may be shortened in the last step)

# Name the four sides of the rectangle
shape.edges.Max(X).name = "right"
shape.edges.Min(X).name = "left"
shape.edges.Max(Y).name = "top"
shape.edges.Min(Y).name = "bottom"

# Periodic identifications
if periodic_y:
    print("Indentify top and bottom boundary")
    shape.edges.Max(Y).Identify(shape.edges.Min(Y), "bt")
if periodic_x:
    print("Identify left and right boundary")
    shape.edges.Max(X).Identify(shape.edges.Min(X), "lr")

geo = OCCGeometry(shape, dim=2)
mesh = Mesh(geo.GenerateMesh(maxh=hmax))


# =============================================================================
# 6. Auxiliary coefficient alpha (weight of the stabilization terms)
# =============================================================================
# A cubic H1 space is used only to build a bubble-like weight: the dofs
# associated with the element interior ("faces" in 2D) are set to 27, all other
# dofs stay 0, so gf_alpha vanishes on element edges and peaks (~1) in the
# middle of each element. It multiplies the epsilon*h stabilization terms below.
fesH1 = H1(mesh, order=3)
from ngsolve import *
facedofs = fesH1.GetDofNrs(NodeId(FACE, 0))
print("dofs on face #0: ", facedofs)

gf_alpha = GridFunction(fesH1)

for face in range(mesh.nface):
    facedofs = fesH1.GetDofNrs(NodeId(FACE, face))
    for fdof in facedofs:
        gf_alpha.vec[fdof] = 27


# =============================================================================
# 7. Constitutive functions (energy E(rho) = rho^(gamma+1) / (gamma*(gamma+1)))
# =============================================================================
def dEdrho(rho):
    # E'(rho) = rho^gamma / gamma
    return 1./gamma*rho**gamma


def dEdrhorho(rho):
    # E''(rho) = rho^(gamma-1)
    return rho**(gamma-1)


def Lrhorho(rho):
    # 1 / E''(rho): mobility-like coefficient used in the dissipation terms
    return 1./rho**(gamma-1)


# =============================================================================
# 8. Finite element spaces
# =============================================================================
fesv = Discontinuous(HDiv(mesh, order=order, dgjumps=True))  # DG-type space for v (vector)
fesrho = L2(mesh, order=order, dgjumps=True)                 # DG space for rho (scalar)

fes_lo = L2(mesh, order=0)  # piecewise constants (one value per element)
gf_epsilon = GridFunction(fes_lo)  # element-wise epsilon of the stabilization
gfrho_aux = GridFunction(fes_lo)   # used to project discontinuous initial data

# Solution at the new / old time level
gfv = GridFunction(fesv)
gfv_old = GridFunction(fesv)
gfrho = GridFunction(fesrho)
gfrho_old = GridFunction(fesrho)

# Space-time unknowns. For every time basis function j = 0..k-1 the block is
#     [ v_j (HDiv),  u_j (HDiv),  rho_j (L2),  s_j (L2) ]
# so component 4*j + c  (c = 0, 1, 2, 3) is v, u, rho, s of the j-th time mode.
fes = fesv*fesv*fesrho*fesrho
for i in range(k-1):
    fes *= fesv*fesv*fesrho*fesrho

(a, b) = fes.TnT()  # trial / test functions (re-defined identically below)


# =============================================================================
# 9. Time basis functions
# =============================================================================
def basis(x):
    # Returns (phi, psi) evaluated at x in [0, 1]:
    #   phi_i(x): i-th basis function in time (shifted Legendre or monomial)
    #   psi_i(x): its primitive, psi_i(x) = int_0^x phi_i(s) ds
    if type_basis == 'monomial':
        phi = np.array([x**i for i in range(k)])
        psi = np.array([x**(i+1) / (i + 1) for i in range(k)])
    elif type_basis == 'GaussLegendre':
        N = k-1
        # We need phi up to degree N+1 to compute psi up to degree N
        phi = np.zeros(N + 2)

        # Base cases for phi
        phi[0] = np.ones_like(x)
        if N >= 0:
            phi[1] = 2.0 * x - 1.0

        # Bonnet's recursion for shifted Legendre polynomials
        for i in range(1, N + 1):
            term1 = (2.0 * i + 1.0) * (2.0 * x - 1.0) * phi[i]
            term2 = i * phi[i-1]
            phi[i+1] = (term1 - term2) / (i + 1.0)

        # Initialize psi
        psi = np.zeros(N + 1)

        # Base case for psi
        psi[0] = x

        # Compute psi using the integral property
        for i in range(1, N + 1):
            psi[i] = (phi[i+1] - phi[i-1]) / (2.0 * (2.0 * i + 1.0))

        # Return arrays truncated to the requested degree N
    return phi[:k], psi


# Basis values at the quadrature points: row l <-> quadrature point xi[l]
phi_GL = np.zeros((nq, k))
psi_GL = np.zeros((nq, k))

for l in range(nq):
    phi_GL[l, :], psi_GL[l, :] = basis(xi[l])

# Basis values at the end of the time step (t = 1 in reference time)
phi_1, psi_1 = basis(1.)


(a, b) = fes.TnT()

# Legacy: alternative layout with a single combined grid function
# gf = GridFunction(fes)
# gfv, gfrho, gfs = gf.components
# gf_old = GridFunction(fes)
# gfv_old, gfrho_old, gfs_old = gf_old.components
#
# gf_Pic = GridFunction(fes)
# gfv_Pic, gfrho_Pic, gfs_Pic = gf_Pic.components

# Integration over element boundaries, normal vector and local mesh size
dS = dx(element_boundary=True)
n = specialcf.normal(mesh.dim)
h = specialcf.mesh_size


# =============================================================================
# 10. Piecewise-constant space for the local conservation error
# =============================================================================
fes_cons = L2(mesh, order=0)
(a_aux, b_aux) = fes_cons.TnT()
cons_err = GridFunction(fes_cons)
cons_err_vec = cons_err.vec.CreateVector()


# =============================================================================
# 11. Nonlinear solver (one time step)
# =============================================================================
def NewtonIteration(gfv_old_vec, gfrho_old_vec, gfstar):
    # -------------------------------------------------------------------------
    # Performs ONE Newton iteration for the space-time system of the current
    # time step, starting from the current iterate `gfstar`.
    #
    # Inputs : v and rho at the old time level, current iterate gfstar
    # Outputs: updated iterate, residual norm after the update, and the
    #          element-wise local conservation defect (cons_err_vec)
    # -------------------------------------------------------------------------

    # Old solution as grid functions
    gfvold = GridFunction(fesv)
    gfvold.vec.data = gfv_old_vec

    gfrhoold = GridFunction(fesrho)
    gfrhoold.vec.data = gfrho_old_vec

    bf = BilinearForm(fes, nonassemble=True)  # nonlinear form (residual)
    bf_lin = BilinearForm(fes)                # linearized form (Jacobian)
    bf_cons = BilinearForm(fes, nonassemble=True)  # local conservation defect

    # Averaged coefficient for the dissipation across element interfaces
    LL = Lrhorho(gfrhoold)
    LR = Lrhorho(gfrhoold.Other())
    Lav = 2*(LL*LR) / (LL + LR)

    for l in range(nq):  # run over time quadrature points
        # ---------------------------------------------------------------------
        # Evaluate the space-time functions at the quadrature point xi[l]
        # ---------------------------------------------------------------------
        # v at time xi[l]: v_old + sum_j psi_j(xi[l]) * (v-coefficient)
        v_l = 1*gfvold
        v_l_lin = 0*gfvold  # "homogeneous" version of v_l (without v_old)
        dvdt_l = 0*gfvold   # time derivative of v
        # u_l (u is the auxiliary variable u = v)
        u_l = 0*gfvold
        divu_l = 0*div(gfvold)
        divu_l_other = 0*div(gfvold)
        u_l_other = 0*gfvold
        # rho_l
        rho_l = 1*gfrhoold
        rho_l_lin = 0*gfrhoold  # "homogeneous" version of rho_l
        drhodt_l = 0*gfrhoold   # time derivative of rho
        # s_l (s is the auxiliary variable s = E'(rho))
        s_l = 0*gfrhoold
        grads_l = 0*grad(gfrhoold)
        grads_l_other = 0*grad(gfrhoold)
        s_l_other = 0*gfrhoold

        # rho at the quadrature point evaluated at the current Newton iterate
        # (linearization point)
        gfrho_l = 1*gfrhoold

        for j in range(k):
            # v_l
            v_l += psi_GL[l, j] * a[4*j]
            v_l_lin += psi_GL[l, j] * a[4*j]
            dvdt_l += phi_GL[l, j] * a[4*j]
            # u_l
            u_l += phi_GL[l, j] * a[4*j + 1]
            divu_l += phi_GL[l, j] * div(a[4*j + 1])
            divu_l_other += phi_GL[l, j] * div(a[4*j + 1].Other())
            u_l_other += phi_GL[l, j] * a[4*j + 1].Other()
            # rho_l
            rho_l += psi_GL[l, j] * a[4*j + 2]
            rho_l_lin += psi_GL[l, j] * a[4*j + 2]
            drhodt_l += phi_GL[l, j] * a[4*j + 2]
            gfrho_l += psi_GL[l, j] * gfstar.components[4*j + 2]
            # s_l
            s_l += phi_GL[l, j] * a[4*j + 3]
            grads_l += phi_GL[l, j] * grad(a[4*j + 3])
            grads_l_other += phi_GL[l, j] * grad(a[4*j + 3].Other())
            s_l_other += phi_GL[l, j] * a[4*j + 3].Other()

        # ---------------------------------------------------------------------
        # Terms of the weak form (initialised to zero, then accumulated)
        # ---------------------------------------------------------------------
        eqv = 0*b[2]*dx
        eqv_BC = 0*b[2]*dx
        eqrho = 0*b[2]*dx
        equ = 0*b[2]*dx
        equ_lin = 0*b[2]*dx
        eqs = 0*b[2]*dx
        eqs_lin = 0*b[2]*dx

        # Local conservation defect: interface term between u and s
        bf_cons += wq[l] * dt * 0.5 * (u_l * n * s_l_other + u_l_other * n * s_l) * b[3] * dS

        for i in range(k):  # run over test functions in time
            # --- v equation:  ( partial_t v, w ) - ( s, div w ) + fluxes -----
            eqv += wq[l] * 1.0 * dvdt_l * b[4*i] * phi_GL[l, i] * dx
            eqv += - wq[l] * dt * s_l * div(b[4*i]) * phi_GL[l, i] * dx
            eqv += wq[l] * dt * 0.5*(s_l + s_l_other) * (b[4*i] - b[4*i].Other()) * n * phi_GL[l, i] * dx(skeleton=True)

            # boundary term for non-periodic boundaries
            if periodic_x == False or periodic_y == False:
                eqv_BC += wq[l] * dt * dEdrho(rho0) * b[4*i] * n * phi_GL[l, i] * ds(skeleton=True)

            # dissipation (jump penalty on u.n and epsilon-stabilization on div u)
            eqv += wq[l] * dt * 0.5 * smax * (u_l - u_l_other) * n * (b[4*i] - b[4*i].Other()) * n * phi_GL[l, i] * dx(skeleton=True)
            eqv += wq[l] * dt * gf_alpha * gf_epsilon * h * divu_l * div(b[4*i]) * phi_GL[l, i] * dx(bonus_intorder=bio)

            # --- u equation:  u = v  ------------------------------------------
            equ += wq[l] * dt * u_l * b[4*i + 1] * phi_GL[l, i] * dx
            equ += -wq[l] * dt * v_l * b[4*i + 1] * phi_GL[l, i] * dx
            # linearized version (Jacobian)
            equ_lin += wq[l] * dt * u_l * b[4*i + 1] * phi_GL[l, i] * dx
            equ_lin += -wq[l] * dt * v_l_lin * b[4*i + 1] * phi_GL[l, i] * dx

            # --- rho equation:  partial_t rho + div u = 0 ---------------------
            eqrho += wq[l] * 1.0 * drhodt_l * b[4*i + 2] * phi_GL[l, i] * dx
            eqrho += - wq[l] * dt * u_l * grad(b[4*i + 2]) * phi_GL[l, i] * dx
            eqrho += wq[l] * dt * 0.5*(u_l + u_l_other) * n * (b[4*i + 2] - b[4*i + 2].Other()) * phi_GL[l, i] * dx(skeleton=True)

            # boundary term for non-periodic boundaries
            if periodic_x == False or periodic_y == False:
                eqrho += wq[l] * dt * u_l * n * b[4*i + 2] * phi_GL[l, i] * ds(skeleton=True)

            # dissipation (jump penalty on s and epsilon-stabilization on grad s)
            eqrho += wq[l] * dt * 0.5 * smax * Lav * (s_l - s_l_other) * (b[4*i + 2] - b[4*i + 2].Other()) * phi_GL[l, i] * dx(skeleton=True)
            eqrho += wq[l] * dt * gf_alpha * gf_epsilon * h * Lrhorho(gfrhoold) * grads_l * grad(b[4*i + 2]) * phi_GL[l, i] * dx(bonus_intorder=bio)

            # --- s equation:  ( s, xi ) = ( dE/drho, xi )  --------------------
            eqs += wq[l] * dt * s_l * b[4*i + 3] * phi_GL[l, i] * dx
            eqs += -wq[l] * dt * dEdrho(rho_l) * b[4*i + 3] * phi_GL[l, i] * dx(bonus_intorder=bio)
            # linearized version: E'(rho) ~ E''(rho*) * rho
            eqs_lin += wq[l] * dt * s_l * b[4*i + 3] * phi_GL[l, i] * dx
            eqs_lin += -wq[l] * dt * dEdrhorho(gfrho_l) * rho_l_lin * b[4*i + 3] * phi_GL[l, i] * dx(bonus_intorder=bio)

        # Add this quadrature point's contribution to the forms
        bf += eqv + equ + eqrho + eqs + eqv_BC
        bf_lin += eqv + equ_lin + eqrho + eqs_lin

    bf_lin.Assemble()

    # Residual at the current iterate
    rhs = gfstar.vec.CreateVector()
    rhs[:] = 0
    bf.Apply(gfstar.vec, rhs)

    # Newton update: solve J * delta = residual
    gfdelta = GridFunction(fes)

    if use_gmres:
        # --------------------------------------------------------------------
        # Block-Jacobi preconditioner: one block per element
        # --------------------------------------------------------------------
        freedofs = fes.FreeDofs()

        blocks = []
        for el in mesh.Elements(VOL):
            dofs = [
                d for d in fes.GetDofNrs(el)
                if freedofs[d]
            ]

            if dofs:
                blocks.append(dofs)

        pre = bf_lin.mat.CreateBlockSmoother(blocks)

        # --------------------------------------------------------------------
        # GMRES
        # --------------------------------------------------------------------
        solvers.GMRes(
            A=bf_lin.mat,
            b=rhs,
            x=gfdelta.vec,
            pre=pre,
            tol=1e-15,
            maxsteps=200,
            restart=30,
            printrates=True
        )
    else:
        # Direct solve
        gfdelta.vec.data = bf_lin.mat.Inverse(freedofs=fes.FreeDofs())*rhs

    gfstar.vec.data += -gfdelta.vec

    # Residual norm after the update
    rhs[:] = 0
    bf.Apply(gfstar.vec, rhs)
    res = np.linalg.norm(rhs)

    # Local conservation defect: one value per element, extracted from the
    # 4th component (s) of the first time mode (the first dof of each element
    # block, which is the element-constant part)
    aux_vec = gfstar.vec.CreateVector()
    bf_cons.Apply(gfstar.vec, aux_vec)

    aux_vec_2 = np.array(aux_vec[fes.Range(3)])
    dimPk = int((order+1)*(order+2)/2)  # dofs per element of P^order
    cons_err_vec = aux_vec_2[::dimPk]

    return gfstar.vec, res, cons_err_vec


def Newton(gfv_old_vec, gfrho_old_vec):
    # Full Newton loop for one time step. Iterates until the residual norm is
    # below tolNewton, then reconstructs v and rho at the end of the step.
    gfstar = GridFunction(fes)
    res = 1e10
    while res > tolNewton:
        gfstar.vec.data, res, cons_err_vec = NewtonIteration(gfv_old_vec, gfrho_old_vec, gfstar)
        print(f"res = {res}")

    # evaluate v and rho at the final time: old value + sum_i psi_i(1) * coefficient_i
    gfv_new_vec = gfv_old_vec.CreateVector()
    gfv_new_vec.data = gfv_old_vec

    gfrho_new_vec = gfrho_old_vec.CreateVector()
    gfrho_new_vec.data = gfrho_old_vec

    for i in range(k):
        gfv_new_vec.data += psi_1[i] * gfstar.components[4*i].vec
        gfrho_new_vec.data += psi_1[i] * gfstar.components[4*i + 2].vec

    return gfv_new_vec, gfrho_new_vec, cons_err_vec


# =============================================================================
# 12. Curl of v (diagnostic)
# =============================================================================
def bot(v):
    # rotation by -90 degrees: (v0, v1) -> (v1, -v0)
    return CF((v[1], -v[0]))


# L2 projection of curl(v) onto a periodic H1 space (weak form:
# (curl v, phi) = (v, bot(grad phi)) on the periodic domain)
fescurl = Periodic(H1(mesh, order=order + 1))
(phi, dphi) = fescurl.TnT()

bf_curl = BilinearForm(fescurl, symmetric=True)  # mass matrix
bf_curl += phi*dphi*dx

pre_curl = Preconditioner(bf_curl, "bddc")

bf_curl.Assemble()

if not use_gmres:
    inv_bf_curl = bf_curl.mat.Inverse(freedofs=fescurl.FreeDofs(), inverse="sparsecholesky")

gf_curl = GridFunction(fescurl)

rhs_curl = LinearForm(fescurl)
rhs_curl += gfv*bot(grad(dphi))*dx


# =============================================================================
# 13. Element-wise epsilon (smoothness indicator for the stabilization)
# =============================================================================
(rho, eta) = fesrho.TnT()
(u, du) = fesv.TnT()

# Weak DG gradient of rho (central flux): maps rho in fesrho -> fesv (test side)
arhs = BilinearForm(trialspace=fesrho, testspace=fesv, nonassemble=True)
arhs += grad(rho) * du*dx - (rho - rho.Other()) * 0.5*(du + du.Other()) * n * dx(skeleton=True)

massv = BilinearForm(fesv, symmetric=True)
massv += u*du*dx
massv.Assemble()


def compute_epsilon(gfv_old_vec, gfrho_old_vec):
    # Returns one value per element: the L2 norm, on each element, of the
    # difference between the element-wise gradient of rho and its DG gradient
    # (which accounts for the jumps of rho across element interfaces).
    # It is small where rho is smooth and large near discontinuities.
    gfvold = GridFunction(fesv)
    gradrho_DG = GridFunction(fesv)
    gfvold.vec.data = gfv_old_vec

    gfrhoold = GridFunction(fesrho)
    gfrhoold.vec.data = gfrho_old_vec

    auxvec = gfvold.vec.CreateVector()
    arhs.Apply(gfrhoold.vec, auxvec)

    gradrho_DG.vec.data = massv.mat.Inverse(inverse="pardiso") * auxvec

    output_vec = sqrt(Integrate((grad(gfrho_old) - gradrho_DG)**2, mesh, element_wise=True))
    denominator_vec = sqrt(Integrate(Norm(gradrho_DG)**2, mesh, element_wise=True)) + 1e-12

    # (normalized indicator; computed but not used - the unnormalized
    # output_vec is returned)
    vec = output_vec/denominator_vec

    return output_vec


# =============================================================================
# 14. Set initial condition
# =============================================================================
if ictype == 1 or ictype == 3 or ictype == 4 or ictype == 5:
    # smooth data: interpolate directly
    gfrho_old.Set(rho0)
else:
    # discontinuous data: go through a piecewise-constant projection first
    gfrho_aux.Set(rho0)
    gfrho_old.Set(gfrho_aux)

gfv_old.Set(v0, bonus_intorder=bio)

# =============================================================================
# 15. Time stepping
# =============================================================================
t = 0

# History of the total energy  E = 0.5*|v|^2 + rho^(gamma+1)/(gamma*(gamma+1))
E = []
E.append(0.5*Integrate(gfv_old**2, mesh) + 1/(gamma*(gamma+1)) * Integrate(gfrho_old**(gamma + 1), mesh, order=10))

# History of the L2 norm of curl(v)
NormCurl = []
NormCurl.append(0)

gfrho.vec.data = gfrho_old.vec
gfv.vec.data = gfv_old.vec

gfEerr = GridFunction(fes_cons)  # element-wise local energy-conservation error

loccons = []  # local conservation error at every time step

with TaskManager():
    while t < tend-1e-12:

        # shorten the last step so that we land exactly on tend
        if t + dt > tend:
            dt = tend - t

        # element-wise epsilon for the stabilization terms
        if adaptive_epsilon:
            gf_epsilon.vec[:] = epsilon_factor * compute_epsilon(gfv_old.vec, gfrho_old.vec)
        else:
            gf_epsilon.Set(epsilon_factor)

        # solve the nonlinear space-time system for this time step
        gfv.vec.data, gfrho.vec.data, cons_err_vec = Newton(gfv_old.vec, gfrho_old.vec)

        # if ictype == 2:
        #     gfv.vec.data, gfrho.vec.data = add_diffusion(gfv.vec, gfrho.vec)

        # element-wise energy before the step
        Eold = (0.5*Integrate(gfv_old**2, mesh, element_wise=True)
                + 1/(gamma*(gamma+1)) * Integrate(gfrho_old**(gamma + 1), mesh, order=5, element_wise=True))

        # advance in time
        gfrho_old.vec.data = gfrho.vec
        gfv_old.vec.data = gfv.vec
        E.append(0.5*Integrate(gfv**2, mesh) + 1/(gamma*(gamma+1)) * Integrate(gfrho**(gamma + 1), mesh, order=5))

        # element-wise energy after the step
        Enew = (0.5*Integrate(gfv**2, mesh, element_wise=True)
                + 1/(gamma*(gamma+1)) * Integrate(gfrho**(gamma + 1), mesh, order=5, element_wise=True))

        # local conservation error: (defect from the scheme) + (energy change)
        cons_err_vec += Enew - Eold
        gfEerr.vec.data = np.abs(cons_err_vec)
        loccons.append(np.abs(cons_err_vec))

        # compute curl of v
        rhs_curl.Assemble()
        if use_gmres:
            print("compute curl")
            solvers.CG(mat=bf_curl.mat, rhs=rhs_curl.vec, pre=pre_curl, sol=gf_curl.vec, tol=1e-14, maxsteps=2000, printrates=True)
        else:
            gf_curl.vec.data = inv_bf_curl*rhs_curl.vec
        norm_curl = sqrt(Integrate(gf_curl**2, mesh))
        NormCurl.append(norm_curl)
        print(f"Norm of curl v = {norm_curl}")

        t += dt

        print(t)


# =============================================================================
# 16. Post-processing: L2 error at the final time (smooth test cases)
# =============================================================================
time.Set(tend)  # exact solution of ictype 4 and 5 depends on `time`
if ictype == 4 or ictype == 5:
    errrho = sqrt(Integrate(Norm(gfrho-rho0)**2, mesh))
    errv = sqrt(Integrate(Norm(gfv - v0)**2, mesh))

    print(f"L2 error for rho at the final time = {errrho}")
    print(f"L2 error for v at the final time = {errv}")

    import os

    error_filename = (
        icname
        + "_k" + str(order)
        + "_m" + str(k-1)
        + "_errors.txt"
    )

    file_exists = os.path.isfile(error_filename)

    # append one line per run (convergence table: h, N, k, m, errors)
    with open(error_filename, "a") as f:
        if not file_exists:
            f.write("# h N k m errrho errv\n")

        f.write(
            f"{hmax:.16e} {nel:d} {order:d} {k-1:d} "
            f"{float(errrho):.16e} {float(errv):.16e}\n"
        )


# =============================================================================
# 17. Post-processing: line cuts of the solution
# =============================================================================
# Note: from here on `gamma` is reused as the coordinate along the cut
# (it no longer denotes the adiabatic exponent).

# Horizontal cut of rho through the middle of the strip (Riemann problems)
if ictype == 2 or ictype == 6:
    rhoval = []

    gamma_vals = []

    for gamma in np.arange(0.5 + hmax/10, 1 - hmax/10, hmax/10):
        xpt = gamma
        ypt = 0.0125
        mip = mesh(xpt, ypt)
        gamma_vals.append(gamma)

        rhoval.append(gfrho(mip))

    data = np.column_stack((gamma_vals, rhoval))

    np.savetxt(
        "rhoh" + icname + "_k" + str(order) + "_m" + str(k-1) + "_N" + str(nel) + ".dat",
        data,
        header="gamma rho",
        fmt="%.16e"
    )

# Radial cuts of rho and v for the explosion problem, along the directions
# theta = 0, pi/4, pi/2, 3*pi/4 starting from the origin (x0, y0)
# (the four blocks below are identical except for the angle theta)
if ictype == 7:
    rhoval = []
    gfv1val = []
    gfv2val = []
    gamma_vals = []

    theta = 0               # angle in radians
    x0, y0 = 0.0, 0.0       # starting point

    for gamma in np.arange(0 + hmax/10, 1 - hmax/10, hmax/10):
        xpt = x0 + gamma * np.cos(theta)
        ypt = y0 + gamma * np.sin(theta)

        mip = mesh(xpt, ypt)

        gamma_vals.append(gamma)
        rhoval.append(gfrho(mip))
        gfv1val.append(gfv[0](mip))
        gfv2val.append(gfv[1](mip))

    data = np.column_stack((gamma_vals, rhoval, gfv1val, gfv2val))

    np.savetxt(
        "rhoh" + icname + "_k" + str(order) + "_m" + str(k-1) + "_N" + str(nel) + "_theta" + str(theta) + ".dat",
        data,
        header="gamma rho u v",
        fmt="%.16e"
    )

    # --- theta = pi/4 ---
    rhoval = []
    gfv1val = []
    gfv2val = []
    gamma_vals = []
    theta = pi/4            # angle in radians

    for gamma in np.arange(0 + hmax/10, 1 - hmax/10, hmax/10):
        xpt = x0 + gamma * np.cos(theta)
        ypt = y0 + gamma * np.sin(theta)

        mip = mesh(xpt, ypt)

        gamma_vals.append(gamma)
        rhoval.append(gfrho(mip))
        gfv1val.append(gfv[0](mip))
        gfv2val.append(gfv[1](mip))

    data = np.column_stack((gamma_vals, rhoval, gfv1val, gfv2val))

    np.savetxt(
        "rhoh" + icname + "_k" + str(order) + "_m" + str(k-1) + "_N" + str(nel) + "_theta" + str(theta) + ".dat",
        data,
        header="gamma rho u v",
        fmt="%.16e"
    )

    # --- theta = pi/2 ---
    rhoval = []
    gfv1val = []
    gfv2val = []
    gamma_vals = []
    theta = pi/2            # angle in radians

    for gamma in np.arange(0 + hmax/10, 1 - hmax/10, hmax/10):
        xpt = x0 + gamma * np.cos(theta)
        ypt = y0 + gamma * np.sin(theta)

        mip = mesh(xpt, ypt)

        gamma_vals.append(gamma)
        rhoval.append(gfrho(mip))
        gfv1val.append(gfv[0](mip))
        gfv2val.append(gfv[1](mip))

    data = np.column_stack((gamma_vals, rhoval, gfv1val, gfv2val))

    np.savetxt(
        "rhoh" + icname + "_k" + str(order) + "_m" + str(k-1) + "_N" + str(nel) + "_theta" + str(theta) + ".dat",
        data,
        header="gamma rho u v",
        fmt="%.16e"
    )

    # --- theta = 3*pi/4 ---
    rhoval = []
    gfv1val = []
    gfv2val = []
    gamma_vals = []
    theta = pi/4 * 3        # angle in radians

    for gamma in np.arange(0 + hmax/10, 1 - hmax/10, hmax/10):
        xpt = x0 + gamma * np.cos(theta)
        ypt = y0 + gamma * np.sin(theta)

        mip = mesh(xpt, ypt)

        gamma_vals.append(gamma)
        rhoval.append(gfrho(mip))
        gfv1val.append(gfv[0](mip))
        gfv2val.append(gfv[1](mip))

    data = np.column_stack((gamma_vals, rhoval, gfv1val, gfv2val))

    np.savetxt(
        "rhoh" + icname + "_k" + str(order) + "_m" + str(k-1) + "_N" + str(nel) + "_theta" + str(theta) + ".dat",
        data,
        header="gamma rho u v",
        fmt="%.16e"
    )


# =============================================================================
# 18. Post-processing: energy, curl and local-conservation histories
# =============================================================================
# Total energy over time (assumes uniform output times 0 .. tend)
tvec = np.linspace(0, tend, len(E))
data = np.column_stack((tvec, E))

np.savetxt(
    icname + "_k" + str(order) + "_m" + str(k-1) + "_N" + str(nel) + "_cons.dat",
    data,
    header="t E",
    fmt="%.16e"
)

# Norm of curl(v) over time
data = np.column_stack((tvec, NormCurl))

np.savetxt(
    icname + "_k" + str(order) + "_m" + str(k-1) + "_N" + str(nel) + "_curl.dat",
    data,
    header="t curlerr",
    fmt="%.16e"
)

# Local conservation error (array: time steps x elements); saved for ictype 3 only
loccons = np.array(loccons)
if ictype == 3:
    np.savetxt(icname + "_k" + str(order) + "_m" + str(k-1) + "_N" + str(nel) + "_LocCons.txt", loccons)

# Energy change relative to the initial energy
print(np.array(E)-np.array(E)[0])


# =============================================================================
# 19. VTK output
# =============================================================================
vtk = VTKOutput(mesh, coefs=[gfrho, gfv], names=["rho", "v"], filename=icname + "_k" + str(order) + "_m" + str(k-1) + "_N" + str(nel), subdivision=3)
vtk.Do()
