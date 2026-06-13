"""
================================================================================
 Physics ToT — Generic SymPy Physics Skills
================================================================================
 Pure computational skills: analytical mechanics, electromagnetism, quantum
 mechanics, thermodynamics, relativity, optics, waves, and fluid dynamics.
 LLMs build the physical model and reasoning path, then call these functions
 to obtain exact analytic or numerical results.

 The ToT-specific layer (hard-rule checking, prompt contracts, plugin
 templates, and the skill registry) lives in ``skills.py``.

================================================================================
"""

from __future__ import annotations

from typing import Any, Dict, List, Optional, Sequence, Tuple, Union

import sympy as sp
from sympy.core.function import AppliedUndef, UndefinedFunction
from sympy import (
    Derivative,
    Eq,
    Function,
    Matrix,
    Rational,
    Symbol,
    diff,
    dsolve,
    expand,
    eye,
    simplify,
    solve,
    symbols,
    zeros,
)



t = sp.Symbol("t", real=True, positive=True)


# ==============================================================================
# Module 1  Analytical Mechanics
# ==============================================================================
#
#   1.1  Lagrangian Mechanics      — automatic Euler-Lagrange derivation
#   1.2  Hamiltonian Mechanics     — Legendre transform + canonical equations
#   1.3  Rigid Body Dynamics       — inertia tensor / principal axes / Euler equations
#
# ------------------------------------------------------------------------------


# ---------- Internal utilities ------------------------------------------------

def _as_functions_of_t(qs: Sequence[Union[str, sp.Function, sp.Expr]],
                       time: sp.Symbol = t
                       ) -> List[sp.Function]:
    """
    Convert entries in ``qs`` into applied sympy.Function objects of the form
    q_i(t).

    Supported inputs:
        - string names such as "q1"
        - unapplied sympy Function objects (for example q = Function('q'))
        - already-applied objects such as q(t)
    """
    out = []
    for q in qs:
        if isinstance(q, str):
            out.append(Function(q)(time))
        elif isinstance(q, AppliedUndef):
            # Already in the form q(t)
            out.append(q)
        elif isinstance(q, UndefinedFunction):
            # The Function class itself, not yet applied
            out.append(q(time))
        else:
            raise TypeError(f"Cannot interpret coordinate {q!r} as q(t).")
    return out


def _qdot(q_t: sp.Expr, time: sp.Symbol = t) -> sp.Expr:
    """Return the first time derivative q'(t) of q(t)."""
    return sp.diff(q_t, time)


def _qddot(q_t: sp.Expr, time: sp.Symbol = t) -> sp.Expr:
    """Return the second time derivative q''(t) of q(t)."""
    return sp.diff(q_t, time, 2)


def _infer_time_symbol(expressions: Sequence[Any], fallback: sp.Symbol = t) -> sp.Symbol:
    """Infer a time symbol from expressions; fall back if the result is ambiguous."""
    candidates: List[sp.Symbol] = []
    for expr in expressions:
        if expr is None:
            continue
        if isinstance(expr, str):
            continue
        try:
            sym_expr = sp.sympify(expr)
        except Exception:
            continue

        if isinstance(sym_expr, AppliedUndef) and len(sym_expr.args) == 1 and isinstance(sym_expr.args[0], sp.Symbol):
            candidates.append(sym_expr.args[0])

        for func in sym_expr.atoms(AppliedUndef):
            if len(func.args) == 1 and isinstance(func.args[0], sp.Symbol):
                candidates.append(func.args[0])

        for deriv in sym_expr.atoms(Derivative):
            for var in deriv.variables:
                if isinstance(var, sp.Symbol):
                    candidates.append(var)

    unique: List[sp.Symbol] = []
    for candidate in candidates:
        if candidate not in unique:
            unique.append(candidate)
    return unique[0] if len(unique) == 1 else fallback


def _expr_is_zero(expr: sp.Expr) -> bool:
    simplified = sp.simplify(expr)
    return bool(simplified == 0 or simplified.equals(0))


def _matrix_is_zero(expr: Union[sp.MatrixBase, Sequence[sp.Expr]]) -> bool:
    matrix = sp.Matrix(expr)
    return all(_expr_is_zero(entry) for entry in matrix)


def _explicit_time_derivative(
    expr: sp.Expr,
    coords: Sequence[sp.Expr],
    velocities: Sequence[sp.Expr],
    time: sp.Symbol,
) -> sp.Expr:
    """Take only the explicit time derivative, ignoring implicit q_i(t) and q̇_i(t) dependence."""
    replacements: Dict[sp.Expr, sp.Symbol] = {}
    for index, q in enumerate(coords, start=1):
        replacements[q] = sp.Symbol(f"_q_{index}")
    for index, qd in enumerate(velocities, start=1):
        replacements[qd] = sp.Symbol(f"_qd_{index}")
    return sp.simplify(sp.diff(expr.xreplace(replacements), time))


def _coordinate_system_with_scale_factors(
    name: str,
    coords: Optional[Sequence[sp.Symbol]] = None,
) -> Tuple[Tuple[sp.Symbol, sp.Symbol, sp.Symbol], Tuple[sp.Expr, sp.Expr, sp.Expr], str]:
    """Return coordinates and Lamé scale factors; rebuild scales if coords are overridden."""
    default_coords, _, label = _coordinate_system(name)
    if coords is None:
        coords_tuple = default_coords
    else:
        if len(coords) != 3:
            raise ValueError("Coordinate override must contain exactly three symbols.")
        coords_tuple = tuple(coords)

    if label == "cartesian":
        scales = (sp.Integer(1), sp.Integer(1), sp.Integer(1))
    elif label == "cylindrical":
        rho, _, _ = coords_tuple
        scales = (sp.Integer(1), rho, sp.Integer(1))
    elif label == "spherical":
        r_coord, theta_coord, _ = coords_tuple
        scales = (sp.Integer(1), r_coord, r_coord * sp.sin(theta_coord))
    else:
        raise ValueError(f"Unknown coordinate system: {label}")

    return coords_tuple, scales, label


# ---------- 1.1  Lagrangian mechanics -----------------------------------------

def lagrangian_equations(params: Dict[str, Any]) -> Dict[str, Any]:
    r"""
    Construct the Lagrangian L = T - V and automatically derive the
    Euler-Lagrange equations:

        d/dt (∂L/∂q̇_i) - ∂L/∂q_i = 0

    Parameters
    ----------
    params : dict
        Required fields:
            "T"            : kinetic-energy expression (sympy.Expr) in terms of
                             q_i(t) and q_i'(t)
            "V"            : potential-energy expression (sympy.Expr)
            "coords"       : list of generalized coordinates, such as
                             ["q1", "q2"], [q1, q2] (Function), or
                             [q1(t), q2(t)] (applied Function objects)
        Optional fields:
            "time"         : time symbol, default t
            "solve_ode"    : bool, whether to call dsolve on the EOM
                             (default False)
            "ics"          : initial-condition dictionary for dsolve (optional)

    Returns
    -------
    dict
        {
            "L"          : Lagrangian expression,
            "coords"     : list [q_i(t)],
            "velocities" : list [q_i'(t)],
            "EL_eqs"     : list [Eq(..., 0)],  # Euler-Lagrange equations
            "EOM"        : list of equations solved explicitly for q_i''(t),
                             when available,
            "solution"   : dsolve output, if solve_ode=True,
        }
    """
    T = sp.sympify(params["T"])
    V = sp.sympify(params["V"])
    time = params.get("time")
    if time is None:
        time = _infer_time_symbol([params.get("T"), params.get("V"), *params.get("coords", [])], t)
    coords = _as_functions_of_t(params["coords"], time)
    vels = [_qdot(q, time) for q in coords]

    L = sp.simplify(T - V)

    EL_eqs: List[sp.Eq] = []
    for q, qd in zip(coords, vels):
        dL_dqd = sp.diff(L, qd)
        dL_dq = sp.diff(L, q)
        eq = sp.simplify(sp.diff(dL_dqd, time) - dL_dq)
        EL_eqs.append(sp.Eq(eq, 0))

    # Try to solve each equation explicitly for q_i''(t)
    EOM: List[sp.Eq] = []
    qdds = [_qddot(q, time) for q in coords]
    try:
        sol_acc = sp.solve([eq.lhs for eq in EL_eqs], qdds, dict=True)
        if sol_acc:
            sol_acc = sol_acc[0]
            for q, qdd in zip(coords, qdds):
                if qdd in sol_acc:
                    EOM.append(sp.Eq(qdd, sp.simplify(sol_acc[qdd])))
    except Exception:
        EOM = []

    out: Dict[str, Any] = {
        "L": L,
        "coords": coords,
        "velocities": vels,
        "EL_eqs": EL_eqs,
        "EOM": EOM,
    }

    if params.get("solve_ode", False):
        try:
            ics = params.get("ics", None)
            sol = sp.dsolve([eq for eq in EL_eqs], coords, ics=ics)
            out["solution"] = sol
            out["solution_status"] = "success"
        except Exception as e:
            out["solution"] = f"dsolve failed: {e}"
            out["solution_status"] = "failed"

    return out


# ---------- 1.2  Hamiltonian mechanics ----------------------------------------

def hamiltonian_equations(params: Dict[str, Any]) -> Dict[str, Any]:
    r"""
    Obtain the Hamiltonian from the Lagrangian via a Legendre transform:

        p_i = ∂L/∂q̇_i,
        H   = Σ p_i q̇_i - L,
        q̇_i =  ∂H/∂p_i,
        ṗ_i = -∂H/∂q_i.

    Parameters
    ----------
    params : dict
        Required:
            "L" or ("T", "V") : the Lagrangian or the pair
                                    (kinetic energy, potential energy)
            "coords"             : generalized coordinates
        Optional:
            "time"               : default t
            "momenta_symbols"    : names of momentum symbols,
                                    default ["p1", "p2", ...]

    Returns
    -------
    dict
        {
            "L"         : Lagrangian,
            "coords"    : [q_i(t)],
            "velocities": [q̇_i(t)],
            "momenta"   : [p_i],          # momentum symbols
            "p_defs"    : [Eq(p_i, ∂L/∂q̇_i)],
            "H"         : Hamiltonian written in q_i and p_i,
            "canonical" : [Eq(q̇_i,∂H/∂p_i), Eq(ṗ_i,-∂H/∂q_i), ...],
        }
    """
    time = params.get("time")
    if time is None:
        time = _infer_time_symbol([params.get("L"), params.get("T"), params.get("V"), *params.get("coords", [])], t)
    coords = _as_functions_of_t(params["coords"], time)
    vels = [_qdot(q, time) for q in coords]

    if "L" in params:
        L = sp.sympify(params["L"])
    else:
        L = sp.sympify(params["T"]) - sp.sympify(params["V"])

    n = len(coords)
    p_names = params.get("momenta_symbols",
                         [f"p{i+1}" for i in range(n)])
    p_syms = [sp.Symbol(name) for name in p_names]
    p_funcs = [sp.Function(name)(time) for name in p_names]

    # p_i = ∂L/∂q̇_i
    p_defs = [sp.Eq(pf, sp.simplify(sp.diff(L, qd)))
              for pf, qd in zip(p_funcs, vels)]

    # Solve for q̇_i(q, p), the key step required by the Legendre transform
    legendre_eqs = [sp.diff(L, qd) - p for p, qd in zip(p_syms, vels)]
    sol_vels = sp.solve(legendre_eqs, vels, dict=True)
    if len(sol_vels) != 1:
        raise ValueError("Legendre transform failed: unable to solve qdot_i from p_i = ∂L/∂qdot_i.")
    sol_vels = sol_vels[0]

    # H = Σ p_i q̇_i - L, then substitute q̇_i -> q̇_i(q, p)
    H_raw = sum(p * qd for p, qd in zip(p_syms, vels)) - L
    H = sp.simplify(H_raw.subs(sol_vels))

    # Canonical equations
    canonical: List[sp.Eq] = []
    momentum_subs = dict(zip(p_syms, p_funcs))
    for q, p, pf in zip(coords, p_syms, p_funcs):
        canonical.append(sp.Eq(sp.diff(q, time), sp.simplify(sp.diff(H, p))))
        canonical.append(
            sp.Eq(
                sp.diff(pf, time),
                sp.simplify((-sp.diff(H, q)).subs(momentum_subs)),
            )
        )

    return {
        "L": sp.simplify(L),
        "coords": coords,
        "velocities": vels,
        "momenta": p_funcs,
        "momenta_symbols": p_syms,
        "p_defs": p_defs,
        "H": H,
        "canonical": canonical,
    }


# ---------- 1.3  Rigid-body dynamics -----------------------------------------

def inertia_tensor(params: Dict[str, Any]) -> Dict[str, Any]:
    r"""
    Compute the inertia tensor I_{ij} and diagonalize it to obtain the
    principal axes and principal moments of inertia.

    Two input modes are supported:
        (A) "particles" : discrete point masses [(m, (x, y, z)), ...]
            I_{ij} = Σ_k m_k ( δ_{ij} r_k² - r_{k,i} r_{k,j} )
        (B) "density"   : continuum body
            "density"  : ρ(x,y,z)
            "ranges"   : [(x,a,b),(y,c,d),(z,e,f)]
            I_{ij} = ∫ ρ ( δ_{ij} r² - x_i x_j ) dV

    Returns
    -------
    dict
        {
            "I"                   : sympy.Matrix 3x3,
            "principal_moments"   : [I1, I2, I3]  # eigenvalues
            "principal_axes"      : [v1, v2, v3]  # column vectors
            "diagonalization"     : (P, D)       # I = P D P^{-1}
        }
    """
    var = sp.symbols("x y z", real=True)
    x, y, z = var
    I = sp.zeros(3, 3)

    if "particles" in params:
        for m, r in params["particles"]:
            r = sp.Matrix(r)
            r2 = (r.T * r)[0, 0]
            for i in range(3):
                for j in range(3):
                    delta = 1 if i == j else 0
                    I[i, j] += m * (delta * r2 - r[i] * r[j])
    elif "density" in params:
        rho = sp.sympify(params["density"])
        ranges = params["ranges"]   # [(x,a,b),(y,c,d),(z,e,f)]
        r2 = x**2 + y**2 + z**2
        coords_xyz = [x, y, z]
        for i in range(3):
            for j in range(3):
                delta = 1 if i == j else 0
                integrand = rho * (delta * r2 - coords_xyz[i] * coords_xyz[j])
                expr = integrand
                for rng in ranges:
                    expr = sp.integrate(expr, rng)
                I[i, j] = sp.simplify(expr)
    else:
        raise ValueError("inertia_tensor requires either 'particles' or 'density' with 'ranges'.")

    I = sp.simplify(I)

    # Diagonalization / principal axes
    P, D = I.diagonalize()
    principal_moments = [sp.simplify(D[i, i]) for i in range(3)]
    principal_axes = [sp.simplify(P.col(i)) for i in range(3)]

    return {
        "I": I,
        "principal_moments": principal_moments,
        "principal_axes": principal_axes,
        "diagonalization": (P, D),
    }


def euler_rigid_body_equations(params: Dict[str, Any]) -> Dict[str, Any]:
    r"""
    Euler rigid-body equations in the principal-axis frame, with or without
    external torque:

        I1 ω̇1 + (I3 - I2) ω2 ω3 = N1
        I2 ω̇2 + (I1 - I3) ω3 ω1 = N2
        I3 ω̇3 + (I2 - I1) ω1 ω2 = N3

    Parameters
    ----------
    params : dict
        "principal_moments" : (I1, I2, I3), principal moments of inertia
        "torque" (optional)  : (N1, N2, N3), default (0, 0, 0)
        "omega_symbols"      : optional names for the three angular-velocity
                               functions, default ["omega1", "omega2", "omega3"]
        "time" (optional)    : default t
        "solve_ode" (optional): bool

    Returns
    -------
    dict
        {
            "omegas"  : [ω_i(t)],
            "eqs"     : [Eq,Eq,Eq],
            "solution": dsolve output, if solve_ode=True,
        }
    """
    time = params.get("time", t)
    I1, I2, I3 = [sp.sympify(x) for x in params["principal_moments"]]
    N = params.get("torque", (0, 0, 0))
    N1, N2, N3 = [sp.sympify(x) for x in N]

    names = params.get("omega_symbols", ["omega1", "omega2", "omega3"])
    w = [sp.Function(n)(time) for n in names]
    w1, w2, w3 = w
    wd = [sp.diff(wi, time) for wi in w]

    eqs = [
        sp.Eq(I1 * wd[0] + (I3 - I2) * w2 * w3, N1),
        sp.Eq(I2 * wd[1] + (I1 - I3) * w3 * w1, N2),
        sp.Eq(I3 * wd[2] + (I2 - I1) * w1 * w2, N3),
    ]

    out: Dict[str, Any] = {"omegas": w, "eqs": eqs}

    if params.get("solve_ode", False):
        try:
            sol = sp.dsolve(eqs, w)
            out["solution"] = sol
        except Exception as e:
            out["solution"] = f"dsolve failed: {e}"

    return out


# ==============================================================================
# Module 2  Electrodynamics
# ==============================================================================
#
#   2.1  Vector differential operators: div / curl / grad / laplacian
#        (cartesian / cylindrical / spherical)
#   2.2  Maxwell-equation checks
#   2.3  Electromagnetic potentials and gauge transforms (Coulomb / Lorenz)
#   2.4  Poynting vector and energy density
#   2.5  Electromagnetic-wave dispersion relations (vacuum / medium)
#
# ------------------------------------------------------------------------------


def _coordinate_system(name: str):
    """
    Return ``(coords, h, basis_name)``:
        coords: the three coordinate symbols (u1, u2, u3)
        h     : the Lamé scale factors (h1, h2, h3)
    """
    name = name.lower()
    if name in ("cartesian", "rect", "rectangular", "xyz"):
        x, y, z = sp.symbols("x y z", real=True)
        return (x, y, z), (sp.Integer(1), sp.Integer(1), sp.Integer(1)), "cartesian"
    if name in ("cylindrical", "cyl"):
        rho = sp.Symbol("rho", nonnegative=True)
        phi = sp.Symbol("phi", real=True)
        z = sp.Symbol("z", real=True)
        return (rho, phi, z), (sp.Integer(1), rho, sp.Integer(1)), "cylindrical"
    if name in ("spherical", "sph"):
        r = sp.Symbol("r", nonnegative=True)
        theta = sp.Symbol("theta", real=True)
        phi = sp.Symbol("phi", real=True)
        return (r, theta, phi), (sp.Integer(1), r, r * sp.sin(theta)), "spherical"
    raise ValueError(f"Unknown coordinate system: {name}")


def vector_divergence(F: Sequence[sp.Expr],
                      coord_system: str = "cartesian",
                      coords: Optional[Sequence[sp.Symbol]] = None
                      ) -> sp.Expr:
    r"""
    Divergence in orthogonal curvilinear coordinates:
        ∇·F = (1/(h1 h2 h3)) Σ ∂_i ( (h1 h2 h3 / h_i) F_i )
    """
    cs, h, _ = _coordinate_system_with_scale_factors(coord_system, coords)
    h1, h2, h3 = h
    F1, F2, F3 = [sp.sympify(f) for f in F]
    H = h1 * h2 * h3
    expr = (sp.diff(H / h1 * F1, cs[0])
            + sp.diff(H / h2 * F2, cs[1])
            + sp.diff(H / h3 * F3, cs[2])) / H
    return sp.simplify(expr)


def vector_curl(F: Sequence[sp.Expr],
                coord_system: str = "cartesian",
                coords: Optional[Sequence[sp.Symbol]] = None
                ) -> sp.Matrix:
    r"""
    Curl in orthogonal curvilinear coordinates, returned as a 3x1 sympy.Matrix.
        (∇×F)_i = (1/(h_j h_k)) [ ∂_j (h_k F_k) - ∂_k (h_j F_j) ]
    """
    cs, h, _ = _coordinate_system_with_scale_factors(coord_system, coords)
    h1, h2, h3 = h
    F1, F2, F3 = [sp.sympify(f) for f in F]
    u1, u2, u3 = cs
    c1 = (sp.diff(h3 * F3, u2) - sp.diff(h2 * F2, u3)) / (h2 * h3)
    c2 = (sp.diff(h1 * F1, u3) - sp.diff(h3 * F3, u1)) / (h3 * h1)
    c3 = (sp.diff(h2 * F2, u1) - sp.diff(h1 * F1, u2)) / (h1 * h2)
    return sp.simplify(sp.Matrix([c1, c2, c3]))


def vector_gradient(phi: sp.Expr,
                    coord_system: str = "cartesian",
                    coords: Optional[Sequence[sp.Symbol]] = None
                    ) -> sp.Matrix:
    r"""Scalar-field gradient: (∇φ)_i = (1/h_i) ∂φ/∂u_i"""
    cs, h, _ = _coordinate_system_with_scale_factors(coord_system, coords)
    phi = sp.sympify(phi)
    grads = [sp.diff(phi, u) / hi for u, hi in zip(cs, h)]
    return sp.simplify(sp.Matrix(grads))


def scalar_laplacian(phi: sp.Expr,
                     coord_system: str = "cartesian",
                     coords: Optional[Sequence[sp.Symbol]] = None
                     ) -> sp.Expr:
    r"""
    Scalar Laplacian:
        ∇²φ = (1/(h1 h2 h3)) Σ ∂_i ( (h1 h2 h3 / h_i²) ∂φ/∂u_i )
    """
    cs, h, _ = _coordinate_system_with_scale_factors(coord_system, coords)
    h1, h2, h3 = h
    H = h1 * h2 * h3
    phi = sp.sympify(phi)
    expr = sum(sp.diff(H / h[i]**2 * sp.diff(phi, cs[i]), cs[i])
               for i in range(3)) / H
    return sp.simplify(expr)


# ---------- 2.2  Maxwell equations -------------------------------------------

def maxwell_equations_check(params: Dict[str, Any]) -> Dict[str, Any]:
    r"""
    Check whether the supplied (E, B, ρ, J) satisfy Maxwell's equations in SI units:

        ∇·E  = ρ/ε₀
        ∇·B  = 0
        ∇×E  = -∂B/∂t
        ∇×B  = μ₀ J + μ₀ ε₀ ∂E/∂t

    Parameters
    ----------
    params : dict
        "E", "B"      : length-3 list or Matrix
        "rho"         : charge density (default 0)
        "J"           : current-density vector (default [0, 0, 0])
        "coord_system": default 'cartesian'
        "coords"      : optional coordinate override
        "time"        : default t
        "eps0", "mu0": default sympy.Symbol values, or concrete constants

    Returns
    -------
    dict
        {
            "gauss_E"   : residual of Eq(∇·E, ρ/ε₀),
            "gauss_B"   : residual of Eq(∇·B, 0),
            "faraday"   : three-component residual of (∇×E + ∂B/∂t),
            "ampere"    : three-component residual of
                           (∇×B - μ₀J - μ₀ε₀ ∂E/∂t),
            "satisfied" : bool, true when all simplified residuals are zero,
        }
    """
    cs_name = params.get("coord_system", "cartesian")
    coords = params.get("coords")
    time = params.get("time", t)
    eps0 = params.get("eps0", sp.Symbol("epsilon_0", positive=True))
    mu0 = params.get("mu0", sp.Symbol("mu_0", positive=True))
    E = [sp.sympify(e) for e in params["E"]]
    B = [sp.sympify(b) for b in params["B"]]
    rho = sp.sympify(params.get("rho", 0))
    J = [sp.sympify(j) for j in params.get("J", [0, 0, 0])]

    divE = vector_divergence(E, cs_name, coords)
    divB = vector_divergence(B, cs_name, coords)
    curlE = vector_curl(E, cs_name, coords)
    curlB = vector_curl(B, cs_name, coords)

    dBdt = sp.Matrix([sp.diff(b, time) for b in B])
    dEdt = sp.Matrix([sp.diff(e, time) for e in E])

    gauss_E = sp.simplify(divE - rho / eps0)
    gauss_B = sp.simplify(divB)
    faraday = sp.simplify(curlE + dBdt)
    ampere = sp.simplify(curlB - mu0 * sp.Matrix(J) - mu0 * eps0 * dEdt)

    satisfied = (
        _expr_is_zero(gauss_E)
        and _expr_is_zero(gauss_B)
        and _matrix_is_zero(faraday)
        and _matrix_is_zero(ampere)
    )

    return {
        "gauss_E": gauss_E,
        "gauss_B": gauss_B,
        "faraday": faraday,
        "ampere": ampere,
        "satisfied": bool(satisfied),
    }


# ---------- 2.3  Electromagnetic potentials & gauge transforms ---------------

def fields_from_potentials(params: Dict[str, Any]) -> Dict[str, Any]:
    r"""
    Compute (E, B) from (φ, A):
        E = -∇φ - ∂A/∂t
        B =  ∇×A
    Also check the Coulomb gauge (∇·A = 0) and Lorenz gauge
    (∇·A + (1/c²) ∂φ/∂t = 0).

    Parameters
    ----------
    params : dict
        "phi"          : scalar potential φ
        "A"            : vector potential [A1, A2, A3]
        "coord_system" : default 'cartesian'
        "coords"       : optional
        "time"         : default t
        "c"            : speed-of-light symbol, default sympy.Symbol('c')

    Returns
    -------
    dict
        {
            "E", "B"           : sympy.Matrix(3,1),
            "div_A"            : ∇·A (Coulomb-gauge condition),
            "lorenz_residual"  : ∇·A + (1/c²) ∂φ/∂t,
            "coulomb_gauge"    : bool,
            "lorenz_gauge"     : bool,
        }
    """
    cs_name = params.get("coord_system", "cartesian")
    coords = params.get("coords")
    time = params.get("time", t)
    c = params.get("c", sp.Symbol("c", positive=True))

    phi = sp.sympify(params["phi"])
    A = [sp.sympify(a) for a in params["A"]]

    grad_phi = vector_gradient(phi, cs_name, coords)
    dAdt = sp.Matrix([sp.diff(a, time) for a in A])
    E = sp.simplify(-grad_phi - dAdt)
    B = vector_curl(A, cs_name, coords)

    divA = vector_divergence(A, cs_name, coords)
    lorenz_res = sp.simplify(divA + sp.diff(phi, time) / c**2)

    return {
        "E": E,
        "B": B,
        "div_A": sp.simplify(divA),
        "lorenz_residual": lorenz_res,
        "coulomb_gauge": _expr_is_zero(divA),
        "lorenz_gauge": _expr_is_zero(lorenz_res),
    }


# ---------- 2.4  Poynting vector ---------------------------------------------

def poynting_vector(params: Dict[str, Any]) -> Dict[str, Any]:
    r"""
    Poynting vector and electromagnetic energy density in SI units:

        S = (1/μ₀) E × B
        u = (ε₀/2) |E|² + (1/(2μ₀)) |B|²

    Parameters
    ----------
    params : dict
        "E", "B"      : length 3
        "mu0", "eps0" : optional
    """
    mu0 = params.get("mu0", sp.Symbol("mu_0", positive=True))
    eps0 = params.get("eps0", sp.Symbol("epsilon_0", positive=True))
    E = sp.Matrix([sp.sympify(e) for e in params["E"]])
    B = sp.Matrix([sp.sympify(b) for b in params["B"]])

    S = sp.simplify(E.cross(B) / mu0)
    u = sp.simplify(eps0 / 2 * (E.dot(E)) + 1 / (2 * mu0) * (B.dot(B)))
    return {"S": S, "u": u}


# ---------- 2.5  Electromagnetic-wave dispersion -----------------------------

def em_wave_dispersion(params: Dict[str, Any]) -> Dict[str, Any]:
    r"""
    Electromagnetic-wave dispersion relations:
        vacuum:      ω² = c² k²
        dielectric:  ω² = k² / (μ ε), with the nonmagnetic form ω = c k / n,
                     n = √(ε_r μ_r)
        lossy case:  k² = μ ε ω² + i μ σ ω  (conducting medium)

    Parameters
    ----------
    params : dict
        "medium": "vacuum" | "dielectric" | "conductor"
        "c"    : speed-of-light symbol, default Symbol('c')
        "eps"  : permittivity (dielectric / conductor)
        "mu"   : permeability (dielectric / conductor)
        "sigma": conductivity (conductor)
        "omega": angular frequency (required for conductor)
    """
    medium = params.get("medium", "vacuum")
    c = params.get("c", sp.Symbol("c", positive=True))
    omega = params.get("omega", sp.Symbol("omega", positive=True))
    k = sp.Symbol("k", positive=True)

    if medium == "vacuum":
        rel = sp.Eq(omega**2, c**2 * k**2)
        v_phase = c
        n = sp.Integer(1)
    elif medium == "dielectric":
        eps = sp.sympify(params["eps"])
        mu = sp.sympify(params["mu"])
        rel = sp.Eq(omega**2, k**2 / (mu * eps))
        v_phase = 1 / sp.sqrt(mu * eps)
        eps0 = sp.Symbol("epsilon_0", positive=True)
        mu0 = sp.Symbol("mu_0", positive=True)
        n = sp.sqrt((eps / eps0) * (mu / mu0))
    elif medium == "conductor":
        eps = sp.sympify(params["eps"])
        mu = sp.sympify(params["mu"])
        sigma = sp.sympify(params["sigma"])
        rel = sp.Eq(k**2, mu * eps * omega**2 + sp.I * mu * sigma * omega)
        v_phase = omega / sp.sqrt(mu * eps * omega**2 + sp.I * mu * sigma * omega)
        n = None
    else:
        raise ValueError(f"Unknown medium: {medium}")

    return {
        "dispersion": rel,
        "k_solution": sp.solve(rel, k),
        "phase_velocity": sp.simplify(v_phase),
        "refractive_index": n,
    }


# ==============================================================================
# Module 3  Quantum Mechanics
# ==============================================================================
#
#   3.1  Operator commutators [A, B]
#   3.2  1D time-independent Schrödinger equation Ĥψ = Eψ
#        - automatic support for infinite wells / harmonic oscillators /
#          free particles / arbitrary V(x)
#   3.3  Pauli matrices & spin algebra
#   3.4  Joint eigenstates of L̂² and L̂_z (spherical harmonics)
#   3.5  First-order nondegenerate stationary perturbation theory
#
# ------------------------------------------------------------------------------


# ---------- 3.1  Operator commutators ----------------------------------------

def commutator(A: Any, B: Any, simplify_result: bool = True) -> Any:
    r"""
    Compute the commutator [A, B] = AB - BA.

    Supported inputs:
        * sympy.Matrix    (finite-dimensional operators)
        * sympy expressions (symbolic algebra)
        * sympy.physics.quantum operators

    Parameters
    ----------
    A, B : Matrix / Expr / Operator
    """
    if isinstance(A, sp.MatrixBase) or isinstance(B, sp.MatrixBase):
        result = A * B - B * A
        return sp.simplify(result) if simplify_result else result

    # quantum module support
    try:
        from sympy.physics.quantum import Commutator
        c = Commutator(A, B).doit()
        return sp.simplify(c) if simplify_result else c
    except Exception:
        pass

    result = A * B - B * A
    return sp.simplify(result) if simplify_result else result


# ---------- 3.2  1D time-independent Schrödinger equation --------------------

def schrodinger_1d(params: Dict[str, Any]) -> Dict[str, Any]:
    r"""
    Solve the one-dimensional time-independent Schrödinger equation:

        -ℏ²/(2m) ψ''(x) + V(x) ψ(x) = E ψ(x)

    Parameters
    ----------
    params : dict
        "potential" : str | sympy.Expr
            Optional preset keys:
                "infinite_well"  : infinite square well on [0, L], requires "L"
                "harmonic"       : 1D harmonic oscillator V = 1/2 m ω² x²,
                                   requires "omega"
                "free"           : free particle, V = 0
            Or pass a direct sympy expression V(x)
        "x"     (optional): coordinate symbol, default Symbol('x')
        "m"     (optional): mass
        "hbar"  (optional): reduced Planck constant
        "n_max" (optional): number of returned eigenstates for the harmonic
                            oscillator or well, default 4
        "ics"   (optional): initial conditions for dsolve in the custom-potential case

    Returns
    -------
    dict
        {
            "potential"      : V(x),
            "equation"       : sp.Eq,
            "eigenvalues"    : [E_n, ...] if an analytic spectrum exists,
            "eigenfunctions" : [ψ_n(x), ...], normalized when available,
            "general_solution": dsolve general solution for a custom potential,
        }
    """
    x = params.get("x", sp.Symbol("x", real=True))
    m = params.get("m", sp.Symbol("m", positive=True))
    hbar = params.get("hbar", sp.Symbol("hbar", positive=True))
    n_max = params.get("n_max", 4)

    psi = sp.Function("psi")
    E = sp.Symbol("E", real=True)

    pot = params["potential"]

    # ------- Preset potentials -------
    if isinstance(pot, str):
        key = pot.lower()
        if key == "infinite_well":
            L = sp.sympify(params["L"])
            V_expr = sp.Integer(0)
            eq = sp.Eq(-hbar**2 / (2 * m) * sp.diff(psi(x), x, 2)
                       + V_expr * psi(x), E * psi(x))
            n = sp.Symbol("n", positive=True, integer=True)
            E_n_sym = (n**2 * sp.pi**2 * hbar**2) / (2 * m * L**2)
            psi_n_sym = sp.sqrt(2 / L) * sp.sin(n * sp.pi * x / L)
            eigvals = [E_n_sym.subs(n, k) for k in range(1, n_max + 1)]
            eigfuncs = [psi_n_sym.subs(n, k) for k in range(1, n_max + 1)]
            return {
                "potential": V_expr,
                "equation": eq,
                "eigenvalues_general": E_n_sym,
                "eigenfunctions_general": psi_n_sym,
                "eigenvalues": eigvals,
                "eigenfunctions": eigfuncs,
                "domain": (0, L),
            }

        if key == "harmonic":
            omega = sp.sympify(params.get("omega", sp.Symbol("omega", positive=True)))
            V_expr = sp.Rational(1, 2) * m * omega**2 * x**2
            eq = sp.Eq(-hbar**2 / (2 * m) * sp.diff(psi(x), x, 2)
                       + V_expr * psi(x), E * psi(x))
            n = sp.Symbol("n", nonnegative=True, integer=True)
            E_n_sym = hbar * omega * (n + sp.Rational(1, 2))
            xi = sp.sqrt(m * omega / hbar) * x
            # ψ_n = (mω/πℏ)^(1/4) / √(2^n n!) · H_n(ξ) · exp(-ξ²/2)
            from sympy.functions.special.polynomials import hermite
            psi_n_sym = ((m * omega / (sp.pi * hbar))**(sp.Rational(1, 4))
                         / sp.sqrt(2**n * sp.factorial(n))
                         * hermite(n, xi) * sp.exp(-xi**2 / 2))
            eigvals = [sp.simplify(E_n_sym.subs(n, k)) for k in range(n_max)]
            eigfuncs = [sp.simplify(psi_n_sym.subs(n, k)) for k in range(n_max)]
            return {
                "potential": V_expr,
                "equation": eq,
                "eigenvalues_general": E_n_sym,
                "eigenfunctions_general": psi_n_sym,
                "eigenvalues": eigvals,
                "eigenfunctions": eigfuncs,
            }

        if key == "free":
            V_expr = sp.Integer(0)
            eq = sp.Eq(-hbar**2 / (2 * m) * sp.diff(psi(x), x, 2), E * psi(x))
            sol = sp.dsolve(eq, psi(x))
            k = sp.sqrt(2 * m * E) / hbar
            return {
                "potential": V_expr,
                "equation": eq,
                "general_solution": sol,
                "k": k,
            }

        raise ValueError(f"Unknown preset potential: {pot}")

    # ------- Arbitrary V(x): return the equation and try for a general solution -------
    V_expr = sp.sympify(pot)
    eq = sp.Eq(-hbar**2 / (2 * m) * sp.diff(psi(x), x, 2)
               + V_expr * psi(x), E * psi(x))
    out: Dict[str, Any] = {"potential": V_expr, "equation": eq}
    try:
        ics = params.get("ics", None)
        sol = sp.dsolve(eq, psi(x), ics=ics)
        out["general_solution"] = sol
    except Exception as e:
        out["general_solution"] = f"dsolve failed: {e}"
    return out


# ---------- 3.3  Pauli matrices ----------------------------------------------

def pauli_matrices() -> Dict[str, sp.Matrix]:
    r"""
    Return the Pauli matrices σ_x, σ_y, σ_z and the identity matrix I₂.
    """
    sx = sp.Matrix([[0, 1], [1, 0]])
    sy = sp.Matrix([[0, -sp.I], [sp.I, 0]])
    sz = sp.Matrix([[1, 0], [0, -1]])
    I2 = sp.eye(2)
    return {"sigma_x": sx, "sigma_y": sy, "sigma_z": sz, "I": I2}


def pauli_algebra(params: Dict[str, Any]) -> Dict[str, Any]:
    r"""
    Pauli-matrix algebra utilities:
        * arbitrary linear combinations a·σ = a_x σ_x + a_y σ_y + a_z σ_z
        * commutators [σ_i, σ_j] and anticommutators {σ_i, σ_j}
        * spin eigenvalues and eigenvectors

    Parameters
    ----------
    params : dict
        "vector" (optional): (a_x, a_y, a_z) to compute a·σ and its eigenvalues
    """
    P = pauli_matrices()
    sx, sy, sz = P["sigma_x"], P["sigma_y"], P["sigma_z"]
    out: Dict[str, Any] = {"pauli": P}

    out["commutators"] = {
        "[sx,sy]": commutator(sx, sy),
        "[sy,sz]": commutator(sy, sz),
        "[sz,sx]": commutator(sz, sx),
    }
    out["anticommutators"] = {
        "{sx,sy}": sp.simplify(sx * sy + sy * sx),
        "{sy,sz}": sp.simplify(sy * sz + sz * sy),
        "{sz,sx}": sp.simplify(sz * sx + sx * sz),
        "{sx,sx}": sp.simplify(sx * sx + sx * sx),
    }

    if "vector" in params:
        ax, ay, az = [sp.sympify(c) for c in params["vector"]]
        M = ax * sx + ay * sy + az * sz
        out["a_dot_sigma"] = sp.simplify(M)
        eig = M.eigenvects()
        out["eigen"] = eig
    return out


# ---------- 3.4  Angular momentum L̂², L̂_z -----------------------------------

def angular_momentum_eigenstates(params: Dict[str, Any]) -> Dict[str, Any]:
    r"""
    The spherical harmonics Y_l^m(θ,φ) are joint eigenstates of L̂² and L̂_z:
        L̂² Y_l^m = ℏ² l(l+1) Y_l^m
        L̂_z Y_l^m = ℏ m Y_l^m

    Parameters
    ----------
    params : dict
        "l", "m"  : integers with |m| ≤ l
        "hbar"    : optional

    Returns
    -------
    dict
        {
            "Y_lm"        : spherical harmonic in simplified sympy.Ynm form,
            "L2_eigenvalue": ℏ² l(l+1),
            "Lz_eigenvalue": ℏ m,
            "L2_check"    : operator-action result for verification,
            "Lz_check"    : operator-action result for verification,
        }
    """
    l = sp.Integer(params["l"])
    m_q = sp.Integer(params["m"])
    hbar = params.get("hbar", sp.Symbol("hbar", positive=True))
    theta, phi = sp.symbols("theta phi", real=True)

    from sympy.functions.special.spherical_harmonics import Ynm
    Y = Ynm(l, m_q, theta, phi).expand(func=True)

    # L_z = -i ℏ ∂/∂φ
    Lz_Y = sp.simplify(-sp.I * hbar * sp.diff(Y, phi))
    # L² = -ℏ² [ 1/sinθ ∂/∂θ(sinθ ∂/∂θ) + 1/sin²θ ∂²/∂φ² ]
    L2_Y = sp.simplify(
        -hbar**2 * (
            sp.diff(sp.sin(theta) * sp.diff(Y, theta), theta) / sp.sin(theta)
            + sp.diff(Y, phi, 2) / sp.sin(theta)**2
        )
    )

    return {
        "Y_lm": Y,
        "L2_eigenvalue": hbar**2 * l * (l + 1),
        "Lz_eigenvalue": hbar * m_q,
        "L2_check": L2_Y,
        "Lz_check": Lz_Y,
    }


# ---------- 3.5  Nondegenerate stationary perturbation theory ----------------

def perturbation_first_order(params: Dict[str, Any]) -> Dict[str, Any]:
    r"""
    First-order nondegenerate stationary perturbation theory:
        E_n^(1) = ⟨n^(0) | Ĥ' | n^(0)⟩

    Parameters
    ----------
    params : dict
        "H_prime"          : perturbing Hamiltonian (Expr or Matrix)
        "psi0"             : unperturbed eigenfunction (Expr) or eigenvector (Matrix);
                             may be a single state or a list of states
        "variable"         : integration variable (required in Expr mode)
        "domain"           : integration interval (a, b), required in Expr mode
        "weight" (optional): volume-element weight function, default 1
        "matrix_mode"      : optional explicit matrix-mode flag
                             (automatic by default)

    Returns
    -------
    dict
        {
            "E1": scalar (single state) or list (multiple states),
        }
    """
    H1 = params["H_prime"]
    psi0 = params["psi0"]

    # ---- Matrix mode ----
    if isinstance(H1, sp.MatrixBase):
        states = psi0 if isinstance(psi0, list) else [psi0]
        E1_list = []
        for v in states:
            v = sp.Matrix(v)
            num = (v.H * H1 * v)[0, 0]
            den = (v.H * v)[0, 0]
            E1_list.append(sp.simplify(num / den))
        return {"E1": E1_list[0] if len(E1_list) == 1 else E1_list}

    # ---- Function mode ----
    var = params["variable"]
    a, b = params["domain"]
    w = sp.sympify(params.get("weight", 1))
    states = psi0 if isinstance(psi0, list) else [psi0]
    E1_list = []
    for psi in states:
        psi = sp.sympify(psi)
        psi_c = sp.conjugate(psi)
        num = sp.integrate(psi_c * H1 * psi * w, (var, a, b))
        den = sp.integrate(psi_c * psi * w, (var, a, b))
        E1_list.append(sp.simplify(num / den))
    return {"E1": E1_list[0] if len(E1_list) == 1 else E1_list}


# ==============================================================================
# Module 4  Thermodynamics & Statistical Mechanics
# ==============================================================================
#
#   4.1  Thermodynamic potentials and Maxwell relations
#   4.2  Canonical partition function Z = Σ exp(-β E_i)
#   4.3  Derive macroscopic quantities U, F, S, C_v from Z
#   4.4  The three standard distributions: MB / FD / BE
#
# ------------------------------------------------------------------------------


def thermodynamic_potentials(params: Dict[str, Any]) -> Dict[str, Any]:
    r"""
    The four thermodynamic potentials, their natural variables, and the
    associated Maxwell relations:

        U(S,V) :  dU =  T dS - p dV
        H(S,p) =  U + p V
        F(T,V) =  U - T S       (Helmholtz)
        G(T,p) =  H - T S       (Gibbs)

    The Maxwell relations follow from the symmetry of d²Φ:
        ( ∂T/∂V )_S = -( ∂p/∂S )_V
        ( ∂T/∂p )_S =  ( ∂V/∂S )_p
        ( ∂S/∂V )_T =  ( ∂p/∂T )_V
        ( ∂S/∂p )_T = -( ∂V/∂T )_p

    Parameters
    ----------
    params : dict (optional)
        "U" : internal-energy expression U(S, V), used to derive T, p, etc.
        "F" : Helmholtz free energy F(T, V)
        "G" : Gibbs free energy G(T, p)
        "H" : enthalpy H(S, p)

    Returns
    -------
    dict
        {
            "potentials"      : { "U":..., "H":..., "F":..., "G":... },
            "maxwell_relations": [Eq,Eq,Eq,Eq],
            "derived"         : { automatically derived T, p, S, V, μ, etc. },
        }
    """
    S, V, T, p = sp.symbols("S V T p", positive=True)

    out: Dict[str, Any] = {}
    out["natural_variables"] = {
        "U": (S, V), "H": (S, p), "F": (T, V), "G": (T, p),
    }

    # Generic Maxwell relations represented symbolically
    out["maxwell_relations"] = [
        sp.Eq(sp.Symbol("(∂T/∂V)_S"), -sp.Symbol("(∂p/∂S)_V")),
        sp.Eq(sp.Symbol("(∂T/∂p)_S"),  sp.Symbol("(∂V/∂S)_p")),
        sp.Eq(sp.Symbol("(∂S/∂V)_T"),  sp.Symbol("(∂p/∂T)_V")),
        sp.Eq(sp.Symbol("(∂S/∂p)_T"), -sp.Symbol("(∂V/∂T)_p")),
    ]

    derived: Dict[str, Any] = {}

    if "U" in params:
        U = sp.sympify(params["U"])
        derived["T_from_U"] = sp.simplify(sp.diff(U, S))     # T = (∂U/∂S)_V
        derived["p_from_U"] = sp.simplify(-sp.diff(U, V))    # p = -(∂U/∂V)_S
    if "F" in params:
        F = sp.sympify(params["F"])
        derived["S_from_F"] = sp.simplify(-sp.diff(F, T))    # S = -(∂F/∂T)_V
        derived["p_from_F"] = sp.simplify(-sp.diff(F, V))    # p = -(∂F/∂V)_T
    if "G" in params:
        G = sp.sympify(params["G"])
        derived["S_from_G"] = sp.simplify(-sp.diff(G, T))    # S = -(∂G/∂T)_p
        derived["V_from_G"] = sp.simplify(sp.diff(G, p))     # V =  (∂G/∂p)_T
    if "H" in params:
        H = sp.sympify(params["H"])
        derived["T_from_H"] = sp.simplify(sp.diff(H, S))     # T =  (∂H/∂S)_p
        derived["V_from_H"] = sp.simplify(sp.diff(H, p))     # V =  (∂H/∂p)_S

    out["derived"] = derived
    out["symbols"] = {"S": S, "V": V, "T": T, "p": p}
    return out


def thermodynamic_partial(params: Dict[str, Any]) -> sp.Expr:
    r"""
    Automatically compute the chained partial derivative using the Jacobian identity:
        ( ∂X / ∂Y )_Z

    Given state equations or potentials X(a,b), Y(a,b), Z(a,b), where a and b
    are independent variables:

        (∂X/∂Y)_Z = ∂(X,Z)/∂(Y,Z)
                  = [ X_a Z_b - X_b Z_a ] / [ Y_a Z_b - Y_b Z_a ]

    Parameters
    ----------
    params : dict
        "X", "Y", "Z" : expressions of (a, b)
        "a", "b"       : independent-variable symbols
    """
    X = sp.sympify(params["X"])
    Y = sp.sympify(params["Y"])
    Z = sp.sympify(params["Z"])
    a = params["a"]; b = params["b"]
    num = sp.diff(X, a) * sp.diff(Z, b) - sp.diff(X, b) * sp.diff(Z, a)
    den = sp.diff(Y, a) * sp.diff(Z, b) - sp.diff(Y, b) * sp.diff(Z, a)
    return sp.simplify(num / den)


# ---------- 4.2 / 4.3  Partition function and macroscopic observables --------

def partition_function(params: Dict[str, Any]) -> Dict[str, Any]:
    r"""
    Canonical-ensemble partition function and the macroscopic quantities derived from it:

        Z   = Σ_i g_i exp(-β E_i),   β = 1/(k_B T)
        F   = -k_B T ln Z
        U   = -∂(ln Z)/∂β
        S   = -(∂F/∂T)_V = k_B (ln Z + β U)
        C_v =  (∂U/∂T)_V

    Parameters
    ----------
    params : dict
        "energies"    : list [E_i] for a finite spectrum,
                         or ("expr", n, (a, b)) to sum over index n
        "degeneracies": same length as energies, default all 1
        "kB"          : Boltzmann-constant symbol, default Symbol('k_B')
        "T"           : temperature symbol, default Symbol('T')
        "beta"        : β symbol, default 1/(k_B T)

    Returns
    -------
    dict {Z, lnZ, F, U, S, Cv, beta}
    """
    kB = params.get("kB", sp.Symbol("k_B", positive=True))
    T = params.get("T", sp.Symbol("T", positive=True))
    beta = params.get("beta", 1 / (kB * T))

    energies = params["energies"]
    if isinstance(energies, tuple) and energies[0] == "expr":
        # Form: ("expr", E_n, (n, a, b))
        _, E_n, (n_sym, a, b) = energies
        deg = params.get("degeneracies", 1)
        boltzmann_weight = deg * sp.exp(-beta * E_n)
        Z = sp.summation(boltzmann_weight, (n_sym, a, b))
        U_num = sp.summation(E_n * boltzmann_weight, (n_sym, a, b))
    else:
        deg = params.get("degeneracies", [1] * len(energies))
        boltzmann_terms = [g * sp.exp(-beta * E) for g, E in zip(deg, energies)]
        Z = sum(boltzmann_terms)
        U_num = sum(g * E * sp.exp(-beta * E) for g, E in zip(deg, energies))

    Z = sp.simplify(Z)
    lnZ = sp.simplify(sp.log(Z))

    F = sp.simplify(-lnZ / beta)
    U = sp.simplify(U_num / Z)
    S = sp.simplify(beta * (U - F))
    if hasattr(beta, "free_symbols") and T in beta.free_symbols:
        Cv = sp.simplify(sp.diff(U, T))
    else:
        Cv = sp.simplify(-kB * beta**2 * sp.diff(U, beta))

    return {
        "beta": beta,
        "Z": Z,
        "lnZ": lnZ,
        "F": F,
        "U": U,
        "S": S,
        "Cv": Cv,
    }


# ---------- 4.4  Statistical distributions -----------------------------------

def statistical_distributions(params: Dict[str, Any]) -> Dict[str, Any]:
    r"""
    The three standard statistical distributions for the single-particle
    occupation number ⟨n(ε)⟩:

        Maxwell-Boltzmann :  exp[-(ε - μ)/(k_B T)]
        Fermi-Dirac       :  1 / ( exp[(ε - μ)/(k_B T)] + 1 )
        Bose-Einstein     :  1 / ( exp[(ε - μ)/(k_B T)] - 1 )

    Parameters
    ----------
    params : dict (all optional)
        "epsilon": energy symbol (default Symbol('epsilon'))
        "mu"     : chemical potential (default Symbol('mu'))
        "kB", "T": default Symbol('k_B'), Symbol('T')
        "which"  : "MB" | "FD" | "BE" | "all" (default "all")
    """
    eps = params.get("epsilon", sp.Symbol("epsilon", real=True))
    mu = params.get("mu", sp.Symbol("mu", real=True))
    kB = params.get("kB", sp.Symbol("k_B", positive=True))
    T = params.get("T", sp.Symbol("T", positive=True))
    which = params.get("which", "all").upper()

    arg = (eps - mu) / (kB * T)
    f_MB = sp.exp(-arg)
    f_FD = 1 / (sp.exp(arg) + 1)
    f_BE = 1 / (sp.exp(arg) - 1)

    table = {"MB": f_MB, "FD": f_FD, "BE": f_BE}
    if which == "ALL":
        return {"distributions": table, "argument": arg}
    return {"distribution": table[which], "argument": arg}


# ==============================================================================
# Module 5  Special Relativity
# ==============================================================================
#
#   5.1  Lorentz transformation matrix Λ^μ_ν
#        (arbitrary-direction boost / x-axis boost)
#   5.2  Transformations of spacetime events (ct, x, y, z)
#   5.3  Four-vector inner products (Minkowski metric, signature (+,-,-,-))
#   5.4  Four-momentum p^μ = (E/c, p) and E² = (pc)² + (mc²)²
#   5.5  Velocity addition / γ-factor utilities
#
# ------------------------------------------------------------------------------


# Minkowski metric (+,-,-,-)
_MINKOWSKI = sp.diag(1, -1, -1, -1)


def _gamma(beta_expr: sp.Expr) -> sp.Expr:
    return 1 / sp.sqrt(1 - beta_expr**2)


def lorentz_boost_matrix(params: Dict[str, Any]) -> Dict[str, Any]:
    r"""
    Construct the Lorentz boost matrix Λ^μ_ν (4x4).

    Parameters
    ----------
    params : dict
        Mode A: boost along the x axis
            "v"       : velocity
            "c"       : speed of light (default Symbol('c'))
        Mode B: boost in an arbitrary direction
            "velocity": (vx, vy, vz)
            "c"       : speed of light

    Returns
    -------
    dict
        {
            "Lambda" : sympy.Matrix(4,4),
            "gamma"  : γ,
            "beta"   : β, either scalar or vector,
        }
    """
    c = params.get("c", sp.Symbol("c", positive=True))

    if "v" in params:
        v = sp.sympify(params["v"])
        beta = v / c
        g = _gamma(beta)
        L = sp.Matrix([
            [g,        -g * beta, 0, 0],
            [-g * beta, g,        0, 0],
            [0,         0,        1, 0],
            [0,         0,        0, 1],
        ])
        return {"Lambda": sp.simplify(L), "gamma": sp.simplify(g), "beta": beta}

    # Arbitrary direction
    vx, vy, vz = [sp.sympify(x) for x in params["velocity"]]
    beta_vec = sp.Matrix([vx, vy, vz]) / c
    b2 = (beta_vec.T * beta_vec)[0, 0]
    g = 1 / sp.sqrt(1 - b2)

    n = sp.Matrix([vx, vy, vz])
    n_norm2 = (n.T * n)[0, 0]

    # Λ^0_0 = γ
    # Λ^0_i = -γ β_i
    # Λ^i_0 = -γ β_i
    # Λ^i_j = δ_ij + (γ-1) β_i β_j / β²
    L = sp.zeros(4, 4)
    L[0, 0] = g
    for i in range(3):
        L[0, i + 1] = -g * beta_vec[i]
        L[i + 1, 0] = -g * beta_vec[i]
    for i in range(3):
        for j in range(3):
            kron = 1 if i == j else 0
            extra = sp.Piecewise(
                ((g - 1) * beta_vec[i] * beta_vec[j] / b2, b2 != 0),
                (0, True),
            )
            L[i + 1, j + 1] = kron + extra

    return {"Lambda": sp.simplify(L),
            "gamma": sp.simplify(g),
            "beta": beta_vec}


def lorentz_transform_event(params: Dict[str, Any]) -> Dict[str, Any]:
    r"""
    Transform the spacetime event X^μ = (ct, x, y, z) into frame S' using Λ:
        X'^μ = Λ^μ_ν X^ν

    Parameters
    ----------
    params : dict
        "event"  : (ct, x, y, z), a 4-element list
        plus the fields required by lorentz_boost_matrix ("v" or "velocity", and "c")

    Returns
    -------
    dict { "Lambda", "X", "X_prime" }
    """
    boost = lorentz_boost_matrix(params)
    L = boost["Lambda"]
    X = sp.Matrix([sp.sympify(c) for c in params["event"]])
    Xp = sp.simplify(L * X)
    return {"Lambda": L, "X": X, "X_prime": Xp,
            "gamma": boost["gamma"], "beta": boost["beta"]}


def four_vector_inner_product(A: Sequence[sp.Expr],
                              B: Sequence[sp.Expr]) -> sp.Expr:
    r"""
    Minkowski inner product of two four-vectors (signature +,-,-,-):
        A · B = A^0 B^0 - A^1 B^1 - A^2 B^2 - A^3 B^3
    """
    A = sp.Matrix([sp.sympify(a) for a in A])
    B = sp.Matrix([sp.sympify(b) for b in B])
    return sp.simplify((A.T * _MINKOWSKI * B)[0, 0])


def relativistic_energy_momentum(params: Dict[str, Any]) -> Dict[str, Any]:
    r"""
    Relativistic energy-momentum relation:

        E² = (pc)² + (mc²)²
        p^μ = (E/c, p_x, p_y, p_z),  p_μ p^μ = (mc)²

    Parameters
    ----------
    params : dict
        "m"     : rest mass
        "c"     : default Symbol('c')
        Plus one of the following groups to solve for the missing quantity:
            ("E",) solves for |p|
            ("p",) or ("px", "py", "pz") solves for E
            ("v",) provides velocity and solves for E, p, γ
        "solve_for": optional explicit target "E" | "p"

    Returns
    -------
        dict { "E", "p", "p_vec" (if available), "p_mu" (four-momentum),
            "invariant", "gamma" (if available) }
    """
    c = params.get("c", sp.Symbol("c", positive=True))
    m = sp.sympify(params.get("m", sp.Symbol("m", nonnegative=True)))

    out: Dict[str, Any] = {}

    if "v" in params or "velocity" in params:
        if "v" in params:
            v = sp.sympify(params["v"])
            beta = v / c
            g = _gamma(beta)
            p_mag = sp.simplify(g * m * v)
            E = sp.simplify(g * m * c**2)
            out.update({"E": E, "p": p_mag, "gamma": g})
            out["p_mu"] = sp.Matrix([E / c, p_mag, 0, 0])
        else:
            vx, vy, vz = [sp.sympify(x) for x in params["velocity"]]
            v_vec = sp.Matrix([vx, vy, vz])
            v2 = (v_vec.T * v_vec)[0, 0]
            beta2 = v2 / c**2
            g = 1 / sp.sqrt(1 - beta2)
            p_vec = sp.simplify(g * m * v_vec)
            E = sp.simplify(g * m * c**2)
            out.update({"E": E, "p_vec": p_vec, "gamma": g,
                        "p": sp.simplify(sp.sqrt((p_vec.T * p_vec)[0, 0]))})
            out["p_mu"] = sp.Matrix([E / c, p_vec[0], p_vec[1], p_vec[2]])
        out["invariant"] = sp.simplify((m * c)**2)
        return out

    if "E" in params and "p" not in params and "px" not in params:
        E = sp.sympify(params["E"])
        p_sol = sp.solve(sp.Eq(E**2, (sp.Symbol("p", nonnegative=True) * c)**2
                                + (m * c**2)**2),
                         sp.Symbol("p", nonnegative=True))
        out["E"] = E
        out["p"] = [sp.simplify(s) for s in p_sol]
    elif "p" in params or "px" in params:
        if "px" in params:
            p_vec = sp.Matrix([sp.sympify(params.get(k, 0))
                               for k in ("px", "py", "pz")])
            p_mag = sp.sqrt((p_vec.T * p_vec)[0, 0])
            out["p_vec"] = p_vec
        else:
            p_mag = sp.sympify(params["p"])
        E = sp.simplify(sp.sqrt((p_mag * c)**2 + (m * c**2)**2))
        out["E"] = E
        out["p"] = sp.simplify(p_mag)
        if "p_vec" in out:
            out["p_mu"] = sp.Matrix([E / c,
                                     out["p_vec"][0],
                                     out["p_vec"][1],
                                     out["p_vec"][2]])
    else:
        raise ValueError("relativistic_energy_momentum requires one of 'E', 'p', or 'v'.")

    if "p_mu" in out:
        out["invariant"] = sp.simplify(four_vector_inner_product(out["p_mu"], out["p_mu"]))
    else:
        out["invariant"] = sp.simplify((m * c)**2)
    return out


def velocity_addition(params: Dict[str, Any]) -> Dict[str, Any]:
    r"""
    Relativistic velocity addition along the x axis:
        u' = (u - v) / (1 - u v / c²)
    Or inversely:
        u  = (u' + v) / (1 + u' v / c²)

    Parameters
    ----------
    params : dict
        "u" or "u_prime" : velocity
        "v"               : relative frame velocity
        "c"               : default Symbol('c')
    """
    c = params.get("c", sp.Symbol("c", positive=True))
    v = sp.sympify(params["v"])
    if "u" in params:
        u = sp.sympify(params["u"])
        return {"u_prime": sp.simplify((u - v) / (1 - u * v / c**2))}
    if "u_prime" in params:
        up = sp.sympify(params["u_prime"])
        return {"u": sp.simplify((up + v) / (1 + up * v / c**2))}
    raise ValueError("velocity_addition requires either 'u' or 'u_prime'.")


# ==============================================================================
# Module 6  Optics & Waves
# ==============================================================================
#
#   6.1  Multi-slit interference intensity distribution
#   6.2  Grating equation d sinθ = n λ
#   6.3  Single-slit Fraunhofer diffraction
#   6.4  Matrix optics: translation / refraction / thin-lens / mirror matrices;
#        system effective focal length and principal planes
#
# ------------------------------------------------------------------------------


def multi_slit_intensity(params: Dict[str, Any]) -> Dict[str, Any]:
    r"""
    Fraunhofer interference for multiple slits (N slits), modulated by the
    single-slit diffraction envelope:

        I(θ) = I_0 [ sin(α)/α ]² · [ sin(N β)/sin(β) ]²

        α = (π a sinθ) / λ        (single-slit diffraction phase)
        β = (π d sinθ) / λ        (half the phase difference between adjacent slits)

    Parameters
    ----------
    params : dict
        "N"      : number of slits
        "d"      : slit spacing
        "a"      : single-slit width (set to 0 to ignore the diffraction envelope)
        "wavelength" or "lam" : λ
        "theta"  : angle symbol (default Symbol('theta'))
        "I0"     : central intensity, default Symbol('I_0')
    """
    N = sp.sympify(params["N"])
    d = sp.sympify(params["d"])
    a = sp.sympify(params.get("a", 0))
    lam = sp.sympify(params.get("wavelength", params.get("lam")))
    theta = params.get("theta", sp.Symbol("theta", real=True))
    I0 = params.get("I0", sp.Symbol("I_0", positive=True))

    beta = sp.pi * d * sp.sin(theta) / lam
    interference_amplitude = sp.sinc(N * beta / sp.pi) / sp.sinc(beta / sp.pi)
    interference = sp.simplify(interference_amplitude**2)

    if a == 0:
        I = I0 * interference
        alpha = None
        diffraction = sp.Integer(1)
    else:
        alpha = sp.pi * a * sp.sin(theta) / lam
        diffraction = sp.simplify(sp.sinc(alpha / sp.pi)**2)
        I = sp.simplify(I0 * diffraction * interference)

    return {
        "I": sp.simplify(I),
        "alpha": alpha,
        "beta": beta,
        "diffraction_envelope": diffraction,
        "interference_factor": sp.simplify(interference),
    }


def grating_equation(params: Dict[str, Any]) -> Dict[str, Any]:
    r"""
    Grating equation d sinθ = n λ.
    Given any three quantities, solve automatically for the fourth.

    Parameters
    ----------
    params : dict (three known quantities)
        "d", "wavelength"/"lam", "n", "theta"
    """
    d = params.get("d", sp.Symbol("d", positive=True))
    lam = params.get("wavelength", params.get("lam", sp.Symbol("lambda", positive=True)))
    n = params.get("n", sp.Symbol("n", integer=True))
    theta = params.get("theta", sp.Symbol("theta", real=True))

    eq = sp.Eq(sp.sympify(d) * sp.sin(sp.sympify(theta)),
               sp.sympify(n) * sp.sympify(lam))

    out: Dict[str, Any] = {"equation": eq, "solution": None, "solved_for": None}
    provided = {
        "d": "d" in params,
        "lam": "lam" in params or "wavelength" in params,
        "n": "n" in params,
        "theta": "theta" in params,
    }
    unknowns = [
        s for key, s in (("d", d), ("lam", lam), ("n", n), ("theta", theta))
        if not provided[key]
    ]
    target = params.get("solve_for", unknowns[0] if len(unknowns) == 1 else None)
    if target is not None:
        if isinstance(target, str):
            target_map = {"d": d, "lam": lam, "wavelength": lam, "n": n, "theta": theta}
            target = target_map[target]
        try:
            out["solution"] = sp.solve(eq, target)
            out["solved_for"] = target
        except Exception as e:
            out["solution"] = f"solve failed: {e}"
    return out


def single_slit_diffraction(params: Dict[str, Any]) -> Dict[str, Any]:
    r"""
    Single-slit Fraunhofer diffraction intensity:

        I(θ) = I_0 [ sin(α)/α ]²,    α = π a sinθ / λ

    Minima condition: a sinθ = m λ  (m = ±1, ±2, ...)
    """
    a = sp.sympify(params["a"])
    lam = sp.sympify(params.get("wavelength", params.get("lam")))
    theta = params.get("theta", sp.Symbol("theta", real=True))
    I0 = params.get("I0", sp.Symbol("I_0", positive=True))
    alpha = sp.pi * a * sp.sin(theta) / lam
    I = sp.simplify(I0 * sp.sinc(alpha / sp.pi)**2)
    m = sp.Symbol("m", integer=True, nonzero=True)
    minima = sp.Eq(a * sp.sin(theta), m * lam)
    return {"I": sp.simplify(I), "alpha": alpha, "minima_condition": minima}


# ---------- 6.4  Matrix optics (ABCD matrices) -------------------------------

def ray_translation_matrix(d: sp.Expr) -> sp.Matrix:
    r"""Free-space translation matrix T(d) = [[1, d], [0, 1]]."""
    return sp.Matrix([[1, sp.sympify(d)], [0, 1]])


def ray_refraction_matrix(n1: sp.Expr, n2: sp.Expr,
                          R: Optional[sp.Expr] = None) -> sp.Matrix:
    r"""
    Refraction matrix for a spherical interface from medium n1 to n2 with
    radius of curvature R:

        [[ 1,                 0      ],
         [ -(n2-n1)/(n2 R),  n1/n2  ]]

    R = ∞ reduces to planar refraction with only the n1/n2 scaling term.
    """
    n1 = sp.sympify(n1); n2 = sp.sympify(n2)
    if R is None or R == sp.oo:
        return sp.Matrix([[1, 0], [0, n1 / n2]])
    R = sp.sympify(R)
    return sp.Matrix([[1, 0],
                      [-(n2 - n1) / (n2 * R), n1 / n2]])


def thin_lens_matrix(f: sp.Expr) -> sp.Matrix:
    r"""Thin-lens matrix L(f) = [[1, 0], [-1/f, 1]]."""
    f = sp.sympify(f)
    return sp.Matrix([[1, 0], [-1 / f, 1]])


def mirror_matrix(R: sp.Expr) -> sp.Matrix:
    r"""Spherical-mirror matrix with the convention R > 0 for a concave focusing mirror: [[1, 0], [-2/R, 1]]."""
    R = sp.sympify(R)
    return sp.Matrix([[1, 0], [-2 / R, 1]])


def optical_system(params: Dict[str, Any]) -> Dict[str, Any]:
    r"""
    Multi-element paraxial optical system: provide the element sequence in the
    direction of propagation and build the system matrix M by repeated right multiplication.

    Element descriptions (each list item is a dict):
        {"type":"translation", "d":...}
        {"type":"refraction",  "n1":..., "n2":..., "R":...(optional; omitted means planar)}
        {"type":"thin_lens",   "f":...}
        {"type":"mirror",      "R":...}
        {"type":"matrix",      "M": <2x2 sympy.Matrix>}

    The system matrix follows optical order: M = M_k ... M_2 M_1.
    Return the effective focal length and the equivalent principal-plane parameters:

        If M = [[A, B], [C, D]]
                        * Effective focal length:             f_eff = -1/C  when n_in = n_out
                        * Front principal plane H from input: x_H  = (D - 1)/C
                        * Back principal plane H' from output: x_H' = (1 - A)/C
                            (positive toward the output side)
                        * Front focal point from input:       x_F  = D/C
                        * Back focal point from output:       x_F' = -A/C

    Parameters
    ----------
    params : dict
        "elements": the element list described above
    """
    elements = params["elements"]
    M = sp.eye(2)
    for el in elements:
        kind = el["type"].lower()
        if kind == "translation":
            Mi = ray_translation_matrix(el["d"])
        elif kind == "refraction":
            Mi = ray_refraction_matrix(el["n1"], el["n2"], el.get("R", sp.oo))
        elif kind == "thin_lens":
            Mi = thin_lens_matrix(el["f"])
        elif kind == "mirror":
            Mi = mirror_matrix(el["R"])
        elif kind == "matrix":
            Mi = sp.Matrix(el["M"])
        else:
            raise ValueError(f"Unknown optical element: {kind}")
        M = Mi * M

    M = sp.simplify(M)
    A, B = M[0, 0], M[0, 1]
    C, D = M[1, 0], M[1, 1]

    out: Dict[str, Any] = {"M": M, "A": A, "B": B, "C": C, "D": D}
    out["det"] = sp.simplify(A * D - B * C)
    if C != 0:
        out["f_eff"] = sp.simplify(-1 / C)
        out["x_H"] = sp.simplify((D - 1) / C)
        out["x_H_prime"] = sp.simplify((1 - A) / C)
        out["x_F"] = sp.simplify(D / C)
        out["x_F_prime"] = sp.simplify(-A / C)
    else:
        out["f_eff"] = sp.oo
        out["note"] = "C = 0: afocal system with no finite focal length."
    return out


# ==============================================================================
# Module 7  Extended Utilities
# ==============================================================================
#
#   7.1  Noether conserved quantities
#   7.2  Effective-potential analysis
#   7.3  Special-function toolbox
#   7.4  Error propagation
#   7.5  Dimensional analysis (Buckingham Π)
#   7.6  Thick lenses and aberrations
#   7.7  Polarization optics (Jones / Stokes-Mueller)
#   7.8  Classical Doppler shift and standing-wave modes
#
# ------------------------------------------------------------------------------


def _total_time_derivative(
    expr: sp.Expr,
    coords: Sequence[sp.Expr],
    velocities: Sequence[sp.Expr],
    time: sp.Symbol,
) -> sp.Expr:
    """Take the total time derivative of expr(q_i(t), t)."""
    total = sp.diff(expr, time)
    for q, qd in zip(coords, velocities):
        total += sp.diff(expr, q) * qd
    return sp.simplify(total)


def noether_conservation(params: Dict[str, Any]) -> Dict[str, Any]:
    r"""
    Automatically identify conserved quantities from a Lagrangian.

    Implemented cases:
        1. Cyclic coordinates: if ∂L/∂q_i = 0, then p_i = ∂L/∂q̇_i is conserved.
        2. Time translation: if ∂L/∂t = 0, then the energy
           h = Σ q̇_i ∂L/∂q̇_i - L is conserved.
        3. User-supplied point-transformation generators X_i(q,t):
              δq_i = ε X_i,
              and verify δL = Σ (∂L/∂q_i) X_i + (∂L/∂q̇_i) dX_i/dt.

    Parameters
    ----------
    params : dict
        "L"        : Lagrangian
        "coords"   : generalized coordinates
        "time"     : optional; inferred from L / coords when omitted
        "symmetry" : optional list [X_1, ..., X_n]
    """
    time = params.get("time")
    if time is None:
        time = _infer_time_symbol([params.get("L"), *params.get("coords", [])], t)

    coords = _as_functions_of_t(params["coords"], time)
    velocities = [_qdot(q, time) for q in coords]
    L = sp.sympify(params["L"])

    cyclic_coordinates = []
    canonical_momenta = []
    for q, qd in zip(coords, velocities):
        momentum = sp.simplify(sp.diff(L, qd))
        canonical_momenta.append(momentum)
        if _expr_is_zero(sp.diff(L, q)):
            cyclic_coordinates.append({
                "coordinate": q,
                "momentum": momentum,
            })

    energy = None
    if _expr_is_zero(_explicit_time_derivative(L, coords, velocities, time)):
        energy = sp.simplify(sum(qd * sp.diff(L, qd) for qd in velocities) - L)

    custom_symmetry = None
    if "symmetry" in params:
        generators = [sp.sympify(xi) for xi in params["symmetry"]]
        if len(generators) != len(coords):
            raise ValueError("symmetry generators must match the number of coordinates.")

        delta_L = sp.Integer(0)
        charge = sp.Integer(0)
        for q, qd, Xi in zip(coords, velocities, generators):
            dXi_dt = _total_time_derivative(Xi, coords, velocities, time)
            delta_L += sp.diff(L, q) * Xi + sp.diff(L, qd) * dXi_dt
            charge += Xi * sp.diff(L, qd)

        custom_symmetry = {
            "delta_L": sp.simplify(delta_L),
            "conserved_charge": sp.simplify(charge),
            "is_symmetry": _expr_is_zero(delta_L),
        }

    return {
        "canonical_momenta": canonical_momenta,
        "cyclic_coordinates": cyclic_coordinates,
        "energy": energy,
        "custom_symmetry": custom_symmetry,
    }


def effective_potential_analysis(params: Dict[str, Any]) -> Dict[str, Any]:
    r"""
    Analyze a one-dimensional effective potential U(x): equilibrium points,
    stability, small-oscillation frequencies, and turning points.
    """
    x = params["x"]
    U = sp.sympify(params["U"])
    m = sp.sympify(params.get("m", sp.Symbol("m", positive=True)))

    U_prime = sp.simplify(sp.diff(U, x))
    U_double_prime = sp.simplify(sp.diff(U, x, 2))
    equilibrium_points = sp.solve(sp.Eq(U_prime, 0), x)

    equilibria = []
    for x_eq in equilibrium_points:
        curvature = sp.simplify(U_double_prime.subs(x, x_eq))
        if curvature.is_positive is True:
            stability = "stable"
            omega = sp.simplify(sp.sqrt(curvature / m))
        elif curvature.is_negative is True:
            stability = "unstable"
            omega = sp.simplify(sp.sqrt(curvature / m))
        else:
            stability = "critical/undetermined"
            omega = None
        equilibria.append({
            "x_eq": x_eq,
            "U_eq": sp.simplify(U.subs(x, x_eq)),
            "curvature": curvature,
            "stability": stability,
            "small_oscillation_omega": omega,
        })

    out = {
        "U_prime": U_prime,
        "U_double_prime": U_double_prime,
        "equilibria": equilibria,
    }
    if "E" in params:
        E = sp.sympify(params["E"])
        out["turning_points"] = sp.solve(sp.Eq(U, E), x)
    return out


def special_functions(params: Dict[str, Any]) -> Dict[str, Any]:
    r"""
    Unified wrapper for common special functions.
    """
    from sympy.functions.special.polynomials import (
        assoc_laguerre,
        assoc_legendre,
        chebyshevt,
        chebyshevu,
        hermite,
        laguerre,
        legendre,
    )
    from sympy.functions.special.bessel import besselj, bessely
    from sympy.functions.special.spherical_harmonics import Ynm

    name = params["name"].lower()
    if name == "legendre":
        expr = legendre(params["n"], params["x"])
    elif name == "assoc_legendre":
        expr = assoc_legendre(params["n"], params["m"], params["x"])
    elif name == "bessel_j":
        expr = besselj(params["n"], params["x"])
    elif name == "bessel_y":
        expr = bessely(params["n"], params["x"])
    elif name == "laguerre":
        expr = laguerre(params["n"], params["x"])
    elif name == "assoc_laguerre":
        expr = assoc_laguerre(params["n"], params["alpha"], params["x"])
    elif name == "hermite":
        expr = hermite(params["n"], params["x"])
    elif name == "chebyshev_t":
        expr = chebyshevt(params["n"], params["x"])
    elif name == "chebyshev_u":
        expr = chebyshevu(params["n"], params["x"])
    elif name == "ynm":
        theta = params.get("theta", sp.Symbol("theta", real=True))
        phi = params.get("phi", sp.Symbol("phi", real=True))
        expr = Ynm(params["l"], params["m"], theta, phi).expand(func=True)
    else:
        raise ValueError(f"Unknown special function: {name}")

    return {"name": name, "expr": sp.simplify(expr)}


def error_propagation(params: Dict[str, Any]) -> Dict[str, Any]:
    r"""
    First-order Gaussian error propagation.
    """
    f = sp.sympify(params["f"])
    vars_ = list(params["vars"])
    sigmas = [sp.sympify(sigma) for sigma in params["sigmas"]]
    covariances = params.get("covariances", {})

    partials = {var: sp.simplify(sp.diff(f, var)) for var in vars_}
    sigma_squared = sum((partials[var] * sigma) ** 2 for var, sigma in zip(vars_, sigmas))
    for (var_i, var_j), covariance in covariances.items():
        sigma_squared += 2 * partials[var_i] * partials[var_j] * sp.sympify(covariance)
    sigma_squared = sp.simplify(sigma_squared)

    out = {
        "partials": partials,
        "sigma_f_squared": sigma_squared,
        "sigma_f": sp.simplify(sp.sqrt(sigma_squared)),
    }

    if "values" in params:
        subs_map = params["values"]
        out["value"] = {
            "f": sp.simplify(f.subs(subs_map)),
            "sigma_f": sp.simplify(out["sigma_f"].subs(subs_map)),
        }

    return out


def _dimension_powers(expr: Any) -> Dict[Any, sp.Expr]:
    if isinstance(expr, str):
        powers: Dict[Any, sp.Expr] = {}
        for token in expr.replace("^", "**").split():
            if "**" in token:
                base, exponent = token.split("**", 1)
                powers[sp.Symbol(base)] = sp.sympify(exponent)
            else:
                symbol = sp.Symbol(token)
                powers[symbol] = powers.get(symbol, 0) + 1
        return powers

    from sympy.physics.units.quantities import Quantity
    from sympy.physics.units.systems.si import dimsys_SI

    sym_expr = sp.sympify(expr)
    converted = sym_expr.replace(
        lambda item: isinstance(item, Quantity),
        lambda item: item.dimension,
    )

    try:
        dependencies = dimsys_SI.get_dimensional_dependencies(converted)
        return {base: sp.simplify(exponent) for base, exponent in dependencies.items()}
    except Exception:
        return {base: sp.simplify(exponent) for base, exponent in converted.as_powers_dict().items()}


def dimensional_analysis(params: Dict[str, Any]) -> Dict[str, Any]:
    r"""
    Buckingham Π theorem: construct dimensionless groups from the null space of
    the dimensional-exponent matrix.
    """
    quantities = params["quantities"]
    names = list(quantities.keys())
    power_maps = {name: _dimension_powers(dim_expr) for name, dim_expr in quantities.items()}

    base_dimensions = params.get("base")
    if base_dimensions is None:
        base_set = set()
        for power_map in power_maps.values():
            base_set.update(power_map.keys())
        base_dimensions = sorted(base_set, key=lambda item: str(item))

    exponent_matrix = sp.Matrix(
        [[power_maps[name].get(base, 0) for name in names] for base in base_dimensions]
    )

    nullspace = exponent_matrix.nullspace()
    pi_groups = []
    for index, vector in enumerate(nullspace, start=1):
        pi_expr = sp.Integer(1)
        for name, exponent in zip(names, vector):
            pi_expr *= sp.Symbol(name) ** sp.simplify(exponent)
        pi_groups.append(sp.Eq(sp.Symbol(f"Pi_{index}"), sp.simplify(pi_expr)))

    return {
        "base_dimensions": base_dimensions,
        "exponent_matrix": exponent_matrix,
        "nullspace": nullspace,
        "pi_groups": pi_groups,
        "n_pi": len(nullspace),
    }


def thick_lens(params: Dict[str, Any]) -> Dict[str, Any]:
    r"""
    Thick-lens ABCD matrix and effective focal length.
    """
    n0 = sp.sympify(params.get("n0", 1))
    n = sp.sympify(params["n"])
    R1 = sp.sympify(params["R1"])
    R2 = sp.sympify(params["R2"])
    d = sp.sympify(params["d"])

    M_front = ray_refraction_matrix(n0, n, R1)
    M_bulk = ray_translation_matrix(d)
    M_back = ray_refraction_matrix(n, n0, R2)
    M = sp.simplify(M_back * M_bulk * M_front)
    A, B = M[0, 0], M[0, 1]
    C, D = M[1, 0], M[1, 1]

    out = {
        "M": M,
        "A": A,
        "B": B,
        "C": C,
        "D": D,
        "optical_power": sp.simplify(-C),
    }
    if not _expr_is_zero(C):
        out["f_eff"] = sp.simplify(-1 / C)
        out["principal_plane_front"] = sp.simplify((D - 1) / C)
        out["principal_plane_back"] = sp.simplify((1 - A) / C)
    else:
        out["f_eff"] = sp.oo
    return out


def aberrations(params: Dict[str, Any]) -> Dict[str, Any]:
    r"""
    Paraxial approximations for spherical and chromatic aberration.
    """
    f = sp.sympify(params["f"])
    h = sp.sympify(params.get("h", sp.Symbol("h", positive=True)))
    shape_factor = sp.sympify(params.get("K", 1))
    abbe = sp.sympify(params.get("V", sp.Symbol("V_d", positive=True)))
    y = sp.sympify(params.get("y", sp.Symbol("y", real=True)))

    return {
        "spherical_aberration": sp.simplify(-h ** 2 * shape_factor / (2 * f)),
        "axial_chromatic_aberration": sp.simplify(-f / abbe),
        "lateral_chromatic_aberration": sp.simplify(y / abbe),
    }


def jones_calculus(params: Dict[str, Any]) -> Dict[str, Any]:
    r"""
    Jones-vector and Jones-matrix utilities.
    """
    operation = params["operation"].lower()

    def rotation(angle: sp.Expr) -> sp.Matrix:
        return sp.Matrix([
            [sp.cos(angle), sp.sin(angle)],
            [-sp.sin(angle), sp.cos(angle)],
        ])

    if operation == "vector_linear":
        theta = sp.sympify(params["theta"])
        return {"jones_vector": sp.Matrix([sp.cos(theta), sp.sin(theta)])}
    if operation == "vector_rcp":
        return {"jones_vector": sp.Matrix([1, -sp.I]) / sp.sqrt(2)}
    if operation == "vector_lcp":
        return {"jones_vector": sp.Matrix([1, sp.I]) / sp.sqrt(2)}
    if operation == "polarizer":
        theta = sp.sympify(params.get("theta", 0))
        matrix = rotation(-theta) * sp.Matrix([[1, 0], [0, 0]]) * rotation(theta)
        return {"jones_matrix": sp.simplify(matrix)}
    if operation == "waveplate":
        theta = sp.sympify(params.get("theta", 0))
        phase = sp.sympify(params["phi"])
        matrix = rotation(-theta) * sp.Matrix([[1, 0], [0, sp.exp(sp.I * phase)]]) * rotation(theta)
        return {"jones_matrix": sp.simplify(matrix)}
    if operation == "apply":
        state = sp.Matrix(params["input"])
        matrices = params["matrices"] if "matrices" in params else [params["matrix"]]
        for matrix in matrices:
            state = sp.Matrix(matrix) * state
        intensity = sp.simplify((state.H * state)[0, 0])
        return {"output": sp.simplify(state), "intensity": intensity}
    raise ValueError(f"Unknown jones operation: {operation}")


def stokes_mueller(params: Dict[str, Any]) -> Dict[str, Any]:
    r"""
    Stokes-vector and Mueller-matrix utilities.
    """
    operation = params["operation"].lower()

    if operation == "from_jones":
        ex, ey = [sp.sympify(component) for component in params["jones"]]
        s0 = sp.simplify(ex * sp.conjugate(ex) + ey * sp.conjugate(ey))
        s1 = sp.simplify(ex * sp.conjugate(ex) - ey * sp.conjugate(ey))
        s2 = sp.simplify(ex * sp.conjugate(ey) + sp.conjugate(ex) * ey)
        s3 = sp.simplify(-sp.I * (ex * sp.conjugate(ey) - sp.conjugate(ex) * ey))
        return {"stokes": sp.Matrix([s0, s1, s2, s3])}

    if operation == "polarizer":
        theta = sp.sympify(params.get("theta", 0))
        c2 = sp.cos(2 * theta)
        s2 = sp.sin(2 * theta)
        matrix = sp.Rational(1, 2) * sp.Matrix([
            [1, c2, s2, 0],
            [c2, c2 ** 2, c2 * s2, 0],
            [s2, c2 * s2, s2 ** 2, 0],
            [0, 0, 0, 0],
        ])
        return {"mueller": sp.simplify(matrix)}

    if operation == "waveplate":
        theta = sp.sympify(params.get("theta", 0))
        phase = sp.sympify(params["phi"])
        c2 = sp.cos(2 * theta)
        s2 = sp.sin(2 * theta)
        cp = sp.cos(phase)
        sp_phase = sp.sin(phase)
        matrix = sp.Matrix([
            [1, 0, 0, 0],
            [0, c2 ** 2 + s2 ** 2 * cp, c2 * s2 * (1 - cp), -s2 * sp_phase],
            [0, c2 * s2 * (1 - cp), s2 ** 2 + c2 ** 2 * cp, c2 * sp_phase],
            [0, s2 * sp_phase, -c2 * sp_phase, cp],
        ])
        return {"mueller": sp.simplify(matrix)}

    if operation == "apply":
        matrix = sp.Matrix(params["mueller"])
        stokes = sp.Matrix(params["stokes"])
        return {"stokes_out": sp.simplify(matrix * stokes)}

    raise ValueError(f"Unknown stokes_mueller operation: {operation}")


def doppler_classical(params: Dict[str, Any]) -> Dict[str, Any]:
    r"""
    Classical Doppler shift:
        f_obs = f_src (v + v_obs) / (v - v_src)

    Convention: v_obs > 0 means the observer moves toward the source;
    v_src > 0 means the source moves toward the observer.
    """
    f_src = sp.sympify(params["f_src"])
    wave_speed = sp.sympify(params["v"])
    v_obs = sp.sympify(params.get("v_obs", 0))
    v_src = sp.sympify(params.get("v_src", 0))
    f_obs = sp.simplify(f_src * (wave_speed + v_obs) / (wave_speed - v_src))
    return {
        "f_obs": f_obs,
        "shift": sp.simplify(f_obs - f_src),
        "ratio": sp.simplify(f_obs / f_src),
    }


def standing_wave_modes(params: Dict[str, Any]) -> Dict[str, Any]:
    r"""
    One-dimensional standing-wave modes: fixed-fixed / open-open / open-closed.
    """
    boundary = params["boundary"].lower()
    length = sp.sympify(params["L"])
    wave_speed = sp.sympify(params["v"])
    n_max = params.get("n_max", 4)
    n = sp.Symbol("n", positive=True, integer=True)

    if boundary in ("fixed-fixed", "open-open"):
        wavelength_general = 2 * length / n
        frequency_general = n * wave_speed / (2 * length)
        indices = range(1, n_max + 1)
    elif boundary in ("open-closed", "closed-open"):
        wavelength_general = 4 * length / (2 * n - 1)
        frequency_general = (2 * n - 1) * wave_speed / (4 * length)
        indices = range(1, n_max + 1)
    else:
        raise ValueError(f"Unknown boundary condition: {boundary}")

    modes = [
        {
            "n": index,
            "wavelength": sp.simplify(wavelength_general.subs(n, index)),
            "frequency": sp.simplify(frequency_general.subs(n, index)),
        }
        for index in indices
    ]

    return {
        "wavelength_general": wavelength_general,
        "frequency_general": frequency_general,
        "modes": modes,
        "fundamental": modes[0],
    }


# ==============================================================================
# Module 8  Fluid Mechanics
# ==============================================================================
#
#   8.1  Continuity equation
#   8.2  Bernoulli equation
#   8.3  Euler / Navier-Stokes checks
#   8.4  Vorticity and stream function
#   8.5  Reynolds number
#   8.6  Hagen-Poiseuille flow
#   8.7  Stokes drag
#   8.8  Speed of sound
#   8.9  Surface tension / capillary rise
#
# ------------------------------------------------------------------------------


def continuity_equation(params: Dict[str, Any]) -> Dict[str, Any]:
    r"""
    Continuity equation:
        ∂ρ/∂t + ∇·(ρ v) = 0
    Or, in the incompressible case:
        ∇·v = 0
    """
    coord_system = params.get("coord_system", "cartesian")
    coords = params.get("coords")
    time = params.get("time", t)
    rho = sp.sympify(params["rho"])
    velocity = [sp.sympify(component) for component in params["v"]]

    div_v = sp.simplify(vector_divergence(velocity, coord_system, coords))
    if params.get("incompressible", False):
        return {
            "div_v": div_v,
            "residual": div_v,
            "satisfied": _expr_is_zero(div_v),
        }

    mass_flux = [rho * component for component in velocity]
    residual = sp.simplify(sp.diff(rho, time) + vector_divergence(mass_flux, coord_system, coords))
    return {
        "div_v": div_v,
        "residual": residual,
        "satisfied": _expr_is_zero(residual),
    }


def bernoulli_equation(params: Dict[str, Any]) -> Dict[str, Any]:
    r"""
    Bernoulli equation:
        1/2 ρ v^2 + ρ g h + p = const
    """
    rho = sp.sympify(params.get("rho", sp.Symbol("rho", positive=True)))
    g = sp.sympify(params.get("g", sp.Symbol("g", positive=True)))

    d = {
        "v1": sp.sympify(params.get("v1", sp.Symbol("v_1", positive=True))),
        "h1": sp.sympify(params.get("h1", sp.Symbol("h_1", real=True))),
        "p1": sp.sympify(params.get("p1", sp.Symbol("p_1", real=True))),
        "v2": sp.sympify(params.get("v2", sp.Symbol("v_2", positive=True))),
        "h2": sp.sympify(params.get("h2", sp.Symbol("h_2", real=True))),
        "p2": sp.sympify(params.get("p2", sp.Symbol("p_2", real=True))),
    }

    equation = sp.Eq(
        sp.Rational(1, 2) * rho * d["v1"] ** 2 + rho * g * d["h1"] + d["p1"],
        sp.Rational(1, 2) * rho * d["v2"] ** 2 + rho * g * d["h2"] + d["p2"],
    )

    out = {"equation": equation, "solution": None, "solved_for": None}
    if "solve_for" in params:
        target = params["solve_for"]
        if isinstance(target, str):
            target = d[target]
        out["solved_for"] = target
        out["solution"] = sp.solve(equation, target)
        return out

    missing = [name for name in ("v1", "h1", "p1", "v2", "h2", "p2") if name not in params]
    if len(missing) == 1:
        out["solved_for"] = d[missing[0]]
        out["solution"] = sp.solve(equation, d[missing[0]])
    return out


def euler_fluid_equation(params: Dict[str, Any]) -> Dict[str, Any]:
    r"""
    Residual of the inviscid Euler equation.

    The current strict implementation supports Cartesian coordinates only; all
    other coordinate systems are rejected explicitly to avoid misleading results.
    """
    coord_system = params.get("coord_system", "cartesian")
    if coord_system != "cartesian":
        raise NotImplementedError("euler_fluid_equation currently supports only Cartesian coordinates.")

    coords = params.get("coords", _coordinate_system("cartesian")[0])
    time = params.get("time", t)
    rho = sp.sympify(params["rho"])
    pressure = sp.sympify(params["p"])
    velocity = [sp.sympify(component) for component in params["v"]]
    body_force = [sp.sympify(component) for component in params.get("g", [0, 0, 0])]

    grad_p = vector_gradient(pressure, "cartesian", coords)
    acceleration = []
    for i in range(3):
        convective = sum(velocity[j] * sp.diff(velocity[i], coords[j]) for j in range(3))
        acceleration.append(sp.diff(velocity[i], time) + convective)

    residual = sp.Matrix([
        sp.simplify(rho * acceleration[i] + grad_p[i] - rho * body_force[i])
        for i in range(3)
    ])
    return {
        "residual": residual,
        "satisfied": _matrix_is_zero(residual),
    }


def navier_stokes_check(params: Dict[str, Any]) -> Dict[str, Any]:
    r"""
    Residual of the incompressible Navier-Stokes equation.

    The current strict implementation supports Cartesian coordinates only.
    """
    coord_system = params.get("coord_system", "cartesian")
    if coord_system != "cartesian":
        raise NotImplementedError("navier_stokes_check currently supports only Cartesian coordinates.")

    coords = params.get("coords", _coordinate_system("cartesian")[0])
    time = params.get("time", t)
    rho = sp.sympify(params["rho"])
    mu = sp.sympify(params["mu"])
    pressure = sp.sympify(params["p"])
    velocity = [sp.sympify(component) for component in params["v"]]
    body_force = [sp.sympify(component) for component in params.get("g", [0, 0, 0])]

    div_v = sp.simplify(vector_divergence(velocity, "cartesian", coords))
    grad_p = vector_gradient(pressure, "cartesian", coords)
    momentum_residual = []
    for i in range(3):
        convective = sum(velocity[j] * sp.diff(velocity[i], coords[j]) for j in range(3))
        laplacian = scalar_laplacian(velocity[i], "cartesian", coords)
        residual_i = rho * (sp.diff(velocity[i], time) + convective)
        residual_i += grad_p[i] - mu * laplacian - rho * body_force[i]
        momentum_residual.append(sp.simplify(residual_i))
    momentum_residual_matrix = sp.Matrix(momentum_residual)

    return {
        "incompressibility": div_v,
        "momentum_residual": momentum_residual_matrix,
        "satisfied": _expr_is_zero(div_v) and _matrix_is_zero(momentum_residual_matrix),
    }


def vorticity_and_stream(params: Dict[str, Any]) -> Dict[str, Any]:
    r"""
    Utilities for vorticity and the 2D stream function.
    """
    if "psi" in params:
        x = params.get("x", sp.Symbol("x", real=True))
        y = params.get("y", sp.Symbol("y", real=True))
        psi = sp.sympify(params["psi"])
        vx = sp.simplify(sp.diff(psi, y))
        vy = sp.simplify(-sp.diff(psi, x))
        omega_z = sp.simplify(sp.diff(vy, x) - sp.diff(vx, y))
        div_v = sp.simplify(sp.diff(vx, x) + sp.diff(vy, y))
        return {
            "v": sp.Matrix([vx, vy, 0]),
            "vorticity_z": omega_z,
            "div_v": div_v,
            "incompressible": _expr_is_zero(div_v),
        }

    coord_system = params.get("coord_system", "cartesian")
    coords = params.get("coords")
    velocity = [sp.sympify(component) for component in params["v"]]
    return {
        "vorticity": vector_curl(velocity, coord_system, coords),
        "div_v": vector_divergence(velocity, coord_system, coords),
    }


def reynolds_number(params: Dict[str, Any]) -> Dict[str, Any]:
    r"""
    Reynolds number: Re = ρ v L / μ = v L / ν
    """
    velocity = sp.sympify(params["v"])
    length = sp.sympify(params["L"])
    if "nu" in params:
        reynolds = sp.simplify(velocity * length / sp.sympify(params["nu"]))
    else:
        rho = sp.sympify(params["rho"])
        mu = sp.sympify(params["mu"])
        reynolds = sp.simplify(rho * velocity * length / mu)

    laminar_limit, turbulent_limit = params.get("regime_thresholds", (2300, 4000))
    regime = "symbolic"
    if reynolds.is_number:
        reynolds_value = float(reynolds)
        if reynolds_value < laminar_limit:
            regime = "laminar"
        elif reynolds_value > turbulent_limit:
            regime = "turbulent"
        else:
            regime = "transitional"

    return {
        "Re": reynolds,
        "regime": regime,
        "thresholds": {"laminar_below": laminar_limit, "turbulent_above": turbulent_limit},
    }


def poiseuille_flow(params: Dict[str, Any]) -> Dict[str, Any]:
    r"""
    Hagen-Poiseuille laminar flow in a circular pipe.
    """
    delta_p = sp.sympify(params.get("dP", params.get("delta_p")))
    mu = sp.sympify(params["mu"])
    length = sp.sympify(params["L"])
    radius = sp.sympify(params["R"])
    r_coord = params.get("r", sp.Symbol("r", nonnegative=True))

    velocity_profile = sp.simplify(delta_p * (radius ** 2 - r_coord ** 2) / (4 * mu * length))
    flow_rate = sp.simplify(sp.pi * radius ** 4 * delta_p / (8 * mu * length))
    average_velocity = sp.simplify(flow_rate / (sp.pi * radius ** 2))
    max_velocity = sp.simplify(velocity_profile.subs(r_coord, 0))
    wall_shear = sp.simplify(delta_p * radius / (2 * length))
    return {
        "velocity_profile": velocity_profile,
        "Q": flow_rate,
        "v_avg": average_velocity,
        "v_max": max_velocity,
        "wall_shear_stress": wall_shear,
    }


def stokes_drag(params: Dict[str, Any]) -> Dict[str, Any]:
    r"""
    Stokes drag and terminal velocity.
    """
    mu = sp.sympify(params["mu"])
    radius = sp.sympify(params["R"])
    out: Dict[str, Any] = {}
    if "v" in params:
        out["F_drag"] = sp.simplify(6 * sp.pi * mu * radius * sp.sympify(params["v"]))
    if "rho_s" in params and "rho_f" in params:
        g = sp.sympify(params.get("g", sp.Symbol("g", positive=True)))
        rho_s = sp.sympify(params["rho_s"])
        rho_f = sp.sympify(params["rho_f"])
        out["v_terminal"] = sp.simplify(2 * g * radius ** 2 * (rho_s - rho_f) / (9 * mu))
    return out


def sound_speed(params: Dict[str, Any]) -> Dict[str, Any]:
    r"""
    Speed of sound for an ideal gas or a fluid/solid bulk-modulus model.
    """
    medium = params.get("medium", "ideal_gas").lower()
    if medium == "ideal_gas":
        gamma = sp.sympify(params.get("gamma", sp.Symbol("gamma", positive=True)))
        if "T" in params and "M" in params:
            gas_constant = sp.sympify(params.get("R", sp.Symbol("R", positive=True)))
            temperature = sp.sympify(params["T"])
            molar_mass = sp.sympify(params["M"])
            return {"c": sp.simplify(sp.sqrt(gamma * gas_constant * temperature / molar_mass))}
        pressure = sp.sympify(params["p"])
        density = sp.sympify(params["rho"])
        return {"c": sp.simplify(sp.sqrt(gamma * pressure / density))}
    if medium == "fluid":
        bulk_modulus = sp.sympify(params["K"])
        density = sp.sympify(params["rho"])
        return {"c": sp.simplify(sp.sqrt(bulk_modulus / density))}
    raise ValueError(f"Unknown medium: {medium}")


def surface_tension(params: Dict[str, Any]) -> Dict[str, Any]:
    r"""
    Young-Laplace and Jurin capillary-rise formulas.
    """
    operation = params["operation"].lower()
    if operation == "laplace_sphere":
        sigma = sp.sympify(params["sigma"])
        radius = sp.sympify(params["R"])
        return {"delta_p": sp.simplify(2 * sigma / radius)}
    if operation == "laplace_general":
        sigma = sp.sympify(params["sigma"])
        radius_1 = sp.sympify(params["R1"])
        radius_2 = sp.sympify(params["R2"])
        return {"delta_p": sp.simplify(sigma * (1 / radius_1 + 1 / radius_2))}
    if operation == "capillary_rise":
        sigma = sp.sympify(params["sigma"])
        theta = sp.sympify(params["theta"])
        density = sp.sympify(params["rho"])
        g = sp.sympify(params.get("g", sp.Symbol("g", positive=True)))
        radius = sp.sympify(params["r"])
        return {"h": sp.simplify(2 * sigma * sp.cos(theta) / (density * g * radius))}
    raise ValueError(f"Unknown operation: {operation}")
