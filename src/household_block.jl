"""
    household_block.jl

Solves the household consumption-savings problem via the
Endogenous Grid Method (EGM, Carroll 2006).

State space: (a, y) where a ∈ A (asset grid), y ∈ Y (income grid).

Problem:
    V(a,y) = max_{c,a'} u(c) + β Σ_{y'} Π(y,y') V(a',y')
    s.t.  c + a' = (1 + r) a + w·y
          a' ≥ ā,   c ≥ 0

EGM inverts the Euler equation:
    u'(c) = β(1+r) Σ_{y'} Π(y,y') u'(c'(a',y'))
"""

using LinearAlgebra
using Interpolations

# ---------------------------------------------------------------------------
# Utility
# ---------------------------------------------------------------------------

@inline u(c::Float64, γ::Float64) = γ == 1.0 ? log(c) : c^(1-γ) / (1-γ)
@inline u′(c::Float64, γ::Float64) = c^(-γ)
@inline u′_inv(x::Float64, γ::Float64) = x^(-1/γ)

# ---------------------------------------------------------------------------
# EGM: one backward step
# ---------------------------------------------------------------------------

"""
    egm_step(c_next, p, grids, r, w)

Given next-period consumption function c_next (n_a × n_y matrix, on asset grid),
return current-period consumption c_cur and saving policy a_cur via EGM.

Returns (c_cur, a_pol) both of size (n_a × n_y), interpolated onto p.ag.a.
"""
function egm_step(
    c_next::Matrix{Float64},
    p::Params,
    ag::AssetGrid,
    ip::IncomeProcess,
    r::Float64,
    w::Float64
)
    n_a, n_y = ag.n, ip.n
    β, γ = p.β, p.γ

    # --- 1. Compute E[u'(c_next(a', y')) | y] for each (a', y) ---
    # Shape: (n_a × n_y): expectation over y' conditional on y
    Eu′ = zeros(n_a, n_y)
    for j in 1:n_y
        for jj in 1:n_y
            Eu′[:, j] .+= ip.Π[j, jj] .* u′.(c_next[:, jj], γ)
        end
    end

    # --- 2. Euler equation: u'(c) = β(1+r) E[u'(c_next)] ---
    rhs = β * (1 + r) .* Eu′          # (n_a × n_y)
    c_egm = u′_inv.(rhs, γ)            # (n_a × n_y): c on the EGM a-grid

    # --- 3. Back out the endogenous grid of current wealth ---
    # a' is the exogenous grid (ag.a); current wealth: m = c + a' = (1+r)a + wy
    # So: m_egm(j, y) = c_egm(j, y) + ag.a[j]
    m_egm = c_egm .+ ag.a'  # broadcast: (n_a × n_y)
    # m_egm[:, j] is the endogenous wealth grid for income state j

    # --- 4. Interpolate onto the exogenous asset grid ---
    c_pol  = zeros(n_a, n_y)
    a_pol  = zeros(n_a, n_y)

    for j in 1:n_y
        wy = w * ip.y[j]
        m_grid_j = m_egm[:, j]   # endogenous wealth grid, income j

        # Interpolation object (extrapolate flat below lower bound)
        itp = linear_interpolation(m_grid_j, c_egm[:, j],
                                   extrapolation_bc=Flat())

        for i in 1:n_a
            # Current wealth on the exogenous grid
            m_i = (1 + r) * ag.a[i] + wy
            # Consumption (constrained to be ≥ 0, assets ≥ ā)
            c_pol[i, j] = max(itp(m_i), 1e-10)
            # Savings: enforce borrowing constraint
            a_pol[i, j] = max(m_i - c_pol[i, j], ag.a[1])
        end
    end

    return c_pol, a_pol
end

# ---------------------------------------------------------------------------
# Iterate to convergence
# ---------------------------------------------------------------------------

"""
    solve_household(p, ag, ip; r, w, tol, max_iter)

Iterate EGM until consumption policy converges.
Returns (c_pol, a_pol) — both (n_a × n_y).
"""
function solve_household(
    p::Params,
    ag::AssetGrid,
    ip::IncomeProcess;
    r::Float64 = p.r_ss,
    w::Float64 = p.w_ss,
    tol::Float64 = 1e-10,
    max_iter::Int = 5_000
)
    n_a, n_y = ag.n, ip.n

    # Initialise with hand-to-mouth guess: consume all income + interest
    c = zeros(n_a, n_y)
    for j in 1:n_y
        wy = w * ip.y[j]
        for i in 1:n_a
            c[i, j] = max((1 + r) * ag.a[i] + wy - ag.a[1], 1e-10)
        end
    end

    for iter in 1:max_iter
        c_new, a_new = egm_step(c, p, ag, ip, r, w)
        err = maximum(abs.(c_new .- c))
        c = c_new
        if err < tol
            return c, a_new
        end
    end
    @warn "EGM did not converge within $max_iter iterations"
    _, a_pol = egm_step(c, p, ag, ip, r, w)
    return c, a_pol
end
