"""
    household_block.jl

Solves the household consumption-savings problem via the
Endogenous Grid Method (EGM, Carroll 2006).
"""

using LinearAlgebra
using Interpolations

@inline u(c::Float64, γ::Float64) = γ == 1.0 ? log(c) : c^(1-γ) / (1-γ)
@inline u′(c::Float64, γ::Float64) = c^(-γ)
@inline u′_inv(x::Float64, γ::Float64) = x^(-1/γ)

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

    # 1. Expected marginal utility: Eu'[a', y] = Σ_{y'} Π[y,y'] u'(c_next[a', y'])
    Eu′_mat = zeros(n_a, n_y)
    for j in 1:n_y
        for jj in 1:n_y
            if ip.Π[j, jj] > 1e-15
                Eu′_mat[:, j] .+= ip.Π[j, jj] .* u′.(c_next[:, jj], γ)
            end
        end
    end

    # 2. Euler equation inversion: c_egm on the exogenous a' grid
    rhs   = β * (1 + r) .* Eu′_mat       # (n_a × n_y)
    c_egm = u′_inv.(rhs, γ)              # (n_a × n_y)

    # 3. Back out endogenous cash-on-hand grid: m = c + a'
    #    ag.a is the a' (next-period) grid; m_egm[:, j] = c_egm[:, j] + ag.a
    m_egm = c_egm .+ ag.a               # (n_a × n_y), ag.a broadcasts over columns

    # 4. Interpolate onto exogenous current-wealth grid (1+r)*a + w*y
    c_pol = zeros(n_a, n_y)
    a_pol = zeros(n_a, n_y)

    for j in 1:n_y
        wy = w * ip.y[j]

        # Endogenous m-grid for this income state (must be sorted for interp)
        m_grid_j = m_egm[:, j]

        # Ensure strict monotonicity — if EGM grid is non-monotone near
        # borrowing constraint, sort and deduplicate
        if !issorted(m_grid_j)
            perm = sortperm(m_grid_j)
            m_grid_j = m_grid_j[perm]
            c_j_sorted = c_egm[perm, j]
        else
            c_j_sorted = c_egm[:, j]
        end

        # Remove duplicate m values (can occur at kink near constraint)
        unique_mask = [true; diff(m_grid_j) .> 1e-14]
        m_grid_j    = m_grid_j[unique_mask]
        c_j_sorted  = c_j_sorted[unique_mask]

        # Linear interpolation with flat extrapolation outside range
        itp = linear_interpolation(m_grid_j, c_j_sorted, extrapolation_bc=Flat())

        for i in 1:n_a
            m_i = (1 + r) * ag.a[i] + wy          # current cash-on-hand
            c_i = Float64(itp(m_i))
            # Enforce: c ≥ ε, a' = m - c ≥ a_min
            c_i       = max(c_i, 1e-10)
            a_i       = m_i - c_i
            if a_i < ag.a[1]
                a_i = ag.a[1]
                c_i = max(m_i - ag.a[1], 1e-10)
            end
            c_pol[i, j] = c_i
            a_pol[i, j] = a_i
        end
    end

    return c_pol, a_pol
end

function solve_household(
    p::Params,
    ag::AssetGrid,
    ip::IncomeProcess;
    r::Float64   = p.r_ss,
    w::Float64   = p.w_ss,
    tol::Float64 = 1e-10,
    max_iter::Int = 5_000
)
    n_a, n_y = ag.n, ip.n

    # Initial guess: hand-to-mouth (consume all cash-on-hand above constraint)
    c = zeros(n_a, n_y)
    for j in 1:n_y
        wy = w * ip.y[j]
        for i in 1:n_a
            c[i, j] = max((1 + r) * ag.a[i] + wy - ag.a[1], 1e-10)
        end
    end

    local a_pol
    for iter in 1:max_iter
        c_new, a_new = egm_step(c, p, ag, ip, r, w)
        err = maximum(abs.(c_new .- c))
        c    = c_new
        a_pol = a_new
        if err < tol
            return c, a_pol
        end
    end
    @warn "EGM did not converge within $max_iter iterations"
    return c, a_pol
end
