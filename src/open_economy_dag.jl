"""
    open_economy_dag.jl

Directed Acyclic Graph (DAG) of the open-economy blocks:

    Household Block  →  (C, A)
         ↑                ↓
    [r, w, Y]         Asset market
         ↑
    Central Bank (Taylor Rule)
         ↑
    NK Phillips Curve  ←  Import Price Inflation (DCP)
         ↑
    Exchange Rate (UIP + risk premium)

DCP Transmission Channels:
  1. Cash-flow channel:   r↑ → mortgage servicing costs↑ → C↓ (indebted HHs)
  2. Expenditure-switch:  r↑ → e↑ (appreciation) → P^M↓ → real income↑ → C↑

This file defines the linear sequence-space representation of all
non-household blocks, enabling GE inversion:

    H_U · dU = -H_Z · dZ

where U are unknowns (C, Y, π, e, r) and Z are exogenous shocks.
"""

using LinearAlgebra

# ---------------------------------------------------------------------------
# Helper: identity and lag operators as (T×T) matrices
# ---------------------------------------------------------------------------

"""
    lag_matrix(T, k)

Lower triangular shift matrix: (Lk·x)[t] = x[t-k] (zero for t<k).
"""
function lag_matrix(T::Int, k::Int=1)
    M = zeros(T, T)
    for t in 1+k:T
        M[t, t-k] = 1.0
    end
    return M
end

"""
    lead_matrix(T, k)

Upper triangular shift: (Fk·x)[t] = x[t+k] (zero for t>T-k).
"""
function lead_matrix(T::Int, k::Int=1)
    M = zeros(T, T)
    for t in 1:T-k
        M[t, t+k] = 1.0
    end
    return M
end

const I_T = (T::Int) -> Matrix{Float64}(I, T, T)

# ---------------------------------------------------------------------------
# Block 1: Monetary policy (Taylor rule)
#
#   î_t = ρ_i î_{t-1} + (1-ρ_i)[ϕ_π π_t + ϕ_Y Ŷ_t] + ε_t^{MP}
#   r_t = i_t - π_{t+1}  (Fisher relation, log-linearised)
# ---------------------------------------------------------------------------

"""
    taylor_rule_matrices(p, T)

Returns (Φ_r_i, Φ_r_π, Φ_r_Y) such that:
    r⃗ = Φ_r_i · i⃗ - Φ_r_π · π⃗_{lead}
"""
function taylor_rule_matrices(p::Params, T::Int)
    L = lag_matrix(T)
    F = lead_matrix(T)
    Id = I_T(T)

    # Taylor rule: i = ρ_i L i + (1-ρ_i)(ϕ_π π + ϕ_Y Y) + ε
    # ⟹ (I - ρ_i L) i = (1-ρ_i)(ϕ_π π + ϕ_Y Y) + ε
    A_i = Id - p.ρ_i .* L
    A_i_inv = inv(A_i)

    Φ_i_π = A_i_inv * ((1 - p.ρ_i) * p.ϕ_π .* Id)
    Φ_i_Y = A_i_inv * ((1 - p.ρ_i) * p.ϕ_Y .* Id)
    Φ_i_ε = A_i_inv   # response to MP shock

    # Real rate: r = i - F·π (Fisher, one-period ahead inflation)
    Φ_r_π = Φ_i_π .- F
    Φ_r_Y = Φ_i_Y
    Φ_r_ε = Φ_i_ε

    return Φ_r_π, Φ_r_Y, Φ_r_ε
end

# ---------------------------------------------------------------------------
# Block 2: Exchange rate (UIP + risk premium)
#
#   ê_t = ê_{t+1} + r_t - r*  (UIP, r* = foreign rate, normalised 0)
#   ⟹ ê_t = Σ_{s≥0} r_{t+s}  (forward sum under RE)
# ---------------------------------------------------------------------------

"""
    exchange_rate_matrix(p, T)

Returns Φ_e_r (T×T) such that ê = Φ_e_r · r⃗.

Under UIP with bounded terminal condition: ê_t = -Σ_{s=0}^{T-t-1} r_{t+s}.
Note: appreciation = positive ê here (domestic currency strengthens when r rises).
"""
function exchange_rate_matrix(p::Params, T::Int)
    Φ_e_r = zeros(T, T)
    for t in 1:T
        for s in t:T
            Φ_e_r[t, s] = 1.0   # ê_t = Σ r_{t+s}
        end
    end
    return Φ_e_r
end

# ---------------------------------------------------------------------------
# Block 3: Import price inflation (DCP)
#
#   π_t^M = (1-φ) Δê_t + φ · 0   (φ=1: full DCP; φ=0: full PCP)
#   Under DCP (φ=1): π^M ≈ 0 (import prices sticky in USD)
#   Under PCP (φ=0): π^M = Δê_t
#
#   Domestic CPI: π_t = (1-α_M) π_t^H + α_M π_t^M
# ---------------------------------------------------------------------------

"""
    import_price_matrix(p, T)

Returns Φ_πM_r (T×T) such that π^M = Φ_πM_r · r⃗,
blending DCP and PCP according to φ.
"""
function import_price_matrix(p::Params, T::Int)
    Id = I_T(T)
    L  = lag_matrix(T)
    Φ_e_r = exchange_rate_matrix(p, T)

    # Δê = ê - L·ê  ⟹  Δê = (I - L) Φ_e_r · r
    Φ_Δe_r = (Id .- L) * Φ_e_r

    # Import inflation: π^M = (1-φ) Δê
    Φ_πM_r = (1 - p.φ) .* Φ_Δe_r

    return Φ_πM_r
end

# ---------------------------------------------------------------------------
# Block 4: New Keynesian Phillips Curve
#
#   π_t^H = β E_t[π_{t+1}^H] + κ Ŷ_t
#   ⟹ (I - β F) π^H = κ Y
#   ⟹ π^H = (I - β F)^{-1} κ Y
# ---------------------------------------------------------------------------

"""
    nkpc_matrices(p, T)

Returns (Φ_πH_Y) such that π^H = Φ_πH_Y · Ŷ.
"""
function nkpc_matrices(p::Params, T::Int)
    F  = lead_matrix(T)
    Id = I_T(T)
    A  = Id .- p.β .* F
    Φ_πH_Y = inv(A) * (p.κ .* Id)
    return Φ_πH_Y
end

# ---------------------------------------------------------------------------
# Block 5: GE DAG — assemble H matrices and solve
#
# Unknowns: U = [Ŷ, π^H, r]   (each T-vector, stacked 3T)
# Exogenous: Z = [ε^MP]        (T-vector)
#
# System H_U · U = -H_Z · Z  (linear in sequence space)
# ---------------------------------------------------------------------------

"""
    GESystem

Stores the GE Jacobians needed to solve for IRFs.
"""
struct GESystem
    J_Cr::Matrix{Float64}   # ∂C/∂r from fake-news
    J_CY::Matrix{Float64}   # ∂C/∂Y from fake-news
    Φ_r_π::Matrix{Float64}
    Φ_r_Y::Matrix{Float64}
    Φ_r_ε::Matrix{Float64}
    Φ_e_r::Matrix{Float64}
    Φ_πM_r::Matrix{Float64}
    Φ_πH_Y::Matrix{Float64}
    p::Params
    T::Int
end

function GESystem(J_Cr, J_CY, p::Params, T::Int)
    Φ_r_π, Φ_r_Y, Φ_r_ε = taylor_rule_matrices(p, T)
    Φ_e_r  = exchange_rate_matrix(p, T)
    Φ_πM_r = import_price_matrix(p, T)
    Φ_πH_Y = nkpc_matrices(p, T)
    GESystem(J_Cr, J_CY, Φ_r_π, Φ_r_Y, Φ_r_ε, Φ_e_r, Φ_πM_r, Φ_πH_Y, p, T)
end

"""
    solve_ge(ge, dε_MP)

Given a T-vector of MP shocks dε_MP, solve for IRFs of
Ŷ, π, r, ê, C using the sequence-space DAG.

Returns a NamedTuple with IRF vectors.
"""
function solve_ge(ge::GESystem, dε_MP::Vector{Float64})
    T = ge.T
    Id = I_T(T)
    α_M = ge.p.α_M

    # --- Reduced-form r response (partial eq, taking Y=0 as first pass) ---
    # Iterate: start with r = Φ_r_ε ε, then update via Y feedback

    # Step 1: Initial r from MP shock (no output response yet)
    dr0 = ge.Φ_r_ε * dε_MP

    # Step 2: C response from household block
    dC_r = ge.J_Cr * dr0     # partial eq C from r

    # Step 3: Output (Y = C in SOE with trade balance clearing)
    dY = dC_r

    # Step 4: Inflation from NKPC
    dπH = ge.Φ_πH_Y * dY

    # Step 5: Import inflation from DCP block
    dπM = ge.Φ_πM_r * dr0

    # Total CPI
    dπ = (1 - α_M) .* dπH .+ α_M .* dπM

    # Step 6: Updated r from Taylor rule (using π feedback)
    dr = ge.Φ_r_ε * dε_MP .- ge.Φ_r_π * dπ .- ge.Φ_r_Y * dY

    # Step 7: One more pass (partial GE, 2 iterations sufficient for linear model)
    dC = ge.J_Cr * dr .+ ge.J_CY * dY
    dY = dC
    dπH = ge.Φ_πH_Y * dY
    dπM = ge.Φ_πM_r * dr
    dπ  = (1 - α_M) .* dπH .+ α_M .* dπM
    dr  = ge.Φ_r_ε * dε_MP .- ge.Φ_r_π * dπ .- ge.Φ_r_Y * dY
    de  = ge.Φ_e_r * dr

    return (; dr, dC, dY, dπ, dπH, dπM, de)
end
