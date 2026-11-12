"""
    SOE_HANK_DCP.jl

Main module: Small Open Economy HANK with Dominant Currency Pricing.

Architecture:
  - Households: HANK with idiosyncratic income risk, solved via zero-allocation FastEGM.jl
  - Steady-state distribution: Young (2010) non-stochastic lottery method (FastEGM.jl)
  - Aggregate dynamics: Full non-Toeplitz Sequence-Space Jacobians (Auclert et al. 2021)
  - Open-economy block: DCP import pricing, UIP exchange rate, NKPC
  - Monetary shock: 100bp Taylor-rule-based interest rate hike

Author: Anuja A. Gokhale
"""

module SOE_HANK_DCP

using LinearAlgebra
using SparseArrays
using Parameters
using FastEGM

export Params, AssetGrid, IncomeProcess
export solve_household
export stationary_distribution, aggregates, check_market_clearing
export SteadyState, jacobian_C_r, jacobian_C_Y
export GESystem, solve_ge
export compute_fast_ssj_jacobians, run_model

include("parameters.jl")
include("household_block.jl")
include("steady_state.jl")
include("fake_news.jl")
include("SSJEngine.jl")
include("open_economy_dag.jl")

using .SSJEngine: compute_household_jacobians

"""
    compute_fast_ssj_jacobians(p, ag, ip)

Solves the steady-state household problem and ergodic distribution via FastEGM.jl,
then computes the full T x T Sequence-Space Jacobians (J_Cr, J_CY) via the
Auclert et al. (2021) Fake-News algorithm.
"""
function compute_fast_ssj_jacobians(p::Params, ag::AssetGrid, ip::IncomeProcess)
    fp = ModelParams(
        p.β, p.γ, 1.0 + p.r_ss, p.w_ss, p.a_min,
        ag.n, ip.n, copy(ag.a), copy(ip.y), copy(ip.Π)
    )

    ws = EGMWorkspace(fp.Na, fp.Ny)
    c_ss = [max(fp.R * fp.a_grid[ia] + fp.w * fp.y_grid[iy] - fp.a_min, 1e-8)
            for ia in 1:fp.Na, iy in 1:fp.Ny]
    solve_steady_state!(c_ss, ws, fp)
    a_ss = copy(ws.a_new)

    D_ss = zeros(fp.Na, fp.Ny)
    dws_ss = DistWorkspace(fp.Na, fp.Ny)
    solve_stationary_dist!(D_ss, dws_ss, a_ss, fp)

    J = compute_household_jacobians(c_ss, a_ss, D_ss, dws_ss, fp; T=p.T)
    # In baseline SOE production Y = w * N with N = 1 (unit-elastic labor supply), dY = dw
    return (; c_ss, a_ss, D_ss, J_Cr = J.J_CR, J_CY = J.J_Cw)
end

"""
    run_model(; p, verbose)

Full model pipeline using FastEGM.jl + exact Sequence-Space GE inversion.
"""
function run_model(; p::Params = Params(), verbose::Bool = true)
    verbose && println("=== SOE-HANK-DCP Model (FastEGM + Full SSJ) ===")

    ag = AssetGrid(p.a_min, p.a_max, p.n_a)
    ip = IncomeProcess(p.n_y, p.ρ_y, p.σ_y)
    verbose && println("✓ Grids: a ∈ [$(round(ag.a[1],digits=2)), $(round(ag.a[end],digits=2))], n_a=$(p.n_a), n_y=$(p.n_y)")

    verbose && print("  Solving FastEGM steady state & Fake-News Jacobians (T=$(p.T))... ")
    hh = compute_fast_ssj_jacobians(p, ag, ip)
    verbose && println("✓  C_ss=$(round(sum(hh.D_ss .* hh.c_ss), digits=4))  A_ss=$(round(sum(hh.D_ss .* hh.a_ss), digits=4))")

    ge = GESystem(hh.J_Cr, hh.J_CY, p, p.T)

    verbose && println("  Running 100bp MP shock (σ_MP=$(p.σ_MP))...")
    dε_MP = zeros(p.T)
    dε_MP[1] = p.σ_MP

    irfs = solve_ge(ge, dε_MP)
    verbose && println("✓ Done.\n")

    return (; ag, ip, p, hh, ge, irfs)
end

end # module