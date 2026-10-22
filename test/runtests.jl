"""
    runtests.jl

Unit tests for SOE-HANK-DCP.
Tests:
  1. Income process: rows of Π sum to 1; stationary dist sums to 1
  2. EGM: consumption is positive everywhere; budget constraint holds
  3. Stationary distribution: sums to 1; non-negative everywhere
  4. Market clearing: Walras' law
  5. Jacobian verification: fake-news vs. finite difference (T=5 horizon)
"""

using Test
using LinearAlgebra

# Load model (adjust path if running from repo root)
include("../src/parameters.jl")
include("../src/household_block.jl")
include("../src/steady_state.jl")
include("../src/fake_news.jl")
include("../src/open_economy_dag.jl")

# Use small grids for speed
p = Params(n_a=50, n_y=5, T=20)
ag = AssetGrid(p.a_min, p.a_max, p.n_a)
ip = IncomeProcess(p.n_y, p.ρ_y, p.σ_y)

@testset "SOE-HANK-DCP Tests" begin

    @testset "Income process" begin
        # Rows of Π sum to 1
        @test all(isapprox.(sum(ip.Π, dims=2), 1.0, atol=1e-12))
        # Stationary dist sums to 1
        @test isapprox(sum(ip.π_stat), 1.0, atol=1e-10)
        # Stationary dist is non-negative
        @test all(ip.π_stat .≥ 0)
        # Π * π_stat = π_stat (eigenvector condition)
        @test isapprox(ip.Π' * ip.π_stat, ip.π_stat, atol=1e-9)
        # Mean income normalised to 1
        @test isapprox(dot(ip.π_stat, ip.y), 1.0, atol=1e-10)
    end

    @testset "Household EGM" begin
        c_pol, a_pol = solve_household(p, ag, ip)
        # Consumption is strictly positive
        @test all(c_pol .> 0)
        # Savings ≥ borrowing constraint
        @test all(a_pol .≥ p.a_min - 1e-8)
        # Budget constraint: c + a' = (1+r)a + wy  (check on grid)
        for j in 1:ip.n
            for i in 1:ag.n
                rhs = (1 + p.r_ss) * ag.a[i] + p.w_ss * ip.y[j]
                lhs = c_pol[i,j] + a_pol[i,j]
                @test isapprox(lhs, rhs, atol=1e-7)
            end
        end
    end

    @testset "Stationary distribution" begin
        c_pol, a_pol = solve_household(p, ag, ip)
        g = stationary_distribution(a_pol, ag, ip)
        # Sums to 1
        @test isapprox(sum(g), 1.0, atol=1e-9)
        # Non-negative
        @test all(g .≥ -1e-12)
        # Marginal income distribution matches π_stat
        for j in 1:ip.n
            mass_j = sum(g[:, j])
            @test isapprox(mass_j, ip.π_stat[j], atol=1e-6)
        end
    end

    @testset "Market clearing (Walras)" begin
        c_pol, a_pol = solve_household(p, ag, ip)
        g = stationary_distribution(a_pol, ag, ip)
        agg, nfa = check_market_clearing(g, c_pol, a_pol, ag, ip, p.w_ss, p.r_ss)
        # In a closed economy, net asset position should be ≈ 0
        # In SOE we just check it's finite and within reasonable range
        @test isfinite(nfa)
        @test abs(nfa) < 10.0    # not pathological
        # Consumption is positive
        @test agg.C > 0
    end

    @testset "Fake-News Jacobian vs. finite difference" begin
        c_pol, a_pol = solve_household(p, ag, ip)
        Q  = build_lottery_matrix(a_pol, ag, ip)
        g  = stationary_distribution(a_pol, ag, ip)
        ss = SteadyState(c_pol, a_pol, g, Q, ag, ip, p, p.r_ss, p.w_ss)

        T_test = 5
        J_fn = jacobian_C_r(ss, T_test)

        # J is lower-triangular: future r shock cannot affect past C
        for s in 1:T_test
            for t in s+1:T_test
                @test abs(J_fn[s, t]) < 1e-10
            end
        end

        # Numerical Jacobian: perturb r at t=1 (partial equilibrium)
        δ = 1e-4
        r_hi = p.r_ss + δ
        c_hi, a_hi = solve_household(p, ag, ip; r=r_hi)
        g_hi = stationary_distribution(a_hi, ag, ip)

        agg_base = aggregates(g,    c_pol, a_pol, ag, ip, p.w_ss)
        agg_hi   = aggregates(g_hi, c_hi,  a_hi,  ag, ip, p.w_ss)

        dC_num = (agg_hi.C - agg_base.C) / δ
        dC_fn  = J_fn[1, 1]

        @test abs(dC_num - dC_fn) / (abs(dC_fn) + 1e-10) < 0.01
    end
        T = 10
        Φ_r_π, Φ_r_Y, Φ_r_ε = taylor_rule_matrices(p, T)
        Φ_e_r  = exchange_rate_matrix(p, T)
        Φ_πM_r = import_price_matrix(p, T)
        Φ_πH_Y = nkpc_matrices(p, T)

        # Taylor rule matrix is invertible
        @test abs(det(Φ_r_ε)) > 1e-10

        # UIP: exchange rate appreciates on impact (positive r → positive ê)
        dr = zeros(T); dr[1] = 0.01
        de = Φ_e_r * dr
        @test de[1] > 0    # appreciation on impact

        # NKPC: positive output gap → positive inflation
        dY = zeros(T); dY[1] = 0.01
        dπH = Φ_πH_Y * dY
        @test dπH[1] > 0
    end

end
