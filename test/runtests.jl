using Test
using LinearAlgebra

include("../src/SOE_HANK_DCP.jl")
using .SOE_HANK_DCP: Params, AssetGrid, IncomeProcess, dot, compute_fast_ssj_jacobians, run_model

@testset "SOE-HANK-DCP Full Suite (FastEGM + SSJEngine)" begin
    p = Params(n_a=80, n_y=5, T=30, ϕ_Y=0.125)
    ag = AssetGrid(p.a_min, p.a_max, p.n_a)
    ip = IncomeProcess(p.n_y, p.ρ_y, p.σ_y)

    @testset "Income Process" begin
        @test all(isapprox.(sum(ip.Π, dims=2), 1.0, atol=1e-12))
        @test isapprox(sum(ip.π_stat), 1.0, atol=1e-10)
        @test isapprox(dot(ip.π_stat, ip.y), 1.0, atol=1e-10)
    end

    @testset "FastEGM Steady State & Full SSJ" begin
        hh = compute_fast_ssj_jacobians(p, ag, ip)
        @test isapprox(sum(hh.D_ss), 1.0, atol=1e-10)
        @test all(hh.c_ss .> 0.0)
        @test all(hh.a_ss .>= p.a_min - 1e-10)
        @test 0.0 < hh.J_CY[1, 1] < 1.0
        @test hh.J_Cr[1, 1] < 0.0
        @test abs(hh.J_CY[1, 2]) > 1e-6
    end

    @testset "Exact GE Inversion & DCP vs PCP" begin
        res_dcp = run_model(p = Params(φ=1.0, ϕ_Y=0.125, T=40, n_a=80, n_y=5), verbose=false)
        res_pcp = run_model(p = Params(φ=0.0, ϕ_Y=0.125, T=40, n_a=80, n_y=5), verbose=false)

        χ_e = res_dcp.p.α_M * res_dcp.p.η * (2.0 - res_dcp.p.φ)
        clearing_err = maximum(abs.(res_dcp.irfs.dY .- (res_dcp.irfs.dC .- χ_e .* res_dcp.irfs.de)))
        @test clearing_err < 1e-13

        @test res_dcp.irfs.de[1] > 0.0
        @test res_dcp.irfs.dY[1] < 0.0
        @test res_pcp.irfs.dY[1] < 0.0

        @test maximum(abs.(res_dcp.irfs.dπM)) == 0.0
        @test maximum(abs.(res_pcp.irfs.dπM)) > 0.0
    end
end

@testset "US Fed Spillover Shock (solve_foreign_shock)" begin
    res_dcp = SOE_HANK_DCP.run_model(p = SOE_HANK_DCP.Params(φ=1.0, T=40, n_a=100, n_y=3), verbose=false)
    res_pcp = SOE_HANK_DCP.run_model(p = SOE_HANK_DCP.Params(φ=0.0, T=40, n_a=100, n_y=3), verbose=false)
    dr_star = [0.01 * (0.8^(t-1)) for t in 1:40]

    sp_dcp = SOE_HANK_DCP.solve_foreign_shock(res_dcp.ge, dr_star)
    sp_pcp = SOE_HANK_DCP.solve_foreign_shock(res_pcp.ge, dr_star)

    # Foreign tightening depreciates domestic currency on impact (de < 0)
    @test sp_dcp.de[1] < 0.0
    @test sp_pcp.de[1] < 0.0

    # Under DCP (φ=1), import price pass-through is zero (dπM == 0); under PCP (φ=0), depreciation raises import inflation
    @test maximum(abs.(sp_dcp.dπM)) < 1e-14
    @test sp_pcp.dπM[1] > 0.0
end
