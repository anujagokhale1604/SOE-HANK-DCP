using LinearAlgebra

function lag_matrix(T::Int, k::Int=1)
    M = zeros(T, T)
    for t in 1+k:T
        M[t, t-k] = 1.0
    end
    return M
end

function lead_matrix(T::Int, k::Int=1)
    M = zeros(T, T)
    for t in 1:T-k
        M[t, t+k] = 1.0
    end
    return M
end

const I_T = (T::Int) -> Matrix{Float64}(I, T, T)

function taylor_rule_matrices(p::Params, T::Int)
    L = lag_matrix(T)
    F = lead_matrix(T)
    Id = I_T(T)

    A_i = Id - p.ρ_i .* L
    A_i_inv = inv(A_i)

    Φ_i_π = A_i_inv * ((1 - p.ρ_i) * p.ϕ_π .* Id)
    Φ_i_Y = A_i_inv * ((1 - p.ρ_i) * p.ϕ_Y .* Id)
    Φ_i_ε = A_i_inv

    Φ_r_π = Φ_i_π .- F
    Φ_r_Y = Φ_i_Y
    Φ_r_ε = Φ_i_ε

    return Φ_r_π, Φ_r_Y, Φ_r_ε
end

function exchange_rate_matrix(p::Params, T::Int)
    Φ_e_r = zeros(T, T)
    for t in 1:T
        for s in t:T
            Φ_e_r[t, s] = p.ρ_e^(s - t)
        end
    end
    return Φ_e_r
end

function import_price_matrix(p::Params, T::Int)
    Id = I_T(T)
    L  = lag_matrix(T)
    Φ_e_r = exchange_rate_matrix(p, T)
    Φ_Δe_r = (Id .- L) * Φ_e_r
    # Domestic currency appreciation (Δe > 0) lowers domestic-currency import prices under PCP (φ=0)
    Φ_πM_r = -(1.0 - p.φ) .* Φ_Δe_r
    return Φ_πM_r
end

function nkpc_matrices(p::Params, T::Int)
    F  = lead_matrix(T)
    Id = I_T(T)
    A  = Id .- p.β .* F
    Φ_πH_Y = inv(A) * (p.κ .* Id)
    return Φ_πH_Y
end

struct GESystem
    J_Cr::Matrix{Float64}
    J_CY::Matrix{Float64}
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

function solve_ge(ge::GESystem, dε_MP::Vector{Float64})
    T = ge.T
    Id = I_T(T)
    α_M = ge.p.α_M

    # Gopinath et al. (2020) expenditure-switching sensitivity:
    # Under PCP (φ=0), both imports & exports switch (2*α_M*η); under DCP (φ=1), export USD prices are sticky (α_M*η)
    χ_e = α_M * ge.p.η * (2.0 - ge.p.φ)

    M_YY = Id .- ge.J_CY
    M_Yr = -(ge.J_Cr .- χ_e .* ge.Φ_e_r)
    M_rY = ge.Φ_r_Y .+ (1.0 - α_M) .* (ge.Φ_r_π * ge.Φ_πH_Y)
    M_rr = Id .+ α_M .* (ge.Φ_r_π * ge.Φ_πM_r)

    H_U = [M_YY M_Yr;
           M_rY M_rr]
    rhs = [zeros(T);
           ge.Φ_r_ε * dε_MP]

    sol = inv(H_U) * rhs
    dY  = sol[1:T]
    dr  = sol[T+1:2T]

    dC  = ge.J_Cr * dr .+ ge.J_CY * dY
    de  = ge.Φ_e_r * dr
    dπH = ge.Φ_πH_Y * dY
    dπM = ge.Φ_πM_r * dr
    dπ  = (1.0 - α_M) .* dπH .+ α_M .* dπM

    return (; dr, dC, dY, dπ, dπH, dπM, de)
end
