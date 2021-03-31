# ======================================================================================================================
#                           Replicate Barberis, Jin, and Wang (2021)
#                              Part 3a: computing expected returns
#
#                                       Author: Gen Li
#                                         03/14/2021
#
#   Note: I run Part 1 on Google Datalab from Google Cloud Platform (GCP). It takes around 1.5h to finish part 1 with
#         32-core CPU and parallel computation.
#
#
# ======================================================================================================================
# Change to your project directory
Project_folder = "/Users/genli/Dropbox/UBC/Course/2020 Term2/COMM 673/COMM673_paper_replica"
# result_folder = Project_folder * "/result"

cd(Project_folder * "/_temp")

using Pkg
# Pkg.activate(joinpath(pwd(),".."))
Pkg.activate(joinpath(pwd(),"code"))


# Pkg.add("Statistics")
# Pkg.add("Distributions")
# Pkg.add("LinearAlgebra")
# Pkg.add("Plots")
# Pkg.add("Parameters")
# Pkg.add("PrettyTables")
# Pkg.add("StatsPlots")
# Pkg.add("SpecialFunctions")
# Pkg.add("Optim")
# Pkg.add("QuadGK")
# Pkg.add("NLsolve")
# Pkg.add("ForwardDiff")
# Pkg.add("CSV")
# Pkg.add("DataFrames")
# Pkg.add("BlackBoxOptim")
# Pkg.add("JuMP")
# Pkg.add("Ipopt")
# Pkg.add("GLPK")
# Pkg.add(url="https://github.com/JuliaMPC/NLOptControl.jl")
# Pkg.add("GR")
# Pkg.add("PGFPlotsX")
# Pkg.add("PlotlyJS")
# Pkg.add("ORCA")
# Pkg.add("PyPlot")
# Pkg.add("PlotThemes")

using LinearAlgebra, Random, Distributions, Plots, Parameters, PrettyTables, Printf
using Optim
using DocStringExtensions
using Plots, StatsPlots
using SpecialFunctions
using QuadGK
using NLsolve
using NLsolve
using ForwardDiff
using Optim: converged, maximum, maximizer, minimizer, iterations #some extra functions
using CSV
using DataFrames
# using BlackBoxOptim
using JuMP, Ipopt
using GLPK
Plots.showtheme(:vibrant)
theme(:vibrant)

#%% Import parameters generated from python part 3a
momr_avg_theta_all = DataFrame(CSV.File(Project_folder * "/data/momr_avg_theta_all.csv"))
momr_beta = DataFrame(CSV.File(Project_folder * "/data/momr_avg_beta_all.csv"))
momr_gi = DataFrame(CSV.File(Project_folder * "/data/momr_avg_g_i_all.csv"))
momr_std_skew = DataFrame(CSV.File(Project_folder * "/data/momr_avg_std_skew_Si_xi_all.csv"))



#%% Set parameters
nu = 7.5
σm = 0.25
Rf = 1

γ̂, b0 = (0.6, 0.6)
α, δ, lamb = (0.7, 0.65, 1.5)

σᵢ_all = momr_std_skew.avg_std
βᵢ_all = momr_beta.avg_beta
g_i_all = momr_gi.avg_gi
Si_all = momr_std_skew.Si
xi_all = momr_std_skew.xi
theta_mi_all = momr_avg_theta_all.avg_theta_mi ./100
theta_i_minus1_all = momr_avg_theta_all.avg_theta_mi ./100


Ri = 0.01
mu = 0.005

#%% Calculate μ̂ and θ̂ᵢ
μ̂ = zeros(10,1)
θ̂ᵢ = zeros(10,1)

for j = 1:10
    println("I am calculating μ̂ and θ̂ᵢ for momentum decile ",j)

    σᵢ = σᵢ_all[j]
    βᵢ = βᵢ_all[j]
    g_i = g_i_all[j]
    Si = Si_all[j]
    xi = xi_all[j]
    theta_mi = theta_mi_all[j]
    theta_i_minus1 = theta_i_minus1_all[j]

    # Define function p_Ri
    function p_Ri(Ri, mu, Si, xi)
        N = 1
        Kl = besselk((nu + N) / 2, sqrt((nu + ((Ri - mu) ^ 2)/Si) * (xi^2) /Si))

        result = (2^(1-(nu+N)/2)) / ( gamma(nu/2) * ((pi * nu)^(N/2)) * (abs(Si)^(1/2))) * (Kl * exp( (Ri - mu) / Si * xi )) / ( (sqrt((nu+((Ri - mu)^2) /Si) * (xi^2) /Si) )^(-(nu+N)/2) * (1+(Ri - mu)^2 / (Si * nu)) ^((nu+N)/2) )

        return result
    end

    # Define P_Ri
    function P_Ri(x, mu, Si, xi)
        integral, err = quadgk(Ri -> p_Ri(Ri, mu, Si, xi), -Inf, x, rtol=1e-8)
        return integral
    end


    # Define dwP_Ri
    function dwP_Ri(x, mu, Si, xi)
        P = P_Ri(x, mu, Si, xi)
        # dwP_Ri = ((δ * P**(δ-1) * (P**δ + (1-P)**δ))
        #           - P**δ * (P**(δ-1) - (1-P)**(δ-1))) / \
        #          ((P**δ + (1-P)**δ)**(1+1/δ)) * p_Ri(Ri, mu, Si, xi)

        return ((δ * P^(δ-1) * (P^δ + (1-P)^δ)) - P^δ * (P^(δ-1) - (1-P)^(δ-1))) /((P^δ + (1-P)^δ)^(1+1/δ)) * p_Ri(x, mu, Si, xi)
    end

    # Define dwP_1_Ri
    function dwP_1_Ri(Ri, mu, Si, xi)
        P = P_Ri(Ri, mu, Si, xi)
        result = -((δ * (1-P)^(δ-1) * (P^δ + (1-P)^δ)) - (1-P)^δ * ((1-P)^(δ-1) - P^(δ-1))) / ((P^δ + (1-P)^δ)^(1+1/δ)) * p_Ri(Ri, mu, Si, xi)

        return result
    end


    # Define neg_integral
    function neg_integral(mu, Si, xi, g_i, theta_mi,theta_i_minus1)
        integral, err = quadgk(x -> ((theta_mi * (Rf-x) - theta_i_minus1 * g_i) ^(α-1))* (Rf-x) * dwP_Ri(x, mu, Si, xi), -100, Rf-theta_i_minus1*g_i/theta_mi, rtol=1e-8)

        return integral
    end

    # Define pos_integral
    function pos_integral(mu, Si, xi, g_i, theta_mi,theta_i_minus1)
        integral, err = quadgk(x -> ((theta_mi * (x-Rf) + theta_i_minus1 * g_i) ^(α-1)) * (x-Rf) * dwP_1_Ri(x, mu, Si, xi), Rf-theta_i_minus1*g_i/theta_mi, 100, rtol=1e-8)

        return integral
    end


    # Define neg_integral in Equation 20
    function neg_integral20(θᵢ, mu, Si, xi, g_i,theta_i_minus1,lamb, b0)
        if θᵢ >= 0
            integral, err = quadgk(x -> (-lamb * b0 *(θᵢ * (Rf-x) - theta_i_minus1 * g_i ) ^(α)) * dwP_Ri(x, mu, Si, xi), -100, Rf-theta_i_minus1*g_i/θᵢ, rtol=1e-8)
        elseif θᵢ < 0
            integral, err = quadgk(x -> (b0 *(θᵢ * (x-Rf) + theta_i_minus1 * g_i) ^(α)) * dwP_Ri(x, mu, Si, xi), -100, Rf-theta_i_minus1*g_i/θᵢ, rtol=1e-8)
        end

        return integral
    end

    # Define pos_integral in Equation 20
    function pos_integral20(θᵢ, mu, Si, xi, g_i,theta_i_minus1,lamb, b0)
        if θᵢ >= 0
            integral, err = quadgk(x -> (-b0 * (θᵢ * (x-Rf) + theta_i_minus1 * g_i) ^(α)) * dwP_1_Ri(x, mu, Si, xi), Rf-theta_i_minus1*g_i/θᵢ, 100, rtol=1e-8)
        elseif θᵢ < 0
            integral, err = quadgk(x -> (lamb * b0 * (θᵢ * (Rf-x) - theta_i_minus1 * g_i ) ^(α)) * dwP_1_Ri(x, mu, Si, xi), Rf-theta_i_minus1*g_i/θᵢ, 100, rtol=1e-8)
        end

        return integral
    end


    # Solve Equation 35 and get μ̂
    function Equation35(mu)
        term1 = (mu[1] + (nu * xi / (nu-2) - Rf)) - γ̂ * βᵢ * σm ^ 2
        term2 = -α * lamb * b0 * neg_integral(mu[1], Si, xi, g_i,theta_mi,theta_i_minus1)
        term3 = - α * b0 * pos_integral(mu[1], Si, xi, g_i,theta_mi,theta_i_minus1)

        return term1 + term2 + term3
    end


    # Equation 20
    function Equation20(θᵢ,μ̂)

        term1 = θᵢ[1] * (μ̂ + (nu * xi)/(nu-2) - Rf) - γ̂ / 2 *(θᵢ[1]^2 * σᵢ^2 + 2*θᵢ[1]*(βᵢ*σm^2 - theta_mi * σᵢ^2))
        term2 =  neg_integral20(θᵢ[1], μ̂, Si, xi, g_i,theta_i_minus1,lamb, b0)
        term3 =  pos_integral20(θᵢ[1], μ̂, Si, xi, g_i,theta_i_minus1,lamb, b0)

        return -(term1 + term2 + term3)
    end

    results = nlsolve(Equation35, [0.5])
    μ̂[j] = results.zero[1]
    # Equation35(μ̂)

    result2 = optimize(θᵢ  -> Equation20(θᵢ,μ̂[j]), -theta_mi, theta_mi*2)
    θ̂ᵢ[j] = Optim.minimizer(result2)[1]


end


#%% Draw Figure 3
function Equation20(θᵢ,μ̂)

    term1 = θᵢ[1] * (μ̂ + (nu * xi)/(nu-2) - Rf) - γ̂ / 2 *(θᵢ[1]^2 * σᵢ^2 + 2*θᵢ[1]*(βᵢ*σm^2 - theta_mi * σᵢ^2))
    term2 =  neg_integral20(θᵢ[1], μ̂, Si, xi, g_i,theta_i_minus1,lamb, b0)
    term3 =  pos_integral20(θᵢ[1], μ̂, Si, xi, g_i,theta_i_minus1,lamb, b0)

    return -(term1 + term2 + term3)
end

θᵢ_rand = LinRange(0.000001,0.002,100)
u_rand = Equation20.(θᵢ_rand,μ̂[j])

θᵢ_rand_neg = LinRange(-0.001,-0.000001,100)
u_rand_neg = Equation20.(θᵢ_rand_neg,μ̂[j])

θᵢ_rand_all = [θᵢ_rand_neg; θᵢ_rand]
u_rand_all = [u_rand_neg; u_rand]


#   Plot graphs
# gr()
# Plots.GRBackend()
pyplot()
Plots.PyPlotBackend()
plot(θᵢ_rand_all, -u_rand_all, w=3, leg = false, color=:blues, dpi=300)
xlabel!("θ₁", xguidefontsize=10)
ylabel!("utility", yguidefontsize=10)
title!("Objective function of Equation 20", titlefontsize=10)
savefig("Figure3.png")


#%% Draw Figure 3 for decile 10
function Equation20(θᵢ,μ̂)

    term1 = θᵢ[1] * (μ̂ + (nu * xi)/(nu-2) - Rf) - γ̂ / 2 *(θᵢ[1]^2 * σᵢ^2 + 2*θᵢ[1]*(βᵢ*σm^2 - theta_mi * σᵢ^2))
    term2 =  neg_integral20(θᵢ[1], μ̂, Si, xi, g_i,theta_i_minus1,lamb, b0)
    term3 =  pos_integral20(θᵢ[1], μ̂, Si, xi, g_i,theta_i_minus1,lamb, b0)

    return -(term1 + term2 + term3)
end

function Equation20_MV(θᵢ,μ̂)

    term1 = θᵢ[1] * (μ̂ + (nu * xi)/(nu-2) - Rf) - γ̂ / 2 *(θᵢ[1]^2 * σᵢ^2 + 2*θᵢ[1]*(βᵢ*σm^2 - theta_mi * σᵢ^2))
    # term2 =  neg_integral20(θᵢ[1], μ̂, Si, xi, g_i,theta_i_minus1,lamb, b0)
    # term3 =  pos_integral20(θᵢ[1], μ̂, Si, xi, g_i,theta_i_minus1,lamb, b0)

    return -(term1)
end

function Equation20_PT(θᵢ,μ̂)

    # term1 = θᵢ[1] * (μ̂ + (nu * xi)/(nu-2) - Rf) - γ̂ / 2 *(θᵢ[1]^2 * σᵢ^2 + 2*θᵢ[1]*(βᵢ*σm^2 - theta_mi * σᵢ^2))
    term2 =  neg_integral20(θᵢ[1], μ̂, Si, xi, g_i,theta_i_minus1,lamb, b0)
    term3 =  pos_integral20(θᵢ[1], μ̂, Si, xi, g_i,theta_i_minus1,lamb, b0)

    return -(term2 + term3)
end

θᵢ_rand = LinRange(0.000001,0.25,100)
u_rand = Equation20.(θᵢ_rand,0.5815)
MV_rand = Equation20_MV.(θᵢ_rand,0.5815)
PT_rand = Equation20_PT.(θᵢ_rand,0.5815)

θᵢ_rand_neg = LinRange(-0.01,-0.00001,100)
u_rand_neg = Equation20.(θᵢ_rand_neg,0.5815)
MV_rand_neg = Equation20_MV.(θᵢ_rand_neg,0.5815)
PT_rand_neg = Equation20_PT.(θᵢ_rand_neg,0.5815)


θᵢ_rand_all = [θᵢ_rand_neg; θᵢ_rand]
u_rand_all = [u_rand_neg; u_rand]
MV_rand_all = [MV_rand_neg; MV_rand]
PT_rand_all = [PT_rand_neg; PT_rand]


#   Plot graphs
# gr()
# Plots.GRBackend()
pyplot()
Plots.PyPlotBackend()
plot(θᵢ_rand_all, -u_rand_all, w=2,xlims=(-0.01,0.25), ylims=(-0.004,0.004) ,color=:red, leg = false, dpi=300)
plot!(θᵢ_rand_all, -MV_rand_all, linestyle=:dash, w=1,xlims=(-0.01,0.25), ylims=(-0.004,0.004) ,leg = false, dpi=300)
plot!(θᵢ_rand_all, -PT_rand_all, linestyle=:dashdot, w=1,xlims=(-0.01,0.25), ylims=(-0.004,0.004) ,leg = false, dpi=300)
xlabel!("θ₁₀", xguidefontsize=10)
ylabel!("utility", yguidefontsize=10)
title!("Objective function for Decile 10", titlefontsize=10)
savefig("Figure4.png")
