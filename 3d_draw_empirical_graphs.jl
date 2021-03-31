# ======================================================================================================================
#                           Replicate Barberis, Jin, and Wang (2021)
#                              Part 3d: computing expected returns
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
theme(:bright)

#%% Import parameters generated from python part 3a
momr_avg_theta_all = DataFrame(CSV.File(Project_folder * "/data/momr_avg_theta_all.csv"))
momr_beta = DataFrame(CSV.File(Project_folder * "/data/momr_avg_beta_all.csv"))
momr_gi = DataFrame(CSV.File(Project_folder * "/data/momr_avg_g_i_all.csv"))
momr_std_skew = DataFrame(CSV.File(Project_folder * "/data/momr_avg_std_skew_Si_xi_all.csv"))

#%% Merge tables
momr_std_skew = select!(momr_std_skew, Not(:Column1))
momr_gi = select!(momr_gi, Not(:Column1))

momr_param_all = leftjoin(momr_std_skew,momr_gi, on=["momr"])


#%% Draw figure 2
# pyplot()
# Plots.PyPlotBackend()
# l = @layout [a  b; c]


# p2 = plot!(momr_param_all.avg_std, momr_param_all.avg_gi, linetype=:scatter ,markershape=:star5, markersize=10,leg = false, dpi=300)
# xlabel!("standard deviation", xguidefontsize=10)
# ylabel!("gain overhang", yguidefontsize=10)
# p3 = plot(momr_param_all.avg_skew, momr_param_all.avg_gi, linetype=:scatter ,markershape=:star5, markersize=10,leg = false, dpi=300)
# xlabel!("skewness", xguidefontsize=10)
# ylabel!("gain overhang", yguidefontsize=10)
# plot(p1, p2, p3, layout = l)

# title!("Objective function of Equation 20", titlefontsize=10)
# gr()
# Plots.GRBackend()
pyplot()
Plots.PyPlotBackend()
plot(momr_param_all.avg_std, momr_param_all.avg_skew, markersize=20, xlims=(0.2,1.2), ylims=(1.8,4.0), framestyle=:box, linetype=:scatter ,markershape=:star5, markersize=10,leg = false, dpi=300)
xlabel!("standard deviation", xguidefontsize=18)
ylabel!("skewness", yguidefontsize=18)
savefig("Figure2a.png")


pyplot()
Plots.PyPlotBackend()
plot(momr_param_all.avg_std, momr_param_all.avg_gi, xlims=(0.2,1.2), ylims=(-0.6,0.2),framestyle=:box,linetype=:scatter ,markershape=:star5, markersize=10,leg = false, dpi=300)
xlabel!("standard deviation", xguidefontsize=18)
ylabel!("gain overhang", yguidefontsize=18)
# title!("Objective function of Equation 20", titlefontsize=10)
savefig("Figure2b.png")


pyplot()
Plots.PyPlotBackend()
plot(momr_param_all.avg_skew, momr_param_all.avg_gi, xlims=(1.8,4.0), ylims=(-0.6,0.2),linetype=:scatter ,framestyle=:box,markershape=:star5, markersize=10,leg = false, dpi=300)
xlabel!("skewness", xguidefontsize=18)
ylabel!("gain overhang", yguidefontsize=18)
# title!("Objective function of Equation 20", titlefontsize=10)
savefig("Figure2c.png")
