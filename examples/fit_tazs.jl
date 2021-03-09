using Pkg; Pkg.activate(".")
using DataFrames
using Printf
using DelimitedFiles
using JSON
using GraphFusedElasticNet
using NPZ

##
splits = readtable("../gfen-reproduce/processed_data/splits_opt_pt.csv")

##
split_to_fit = 3
split = splits[split_to_fit, :]

##
fname = @sprintf("../gfen-reproduce/productivity_splits/%02d.csv", split_to_fit + 1)
splitdata = readdlm(fname, ',')

## find best lambda
fname = @sprintf("../gfen-reproduce/modelfit_metrics/cvloss_%02d.csv", split_to_fit + 1)
best_lams = readtable(fname)
best_row = argmax(best_lams.final_pred)
λsl1 = best_lams.λsl1[best_row]
λsl2 = best_lams.λsl2[best_row]
λtl1 = best_lams.λtl1[best_row]
λtl2 = best_lams.λtl2[best_row]

##
edges_df = readtable("../gfen-reproduce/processed_data/spatiotemporal_graph.csv")

##
num_nodes = size(splitdata, 1)
num_edges = size(edges_df, 1)

## nbr map
edges = [(r.vertex1 + 1, r.vertex2 + 1) for r in eachrow(edges_df)]
tv1 = [(r.temporal == 1) ? λtl1 : λsl1 for r in eachrow(edges_df)]
tv2 = [(r.temporal == 1) ? λtl2 : λsl2 for r in eachrow(edges_df)]

##
mod = BayesianBinomialGFEN(edges, tv1=tv1, tv2=tv2)

## load data and map estimate for fast mixing
fname = @sprintf("../gfen-reproduce/best_betas/betas_%02d.csv", split_to_fit)
init = vec(readdlm(fname, ','))
s = splitdata[:, 1]
a = splitdata[:, 2]

## fit model
n = 1_000
thinning = 5
burnin = 0.5

##
chain = sample_chain(mod, s, a, n, init=init, verbose=true, async=true)

##
nstart = ceil(Int, size(chain, 2) * burnin)
chain = chain[:, nstart:end]

##
writedlm()