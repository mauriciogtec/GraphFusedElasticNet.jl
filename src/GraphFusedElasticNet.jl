# __precompile__(false)
module GraphFusedElasticNet

include("Utils.jl")
include("L1Solver.jl")
include("L2Solver.jl")
include("BinomialGFEN.jl")
include("GaussianGFEN.jl")
include("DensityTree.jl")
include("GraphTrails.jl")
include("BayesianOpt.jl")
include("ARS.jl")
include("BayesianBinomialGFEN.jl")


export BinomialGFEN, GaussianGFEN
export BayesianBinomialGFEN
export sample_chain
export filter1D!, filter1Dl2!, filter1D, filter1Dl2
export fit!, predict
export Trails
export graph_from_edgelist, find_trails, grid_trails
export GaussianProcessSampler, RandomGaussianProcessSampler
export addobs!, gpsample, gpeval
export eval_logprob, make_tree_from_bfs, uniform_binary_splits, eval_logdens

end
