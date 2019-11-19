using Distributed

"""
Graph Fused Lasso Filter
"""
mutable struct DensityTree
    beta::Matrix{Float64}
    bins::Vector{Tuple{Int, Int}}
    levels::Int
    splits::Vector{Float64}

    # model specification with tuning parameters
    function DensityTree(splits::Vector{Float64})
        @assert issorted(splits)
        @assert isinteger(log2(length(splits) - 1))
        x = new()
        x.splits = splits
        x.levels = log2(length(splits) - 1)

        # bins
        x.bins = generate_bins2(x.levels)

        # empty models
        x.beta = zeros(0, 0)
        x
    end
end

function predict(
    model::DensityTree,
    evalpts::Vector{Float64};
    asdensity::Bool = true
)
    @assert issorted(evalpts)

    levels, splits = model.levels, model.splits
    n, m = size(model.beta)

    # takes the ending nodes
    nbreaks = length(splits) - 1
    endpoints = level_numnodes(levels + 1)

    # preallocate space for the prediction at each cut
    cutprobs = ones(n, nbreaks)
    if asdensity # compute model instead of mass
        for j in 1:nbreaks
            @views fill!(cutprobs[:, j], 1. / (splits[j + 1] - splits[j]))
        end
    end

    # now descend each leave
    for j in 1:nbreaks
        genealogy = allpred(endpoints[j])
        for l in 1:levels
            parent_node, direction = genealogy[l]
            @simd for i in 1:n
                x = 1. / (1. + exp(-model.beta[i, parent_node]))
                cutprobs[i, j] *= direction ? 1. - x : x
            end
        end
    end

    # find the cutting point corresponding to each evaluation point
    N = length(evalpts)
    evalpts_cut = zeros(Int, N)
    for i in 1:N
        j = 1
        while j <= nbreaks + 1 && evalpts[i] > splits[j]
            j += 1
        end
        evalpts_cut[i] = j
    end

    # output
    [[e == 1 || e == nbreaks + 2 ? 0. : cutprobs[i, e - 1] for e in evalpts_cut] for i in 1:n]
end



# function spacetimedensity_bestlambdas(;
#     y::Vector{Float64},
#     node::Vector{Int},
#     levels::Int,
#     splits::Vector{Float64},
#     spacelambdas::Vector{Float64},
#     timelambdas::Vector{Float64},
#     ptr::Vector{Int},
#     brks::Vector{Int},
#     istemp::Vector{Bool},
#     priorcount::Float64 = 0.,
#     numnodes::Int = maximum(node)
#     opts...
# )
#     levels = Int(log2(length(splits) - 1))
#     countmat = bincount(y, node, splits; priorcount=priorcount, numnodes=numnodes)
#     fn = tempname()
#     writedlm(fn, countmat, ',')
#
#     # load data everywhere
#     @everywhere begin
#         countmat = readdlm($fn, ',')
#     end
#
#     bins = generate_bins(levels)
#
#     bestmodels = pmap(bins) do fn
#         # read data
#
#
#         # read trail
#         print(stdout, "Reading data", fn, "...")
#         edgeinfo = readdlm("/home1/05863/mgarciat/traildata/edgedata.tsv", Int)
#         ptr = readdlm("/home1/05863/mgarciat/traildata/ptr.txt", Int)[:, 1]
#         brks = readdlm("/home1/05863/mgarciat/traildata/brks.txt", Int)[:, 1]
#         istemp = readdlm("/home1/05863/mgarciat/traildata/istemp.txt", Bool)[:, 1]
#
#         # read split data
#         data = readdlm("/home1/05863/mgarciat/datasplits/" * fn, '\t', Int)
#
#         # binomial model data with prior
#         s0, N0 = 0.05, 0.01
#         successes = data[:, 1] .+ s0
#         attempts = data[:, 2] .+ N0
#
#         # lambas to try
#         spacelambdas = [0.001, 0.005, 0.01, 0.05, 0.1,  0.5, 1.0, 5.0]
#         timelambdas = [0.001, 0.005, 0.01, 0.05, 0.1,  0.5, 1.0, 5.0]
#
#         # model options
#         modelopts = Dict(
#             :admm_balance_every => 10,
#             :admm_init_penalty => 5.0,
#             :admm_residual_balancing => true,
#             :admm_adaptive_inflation => true,
#             :plateau_tol =>  0.001,
#             :reltol => 1e-5
#         )
#         autofit = BinomialGFLSpaceTimeAutoFit(
#             spacelambdas, timelambdas, ptr, brks, istemp; modelopts...)
#
#         # fit the model
#         print(stdout, "Training model", fn, "...\n")
#         fit!(autofit, successes, attempts; walltime=500.0)
#
#         # return best beta
#         print(stdout, "Writing best beta and saving model", fn, "...\n")
#         bestmodel = autofit.modellist[autofit.bestindex]
#         bestbeta = bestmodel.beta
#         print(stdout, "Best lambda for split", fn, "is", autofit.alllambdas[autofit.bestindex], "...\n")
#
#         writedlm("/home1/05863/mgarciat/results/beta" * fn[6:end], bestbeta, '\t')
#         # save("/home1/05863/mgarciat/results/model" * fn[6:end-4] * ".jld2", "bestmodel", bestmodel)
#
#         print(stdout, "Finished split\n", fn)
#         bestmodel
#     end
# end




#
# mutable struct GFDensityAutoFit
#     lambda_list::Vector{Float64} # will try the same for every nodes_in_level
#     tree::Vector{GFBinomialOddsAutoFit}
#     levels::Int
#     nbins::Int
#     bins::Vector{NTuple{2, NTuple{2, Int}}}
#     cuts::Vector{Float64}
#     tree_best_index::Vector{Int}
#     trained::Bool
#
#     GFDensityAutoFit(
#             lambda_list::Vector{Float64},
#             levels::Int;
#             model_kwargs...
#             ) = begin
#         !issorted(lambda_list) && warn("lambda_list must be in ascending order; sorting...")
#         x = new()
#         x.lambda_list = lambda_list
#         x.levels = levels
#         x.bins = Vector{NTuple{2, NTuple{2, Int}}}(0)
#         x.nbins = 2^levels - 1
#         x.cuts = Float64[]
#         x.tree_best_index = zeros(Int, x.nbins)
#         x.tree = [GFBinomialOddsAutoFit(lambda_list; model_kwargs...) for _ in 1:x.nbins]
#         x.trained = false
#         x
#     end
# end
# @inline getindex(A::GFDensityAutoFit, i::Int) = (@boundscheck checkbounds(A.tree, i); A.tree[i])
# @inline setindex!(A::GFDensityAutoFit, B::GFBinomialOddsAutoFit, i::Int) = (@boundscheck checkbounds(A.tree, i); A.tree[i] = B)
#
# function fit!(
#         model::Union{GFDensity, GFDensityAutoFit},
#         y::Vector{Vector{Float64}},
#         trails::Vector{Vector{Int}};
#         weights::Vector{Float64} = ones(length(y)),
#         ymin::Float64 = recursive_minimum(y) - 1e-12,
#         ymax::Float64 = recursive_maximum(y) + 1e-12,
#         cuts::Vector{Float64} = (m = 2^model.level; [ymin + (ymax - ymin) * k / m for k in 0:m]),
#         init_state::Union{Void, Vector{Void}, Vector{GFBinomialOdds}} = nothing,
#         verbose::Bool = true)
#     # ymin  = recursive_minimum(y) - 1e-12
#     # ymax  = recursive_maximum(y) + 1e-12
#     # weights = ones(length(y))
#     # cuts = (m = 2^model.levels; [ymin + (ymax - ymin) * k / m for k in 0:m])
#     bins = binary_column_splits(model.levels)
#     @assert init_state == nothing || length(init_state) == length(bins)
#     model.bins = bins
#     model.cuts = cuts
#     counts_per_bin = binary_tree_counts_per_bin(y, model.levels, ymin, ymax, cuts)
#
#     if init_state == nothing
#         init_state = [nothing for i = 1:length(bins)]
#     end
#
#     if nworkers() == 1
#         Threads.@threads for i = 1:length(bins)
#             successes = [sum(x[bins[i][1][1]:bins[i][1][2]]) for x in counts_per_bin]
#             failures = [sum(x[bins[i][2][1]:bins[i][2][2]]) for x in counts_per_bin]
#             attempts = successes + failures
#             fit!(model[i], successes, attempts, trails)
#         end
#     else
#         ddata = map(1:length(bins)) do i
#             bin = bins[i]
#             successes = [sum(x[bin[1][1]:bin[1][2]]) for x in counts_per_bin]
#             attempts = [sum(x[bin[2][1]:bin[2][2]]) for x in counts_per_bin]
#             for t = 1:T
#                 @simd for k = 1:n
#                     attempts[k, t] += successes[k, t]
#                 end
#             end
#             (model[i], successes, attempts, trails, bin)
#         end
#         dres = pmap(ddata) do x
#             fit!(x[1], x[2], x[3], x[4])
#             verbose && println("finished bin $(x[5])")
#             x[1]
#         end
#     end
#
#     if model isa GFDensityAutoFit
#         model.tree_best_index = [x.best_index for x in model.tree]
#     end
#
#     model.trained = true
# end
#
# function predict(
#     model::Union{GFDensity, GFDensityAutoFit},
#     evalpts::Vector{Float64};
#     discrete::Bool = false)
#
#     levels, cuts = model.levels, model.cuts
#     n = length(predict(model[1]))
#
#     # takes the ending nodes
#     nbreaks = length(cuts) - 1
#     endpoints = nodes_in_level(levels + 1)
#
#     # preallocate space for the prediction at each cut
#     cutprobs = ones(n, nbreaks)
#     if !discrete # compute model instead of mass
#         for j in 1:nbreaks
#             cutprobs[:, j] = fill(1. / (cuts[j + 1] - cuts[j]), n, 1)
#         end
#     end
#
#     # now descend each leave
#     for j in 1:nbreaks
#         genealogy = all_parents(endpoints[j])
#         for l in 1:levels
#             parent_node, direction = genealogy[l]
#             probs = predict(model[parent_node])
#             for i in 1:n
#                 cutprobs[i, j] *= direction == 0 ? probs[i] : 1. - probs[i]
#             end
#         end
#     end
#
#     # find the cutting point corresponding to each evaluation point
#     N = length(evalpts)
#     evalpts_cut = zeros(Int, N)
#     for i in 1:N
#         j = 1
#         while j <= nbreaks + 1 && evalpts[i] > cuts[j]
#             j += 1
#         end
#         evalpts_cut[i] = j
#     end
#
#     # output
#     [[e == 1 || e == nbreaks + 2 ? 0. : cutprobs[i, e - 1] for e in evalpts_cut] for i in 1:n]
# end
#
# function best_tree_lambdas(model::GFDensityAutoFit)
#     model.lambda_list[model.tree_best_index]
# end
#
# function best_tree_models(model::GFDensityAutoFit)
#     idx = model.tree_best_index
#     [model[i][idx[i]] for i = 1:length(idx)]
# end
