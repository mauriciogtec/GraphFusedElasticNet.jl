using LinearAlgebra
using ProgressMeter
using StatsFuns


"""
Binomial Likelihood Bayesian Graph Fused Elastic Net Model
"""
mutable struct BayesianBinomialGFEN
    # trail info
    edges::Vector{Tuple{Int, Int}}

    num_nodes::Int
    num_edges::Int
    num_trails::Int

    # model parameters and computed quantities
    nbrs::Vector{Vector{Int}}
    tv1::Vector{Vector{Float64}}
    tv2::Vector{Vector{Float64}}
    ridge::Vector{Float64}
    lasso::Vector{Float64}

    # convergence and steps
    trained::Bool
    steps::Int

    # main constructor
    function BayesianBinomialGFEN(
        edges::Vector{Tuple{Int, Int}};
        tv1::Union{Vector{Float64},Float64} = 0.0,
        tv2::Union{Vector{Float64},Float64} = 0.0,
        lasso::Union{Vector{Float64},Float64} = 0.0,
        ridge::Union{Vector{Float64},Float64} = 0.0
    )
        # new object
        x = new()

        # assign trails
        x.edges = edges
        x.num_edges = length(edges)
        x.num_nodes = maximum(maximum(s) for s in edges)

        # assign penalties
        tv1 = fill_if_scalar(tv1, x.num_edges)
        tv2 = fill_if_scalar(tv2, x.num_edges)
        lasso = fill_if_scalar(lasso, x.num_nodes)
        ridge = fill_if_scalar(ridge, x.num_nodes)

        @assert length(tv1) == x.num_edges
        @assert length(tv2) == x.num_edges
        @assert length(lasso) == x.num_nodes
        @assert length(ridge) == x.num_nodes

        x.lasso = lasso
        x.ridge = ridge

        # make neighbors map
        x.nbrs = [Int[] for _ in 1:x.num_nodes]
        x.tv1 = [Float64[] for _ in 1:x.num_nodes]
        x.tv2 = [Float64[] for _ in 1:x.num_nodes]

        for (i, (s, t)) in enumerate(edges)
            push!(x.nbrs[s], t)
            push!(x.nbrs[t], s)

            push!(x.tv1[s], tv1[i])
            push!(x.tv1[t], tv1[i])

            push!(x.tv2[s], tv2[i])
            push!(x.tv2[t], tv2[i])
        end

        for i in 1:x.num_nodes
            @assert length(x.nbrs[i]) == length(x.tv1[i])
            @assert length(x.nbrs[i]) == length(x.tv2[i])
        end

        # convergence and steps
        x.trained = false
        x.steps = 0

        # end constructor
        return x
    end
end
Base.show(io::IO, x::BayesianBinomialGFEN) = print(io, "BayesianBinomialGFEN")
Base.print(io::IO, x::BayesianBinomialGFEN) = print(io, "BayesianBinomialGFEN")



function binomial_gibbs_step(
    s::Float64,  # successes
    a::Float64,  # attempts
    nbr_values::Vector{Float64},  # values of neighbors
    tv1::Vector{Float64},  # edge penalties l1
    tv2::Vector{Float64},  # edge penalties l2
    lasso::Float64,  # l1 reg
    ridge::Float64,  # l2 reg
)::Float64
    # add a small constant for numerical stability
    ϵ = 1e-8
    a = a + 2ϵ
    s = s + ϵ
    n_nbrs =  length(nbr_values)
    
    # define target loglikelihood, any target is ok as long as it is concave
    # and it is easy to provide one point with positive and negative slope
    target_dens(θ) = begin
        logll = if θ >= 0
        - a * log(1.0 + exp(-θ)) - (a - s) * θ
    else
        s * θ - a * log(1.0 + exp(θ))
    end
    
    if n_nbrs > 0
        # multiply by 0.5 since edges are repeated
        tv1_reg = 0.5 * dot(abs.(θ .- nbr_values), tv1)
        tv2_reg = 0.5 * dot((θ .- nbr_values).^2, tv2)
    else
        tv1_reg = 0.0
        tv2_reg = 0.0
    end
        lasso_reg = lasso * abs(θ)
        ridge_reg = ridge * θ^2
        logll - tv1_reg - tv2_reg - lasso_reg - ridge_reg
    end
    
    # it's easy to define points with positive and negative slopes
    # based on the minimum and max posible values
    # efficiency of the sample increase with good envelope points
    envelope_init = (
        min(logit(s / a),  minimum(nbr_values)) - 1e-6,
        max(logit(s / a), maximum(nbr_values)) + 1e-6
    )
 
    # build sampler
    support = (-Inf, Inf)
    sampler = RejectionSampler(
        target_dens, support, envelope_init, max_segments=10, from_log=true
    )
    
    # return one sample
    run_sampler!(sampler, 1)[1]
end 


function binomial_gibbs_sweep(
    θ::Vector{Float64},  # current
    s::Vector{Float64},  # successes
    a::Vector{Float64},  # attempts
    nbrs::Vector{Vector{Int}}, # neighbors
    tv1::Vector{Vector{Float64}},
    tv2::Vector{Vector{Float64}},
    lasso::Vector{Float64},
    ridge::Vector{Float64};
    async::Bool=true
)
    T = length(θ)
    θnew = zeros(Float64, T)
    
    # async updates using multiple threads
    if async
        @inbounds @threads for t in 1:T
            nbr_values = [θ[i] for i in nbrs[t]]
            θnew[t] = binomial_gibbs_step(
                s[t], a[t], nbr_values, tv1[t], tv2[t], lasso[t], ridge[t]
            )
        end
    else
        @inbounds for t in 1:T
            nbr_values = [θ[i] for i in nbrs[t]]
            θnew[t] = binomial_gibbs_step(
                s[t], a[t], nbr_values, tv1[t], tv2[t], lasso[t], ridge[t]
            )
        end
    end
    return θnew
end


function sample_chain(
    m::BayesianBinomialGFEN,  # model
    s::Vector{Float64},  # successes
    a::Vector{Float64},  # attempts
    n::Int;  # chain iterations (full sweeps),
    init::Union{Vector{Float64},Nothing},  # initial values for chain,
    init_eps::Float64=1e-8,
    async::Bool
)
    # store each gibbs sweep in matrix
    θ = zeros(length(s), n + 1)
    (!@isdefined init) && (init = (s + init_eps) / (a + 2init_eps))
    θ[:, 1] = init
    @showprogress for i in 1:n
        θ[:, i + 1] = binomial_gibbs_sweep(
            θ[:, i], s, a, m.nbrs, m.tv1, m.tv2, m.lasso, m.ridge, async=async
        )
    end
    return θ
end
