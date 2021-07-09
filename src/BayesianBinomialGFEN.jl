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
    ridge::Float64;  # l2 reg
    clamp_value::Float64 = 10.0
)::Float64
    # add a small constant for numerical stability
    n_nbrs =  length(nbr_values)
    # heuristic_scale = clamp(2^(1.0 / a), 1.0, 1e6)
    
    # define target loglikelihood, any target is ok as long as it is concave
    # and it is easy to provide one point with positive and negative slope
    function target_dens(θ)::Tuple{Float64, Float64}
        if θ >= 0.0
            z = exp(-θ)
            ω = z / (1.0 + z)
            logll = - a * log(1.0 + z) - (a - s) * θ
            ∇logll = a * ω - (a - s)
        else
            z = exp(θ)
            ω = z / (1.0 + z)
            logll = s * θ - a * log(1.0 + z)
            ∇logll = s - a * ω
        end
    
        tv1_reg, tv2_reg, ∇tv1_reg, ∇tv2_reg = 0., 0., 0., 0.
        if n_nbrs > 0
            δ_nbrs = θ .- nbr_values
            tv1_reg = dot(abs.(δ_nbrs), tv1)
            tv2_reg = 0.5 * dot(δ_nbrs.^2, tv2)
            ∇tv1_reg = dot(sign.(δ_nbrs), tv1)
            ∇tv2_reg = dot(δ_nbrs, tv2)
        end

        lasso_reg = lasso * abs(θ)
        ∇lasso_reg = lasso * sign(θ)
        ridge_reg = 0.5ridge * θ^2
        ∇ridge_reg = ridge * θ
        f = (logll - tv1_reg - tv2_reg - lasso_reg - ridge_reg) # / heuristic_scale
        g = (∇logll - ∇tv1_reg - ∇tv2_reg - ∇lasso_reg - ∇ridge_reg)  #/ heuristic_scale
        f, g
    end
    
    # it's easy to define points with positive and negative slopes
    # based on the minimum and max posible values
    # efficiency of the sample increase with good envelope points
    lower = max(min(logit(s / a),  minimum(nbr_values) - 1e-6), -clamp_value)
    upper = min(max(logit(s / a), maximum(nbr_values) + 1e-6), clamp_value)
    # lower = 0.0
    # upper = 0.0
    # build sampler
    support = (-clamp_value, clamp_value)
    init = (lower, upper)
    sampler = RejectionSampler(
        target_dens, support, init, autograd=false, use_secants=true, apply_log=false
    )
    
    # return one sample
    out = run_sampler!(sampler, 1)[1]
    num_attempts = length(sampler.envelop)
    # println("Num attempts: $num_attempts")
    # error("hi")
    return out
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
    init::Union{Vector{Float64},Nothing} = nothing,  # initial values for chain,
    init_eps::Float64=0.0,
    async::Bool,
    thinning::Int = 1,
    verbose::Bool = false,
    burnin::Int = 0
)
    # store each gibbs sweep in matrix
    T = length(s)    
    θ = zeros(Float64, T, (n - burnin) ÷ thinning)

    num_saved = 0
    if isnothing(init)
        init = logistic.((s .+ init_eps) ./ (a .+ 2init_eps))
    end
    θ_curr = init

    verbose && (pbar = Progress(n))
    for i in 1:n
        θ_curr = binomial_gibbs_sweep(
            θ_curr, s, a, m.nbrs, m.tv1, m.tv2, m.lasso, m.ridge, async=async
        )
        if i > burnin && i % thinning == 0
            θ[:, num_saved + 1] = θ_curr
            num_saved += 1
        end
        verbose && (next!(pbar))
    end

    return θ
end
