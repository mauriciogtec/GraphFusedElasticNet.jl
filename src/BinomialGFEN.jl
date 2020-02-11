using Base.Threads
using DataStructures
import Base: print, show


"""
Binomial Likelihood Graph Fused Elastic Net Model
"""
mutable struct BinomialGFEN
    # trail info
    ptr::Vector{Int}
    brks::Vector{Int}
    lambdasl1::Vector{Float64}
    lambdasl2::Vector{Float64}

    num_nodes::Int
    num_slacks::Int
    num_trails::Int

    # model parameters and computed quantities
    beta::Vector{Float64}
    slack::Vector{Float64}
    dual::Vector{Float64}

    # convergence and steps
    trained::Bool
    steps::Int
    maxsteps::Int
    converged::Bool
    check_convergence_every::Int
    abstol::Float64
    reltol::Float64
    save_norms::Bool
    save_loss::Bool
    prim_norms::Vector{Float64}
    dual_norms::Vector{Float64}
    loss::Vector{Float64}

    # admm associated parameters
    admm_penalty::Float64
    admm_init_penalty::Float64
    admm_residual_balancing::Bool
    admm_residual_balancing_gap::Float64
    admm_adaptive_inflation::Bool
    admm_balance_every::Int
    admm_fixed_inflation_factor::Float64
    admm_penalty_max_inflation::Float64
    admm_min_penalty::Float64

    # Nesterov acceleration parameters
    accelerate::Bool
    accelerate_min_improvement::Float64
    accelerate_use_restarts::Bool

    # main constructor
    function BinomialGFEN(
        ptr::Vector{Int},
        brks::Vector{Int},
        lambdasl1::Vector{Float64},
        lambdasl2::Vector{Float64};
        maxsteps::Int = Int(1e5),
        check_convergence_every::Int = 1,
        abstol::Float64 = 1e-9,
        reltol::Float64 = 1e-3,
        save_norms::Bool = true,
        save_loss::Bool = true,
        admm_init_penalty::Float64 = 1.0,
        admm_fixed_inflation_factor::Float64 = 2.,
        admm_balance_every::Int = 10,
        admm_residual_balancing_gap::Float64 = 10.,
        admm_penalty_max_inflation::Float64 = 100.,
        admm_residual_balancing::Bool = true,
        admm_adaptive_inflation::Bool = true,
        admm_min_penalty::Float64 = 0.01,
        accelerate::Bool = false,
        accelerate_min_improvement::Float64 = 0.999,
        accelerate_use_restarts::Bool = true
    )
        # Model Validation
        @assert maxsteps > 0
        @assert abstol >= 0.
        @assert reltol > 0.
        @assert admm_init_penalty > 0.
        @assert admm_fixed_inflation_factor > 0.
        @assert admm_balance_every > 0
        @assert admm_residual_balancing_gap > 0.
        @assert check_convergence_every > 0
        @assert admm_penalty_max_inflation > 0.
        @assert admm_min_penalty > 0.

        # new object
        x = new()

        # assign trails
        x.ptr = ptr
        x.brks = brks
        x.lambdasl1 = lambdasl1
        x.lambdasl2 = lambdasl2
        x.num_nodes = maximum(ptr)
        x.num_trails = length(brks) - 1
        x.num_slacks = length(ptr)

        # set values for variables to be trained
        x.beta = zeros(x.num_nodes)
        x.slack = zeros(x.num_slacks)
        x.dual = zeros(x.num_slacks)

        # convergence and steps
        x.trained = false
        x.steps = 0
        x.maxsteps = maxsteps
        x.converged  = false
        x.check_convergence_every = check_convergence_every
        x.abstol = abstol
        x.reltol = reltol
        x.save_norms = save_norms
        x.save_loss = save_loss
        x.prim_norms = Float64[]
        x.dual_norms = Float64[]
        x.loss = Float64[]

        # admm associated parameters
        x.admm_penalty = admm_init_penalty
        x.admm_init_penalty = admm_init_penalty
        x.admm_residual_balancing = admm_residual_balancing
        x.admm_residual_balancing_gap = admm_residual_balancing_gap
        x.admm_adaptive_inflation = admm_adaptive_inflation
        x.admm_balance_every = admm_balance_every
        x.admm_fixed_inflation_factor = admm_fixed_inflation_factor
        x.admm_penalty_max_inflation = admm_penalty_max_inflation
        x.admm_min_penalty = admm_min_penalty

        # Acceleration parameters
        x.accelerate = accelerate
        x.accelerate_min_improvement = accelerate_min_improvement
        x.accelerate_use_restarts = accelerate_use_restarts

        # end constructor
        return x
    end
end
Base.show(io::IO, x::BinomialGFEN) = print(io, "BinomialGFEN")
Base.print(io::IO, x::BinomialGFEN) = print(io, "BinomialGFEN")

function init_vars(
    model::BinomialGFEN, # model
    successes::Vector{Float64},
    attempts::Vector{Float64}
)
    ptr = model.ptr
    brks = model.brks
    α = model.admm_init_penalty
    edgewts = model.lambdasl1 / α
    edgewts2 = model.lambdasl2 / α

    # Initiate ADMM variables
    p = [successes[i] / attempts[i] for i in eachindex(successes)]
    β = [p[i] / (1 - p[i]) for i in eachindex(successes)] # odds
    z = zeros(model.num_slacks) # slacks
    z2 = zeros(model.num_slacks) # slacks
    u = zeros(model.num_slacks) # scaled dual
    u2 = zeros(model.num_slacks) # scaled dual
    Δz = zeros(model.num_slacks) # used for momentum and convergence
    Δu = zeros(model.num_slacks) # used for momentum
    Δz2 = zeros(model.num_slacks) # used for momentum
    Δu2 = zeros(model.num_slacks) # used for momentum

    # trail visits to each node
    num_visits = zeros(Int, model.num_nodes)
    @simd for j in 1:model.num_slacks
        z[j] = β[ptr[j]] # best guess for z
        z2[j] = β[ptr[j]] # best guess for z
        num_visits[ptr[j]] += 1
    end

    εabs_p = sqrt(2.0 * model.num_slacks) * model.abstol # max dual error for the absolute convergence
    εabs_d = sqrt(model.num_nodes) * model.abstol # max prim error for the absolute convergence

    β, z, u, Δz, Δu, z2, u2, Δz2, Δu2, α, edgewts, edgewts2, num_visits, εabs_p, εabs_d
end

# sorted(i::Int, j::Int) = (min(i, j), max(i, j))
# function cnt_reps(ptr::Vector{Int}, brks::Vector{Int})
#     edges = [sorted(ptr[i], ptr[i+1]) for i in 1:length(ptr) if (i + 1) ∉ brks]
#     ctr = counter(edges)
#     edge_cnt = [ctr[sorted(ptr[i], ptr[i+1])] for i in 1:length(ptr) - 1]
#     return edge_cntmodel.
# end

function update_primal!(
    model::BinomialGFEN,
    β::Vector{Float64},
    z::Vector{Float64},
    u::Vector{Float64},
    z2::Vector{Float64},
    u2::Vector{Float64},
    num_visits::Vector{Int},
    successes::Vector{Float64},
    attempts::Vector{Float64},
    α::Float64
)
    ptr, brks = model.ptr, model.brks
    clamp_cnst = 5.0
    
    # admm pseudovalue for beta
    r = zeros(model.num_nodes)
    @simd for j = 1:model.num_slacks # can't do simd here :(
        r[ptr[j]] += z[j] - u[j] + z2[j] - u2[j]
    end

    @simd for i = 1:model.num_nodes
        η = 1. / (1. + exp(-β[i]))
        ω = attempts[i] * η * (1. - η)
        ε = attempts[i] * η - successes[i]
        H = 2.0 * ω  + 2.0 * α * num_visits[i]
        b = (2.0 * (ω * β[i] - ε)  +  α * r[i]) / H
        β[i] = clamp(b, -clamp_cnst, clamp_cnst) # this is very important for stability!
    end
end

@inline function update_slack_and_dual!(
    model::BinomialGFEN,
    z::Vector{Float64},
    u::Vector{Float64},
    Δz::Vector{Float64},
    Δu::Vector{Float64},
    z2::Vector{Float64},
    u2::Vector{Float64},
    Δz2::Vector{Float64},
    Δu2::Vector{Float64},
    β::Vector{Float64},
    edgewts::Vector{Float64},
    edgewts2::Vector{Float64},
    parallel::Bool
)
    ptr = model.ptr
    brks = model.brks

    @simd for j in 1:model.num_slacks
        Δu[j] = -u[j] # we'll add u(k+1) in last loop
        Δz[j] = -z[j] # we'll add z(k+1) in last loop
        Δu2[j] = -u2[j] # we'll add u2(k+1) in last loop
        Δz2[j] = -z2[j] # we'll add z2(k+1) in last loop
        u[j] += β[ptr[j]] # u(k+1) += beta(k+1) - z(k+1), z is subtracted in last loop
        u2[j] += β[ptr[j]] # u2(k+1) += beta(k+1) - z2(k+1), z2 is subtracted in last loop
    end

    zbuff = u # at this point u is u(k) + β(k) because of previous loop
    zbuff2 = u2 # at this point u is u(k) + β(k) because of previous loop
    if parallel
        @threads for t = 1:model.num_trails
            from = brks[t]
            to = brks[t + 1] - 1
            filter1D!(z, zbuff, edgewts, from, to)
            filter1Dl2!(z2, zbuff2, edgewts2, from, to)
        end
    else
        for t = 1:model.num_trails
            from = brks[t]
            to = brks[t + 1] - 1
            filter1D!(z, zbuff, edgewts, from, to)
            filter1Dl2!(z2, zbuff2, edgewts2, from, to)
        end
    end

    @simd for j in 1:model.num_slacks
        u[j] -= z[j] # now u(k+1) += beta(k+1) - z(k+1)
        Δz[j] += z[j] # now Δz(k+1) = z(k+1) - z(k)
        Δu[j] += u[j] # now Δu(k+1) = u(k+1) - u(k)
        u2[j] -= z2[j] # now u2(k+1) += beta(k+1) - z2(k+1)
        Δz2[j] += z2[j] # now Δz2(k+1) = z2(k+1) - z2(k)
        Δu2[j] += u2[j] # now Δu2(k+1) = u2(k+1) - u2(k)
    end
end


@inline function compute_negll(
    β::Vector{Float64},
    successes::Vector{Float64},
    attempts::Vector{Float64}
)
    ll = 0.0
    @simd for i = 1:length(β)
        η = 1. / (1. + exp(-β[i]))
        ll -= successes[i] * log(η + 1e-12) 
        ll -= (attempts[i] - successes[i]) * log(1.0 - η + 1e-12)
    end
    ll
end


@inline function compute_loss(
    model::BinomialGFEN,
    β::Vector{Float64},
    successes::Vector{Float64},
    attempts::Vector{Float64},
)
    ptr = model.ptr
    brks = model.brks
    lambdasl1 = model.lambdasl1
    lambdasl2 = model.lambdasl2

    ll = compute_negll(β, successes, attempts)
    tv = 0.0
    @simd for j in 1:model.num_slacks
        if !(j + 1 in brks)
            δ = β[ptr[j + 1]] - β[ptr[j]]
            tv += lambdasl1[j] * abs(δ) + 0.5 * lambdasl2[j] * δ^2
        end
    end
    return ll + tv
end

@inline function residual_norms!(
    model::BinomialGFEN,
    β::Vector{Float64},
    z::Vector{Float64},
    u::Vector{Float64},
    Δz::Vector{Float64},
    z2::Vector{Float64},
    u2::Vector{Float64},
    Δz2::Vector{Float64},
    α::Float64,
    εabs_p::Float64,
    εabs_d::Float64
)
    ptr = model.ptr

    prim_norm = 0.0
    dual_norm = 0.0
    prim_size = 0.0
    dual_size = 0.0
    @simd for i = 1:model.num_slacks
        prim_norm += (β[ptr[i]] - z[i])^2 + (β[ptr[i]] - z2[i])^2
        dual_norm += Δz[i]^2 + Δz2[i]^2
        prim_size += β[ptr[i]]^2
        dual_size += u[i]^2 + u2[i]^2
    end

    # convergence in terms of relative norms
    prim_size = √prim_size
    dual_size = α * √dual_size
    prim_norm =  √prim_norm
    dual_norm = α * √dual_norm

    # convergence in terms of relative norms
    if prim_size > 0. && dual_size > 0.
        ε_p = εabs_p / prim_size + model.reltol
        ε_d = εabs_d / dual_size + model.reltol
        converged  = (prim_norm < ε_p) && (dual_norm < ε_d)
    else
        converged = (prim_size == 0) # this is a trivial extreme case
    end

    # store norms if necessary
    if model.save_norms
        push!(model.prim_norms, prim_norm)
        push!(model.dual_norms, dual_norm)
    else
        model.prim_norm = [prim_norm]
        model.dual_norm = [dual_norm]
    end

    converged, prim_norm, dual_norm, prim_size, dual_size
end

@inline function inflate_penalty!(
    model::BinomialGFEN,
    u::Vector{Float64},
    u2::Vector{Float64},
    currentα::Float64,
    edgewts::Vector{Float64},
    edgewts2::Vector{Float64},
    prim_norm::Float64,
    dual_norm::Float64,
    prim_size::Float64,
    dual_size::Float64,
)
    maxgap = model.admm_residual_balancing_gap

    if model.admm_adaptive_inflation
        ρ = sqrt(prim_norm / dual_norm)
        κ = max(ρ, 1.0 / ρ)
    else
        κ = model.admm_fixed_inflation_factor
    end
    if currentα * κ < model.admm_min_penalty
        κ = model.admm_min_penalty / currentα
    end

    α = currentα
    if (prim_norm / prim_size) > maxgap * (dual_norm / dual_size)
        α *= κ
        for j in eachindex(u)
            u[j] /= κ
            u2[j] /= κ
            edgewts[j] /= κ
            edgewts2[j] /= κ
        end
    elseif (dual_norm / dual_size) > maxgap * (prim_norm / prim_size)
        α /= κ
        for j in eachindex(u)
            u[j] *= κ
            u2[j] *= κ
            edgewts[j] *= κ
            edgewts2[j] *= κ
        end
    end
    return α
end

"""
ADMM algorithm for graph fused lasso
"""
function fit!(
        model::BinomialGFEN,
        successes::Vector{Float64},
        attempts::Vector{Float64};
        steps::Int = typemax(Int),
        walltime::Float64 = Inf,
        parallel::Bool = false)

    # init algorithm variables, see init_vars code for symbol descriptions
    β, z, u, Δz, Δu, z2, u2, Δz2, Δu2, α, edgewts, edgewts2, num_visits, εabs_p, εabs_d = init_vars(model, successes, attempts)

    step = 0
    converged = model.converged
    maxsteps = min(model.maxsteps - model.steps, steps)

    @fastmath @inbounds begin # macros for speed
    traintime = 0.0
    while !converged && step < maxsteps && traintime < walltime
        # add to train time this iteration time
        traintime += @elapsed begin

        # update variables
        step += 1
        update_primal!(model, β, z, u, z2, u2, num_visits, successes, attempts, α)
        update_slack_and_dual!(model, z, u, Δz, Δu, z2, u2, Δz2, Δu2, β, edgewts, edgewts2, parallel)

        # evaluate convergence and balance residuals
        if step % model.check_convergence_every == 0
            if model.save_loss
                loss = compute_loss(model, β, successes, attempts)
                push!(model.loss, loss)
            end
            converged, prim_norm, dual_norm, prim_size, dual_size = residual_norms!(model, β, z, u, Δz, z2, u2, Δz2, α, εabs_p, εabs_d)
            converged && break

            if model.admm_residual_balancing && step % model.admm_balance_every == 0
                α = inflate_penalty!(model, u, u2, α, edgewts, edgewts2, prim_norm, dual_norm, prim_size, dual_size)
            end
        end

        end # add iteration time to train_time
    end # closes while loop
    end # closes  macros for speed

    model.trained = true
    model.beta = β
    model.slack = z
    model.dual = u
    model.admm_penalty = α
    model.steps += step
    model.converged  = converged

    converged
end

function predict(model::BinomialGFEN; probs::Bool = true)
    @assert model.trained "model hasn't been trained, use fit! first"
    probs ? [1. / (1. + exp(-βᵢ)) for βᵢ in model.beta] : model.beta
end
