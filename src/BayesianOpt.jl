using Distributions
using LinearAlgebra
using Distances
using Statistics

##

function rbfkernel(X::AbstractArray{Float64}, a::Float64)
    exp.(- a .* pairwise(SqEuclidean(), X, dims=2))
end

function rbfkernel(
    X::AbstractArray{Float64},
    Z::AbstractArray{Float64},
    a::Float64
)
    exp.(- a .* pairwise(SqEuclidean(), X, Z, dims=2))
end

mutable struct GaussianProcessSampler
    X::AbstractMatrix{Float64}
    y::AbstractVector{Float64}
    tested::AbstractVector{Bool}
    a::Float64
    σ::Float64
    K::Matrix{Float64}
    b::Float64
    offset::Float64

    function GaussianProcessSampler(
            X::Array{Float64};
            dim::Int=ndims(X),
            σ::Float64=0.001,
            a::Float64=0.5,
            b::Float64=1.0,
            offset::Float64=0.0)
        N = size(X, dim)
        y = zeros(N)
        tested = zeros(Bool, N)
        K = b * rbfkernel(X, a)
        new(X, y, tested, a, σ, K, b, offset)
    end
end


function addobs!(
        gp::GaussianProcessSampler,
        pos::Vector{Int},
        ynew::Vector{Float64})
    for (i, yi) in zip(pos, ynew)
        gp.tested[i] = true
        gp.y[i] = yi
    end
end


function gpsample(gp::GaussianProcessSampler, n::Int=1)
    # gp regression mean and variance
    @assert n <= length(gp.y) - sum(gp.tested)
    if sum(gp.tested) == 0  # choose at random
        M = length(gp.y)
        idx = sample(1:M, n, replace=false)
        vals = [-Inf for _ in 1:n]
        Xidx = gp.X[:, idx]
        return idx, vals, Xidx
    end
    y = gp.y
    σ2 = gp.σ^2
    s = gp.tested
    K = gp.K
    offset = gp.offset
    #
    A = K[s, s] + (σ2 + 1e-12)*I
    β = y[s] .- offset
    f = K[:, s] * (A \ β) .+ offset
    Σ = K - K[:, s] * (A \ K[s, :]) 
    # sample from multivariate normal
    # and take elements with higher sampled values
    Σ = Symmetric(Σ + 1e-12I)
    vals = Float64[]
    idx = Int[]
    idx_ = Set{Int}()
    for _ in 1:n
        W = rand(MvNormal(f, Σ))
        Z = sortperm(W, rev=true)
        i = 1
        while Z[i] ∈ idx_
            i += 1
        end
        push!(idx, Z[i])
        push!(idx_, Z[i])
        push!(vals, W[Z[i]])
    end
    Xidx = gp.X[:, idx]
    predidx = f[idx]
    bandidx = sqrt.(diag(Σ)[idx])
    return idx, vals, Xidx, predidx, bandidx
end


function gpeval(gp::GaussianProcessSampler)
    # gp regression mean and variance
    @assert sum(gp.tested) > 0
    σ2 = gp.σ^2
    s = gp.tested
    K = gp.K
    y = gp.y
    N = length(gp.y)
    offset = gp.offset
    #
    A = K[s, s] + (σ2 + 1e-12)*I
    β = y[s] .- offset
    f = K[:, s] * (A \ β) .+ offset
    Σ = K - K[:, s] * (A \ K[s, :]) 
    band = sqrt.(diag(Σ))
    return f, band
end

##

mutable struct RandomGaussianProcessSampler
    dists::Vector{T} where T <: Distribution
    X::Vector{Vector{Float64}}
    y::Vector{Float64}
    a::Float64
    σ::Float64
    b::Float64
    offset::Float64
    adaptive_normalization::Bool

    function RandomGaussianProcessSampler(
        dists::Vector{T};
        σ::Float64=0.001,
        a::Float64=0.5,
        b::Float64=1.0,
        offset::Float64=0.0,
        adaptive_normalization::Bool=false
    ) where T <: Distribution
        X = Vector{Float64}[]
        y = Float64[]
        new(dists, X, y, a, σ, b, offset, adaptive_normalization)
    end
end
Base.length(gp::RandomGaussianProcessSampler) = length(gp.X)


function addobs!(
    gp::RandomGaussianProcessSampler,
    xnew::Vector{Float64},
    ynew::Float64
)
    # update X, y
    push!(gp.X, xnew)
    push!(gp.y, ynew)
    # todo: eval kernel smartly on the fly
end


function gpsample(
    gp::RandomGaussianProcessSampler;
    batch_size::Int = 100
)
    # gp regression mean and variance
    n = batch_size
    candidates = [rand(d, n) for d in gp.dists]
    if length(gp) == 0  # choose at random
        pars = vcat([x' for x in candidates]...)
        pred = fill(-Inf, n)
        band = zeros(n)
        return pars, pred, band
    else
        y = gp.y
        if gp.adaptive_normalization
            μ_seen = mean(y)
            σ_seen = std(y)
            y = (y .- μ_seen) ./ (σ_seen + eps())
        end
        Xoldmat = hcat([x for x in gp.X]...)
        Xnewmat = vcat([x' for x in candidates]...)
        K00 = gp.b * rbfkernel(Xoldmat, gp.a)
        K11 = gp.b * rbfkernel(Xnewmat, gp.a)
        K01 = gp.b * rbfkernel(Xoldmat, Xnewmat, gp.a)
        K10 = K01'
        offset = gp.offset
        s = length(gp)
        #
        σ2 = gp.σ^2
        A = K00 + (σ2 + 1e-12)*I
        β = y .- offset
        pred = K10 * (A \ β) .+ offset
        Σ = K11 - K10 * (A \ K01) 
        Σ = Symmetric(Σ + 1e-12I)
        W = rand(MvNormal(pred, Σ))
        band = sqrt.(diag(Σ))
        order = sortperm(pred, rev=true)
        pred = pred[order]
        band = band[order]
        Xnewmat = Xnewmat[:, order]
        if gp.adaptive_normalization
            pred = (σ_seen + eps()) .* pred .+ μ_seen
            band = (σ_seen + eps()) .* band
        end
        return Xnewmat, pred, band
    end
end


function gpsample(
    gp::RandomGaussianProcessSampler,
    n::Int;
    kwargs...
)
    xs = []
    ps = Float64[]
    bs = Float64[]
    for i in 1:n
        x, p, b = gpsample(gp; kwargs...)
        push!(xs, x[:, 1])
        push!(ps, p[1])
        push!(bs, b[1])
    end
    xs = hcat(xs...)
    xs, ps, bs
end

function gpeval(gp::RandomGaussianProcessSampler)
    # gp regression mean and variance
    @assert length(gp) > 0  "add data first"
    y = gp.y
    if gp.adaptive_normalization
        μ_seen = mean(y)
        σ_seen = std(y)
        y = (y .- μ_seen) ./ (σ_seen + eps())
    end
    σ2 = gp.σ^2
    X = hcat([x for x in gp.X]...)
    (size(X, 2) ≥ 1_000) && warn("kernel matrix too large, try sampling less candidates")
    K = gp.b * rbfkernel(X, gp.a)
    offset = gp.offset
    s = length(gp)
    #
    A = K + (σ2 + 1e-12)*I
    β = y .- offset
    f = K * (A \ β) .+ offset
    Σ = K - K * (A \ K) 
    Σ = Symmetric(Σ + 1e-12I)
    band = sqrt.(diag(Σ))
    if gp.adaptive_normalization
        f = (σ_seen + eps()) .* f .+ μ_seen
        band = (σ_seen + eps()) .* band
    end
    return f, band
end

function gpeval(
    gp::RandomGaussianProcessSampler,
    Xnew::Vector{Vector{Float64}}
)
    # gp regression mean and variance
    if length(gp) == 0  # choose at random
        pars = hcat([x for x in Xnew]...)
        n = length(Xnew)
        pred = fill(-Inf, n)
        band = zeros(n)
        return pred, band
    else
        y = gp.y
        if gp.adaptive_normalization
            μ_seen = mean(y)
            σ_seen = std(y)
            y = (y .- μ_seen) ./ (σ_seen + eps())
        end
        σ2 = gp.σ^2
        Xoldmat = hcat([x for x in gp.X]...)
        Xnewmat = hcat([x for x in Xnew]...)
        K00 = gp.b * rbfkernel(Xoldmat, gp.a)
        K11 = gp.b * rbfkernel(Xnewmat, gp.a)
        K01 = gp.b * rbfkernel(Xoldmat, Xnewmat, gp.a)
        K10 = K01'
        offset = gp.offset
        s = length(gp)
        #
        A = K00 + (σ2 + 1e-12)*I
        β = y .- offset
        pred = K10 * (A \ β) .+ offset
        Σ = K11 - K10 * (A \ K01) 
        Σ = Symmetric(Σ + 1e-12I)
        band = sqrt.(diag(Σ))
        if gp.adaptive_normalization
            pred = (σ_seen + eps()) .* pred .+ μ_seen
            band = (σ_seen + eps()) .* band
        end
        return pred, band
    end
end


## test random gp 
# using Plots
# N = 4
# a = 0.5
# σ = 0.05
# b = 1.0 / √(2π) * 2a
# scale = 100.0
# offset = -10.0 / scale
# x = collect(range(0., stop=2π, length=N))
# y = zeros(N)
# draw(x::Float64, σ::Float64) = (sin(x) + σ * randn()) / scale + offset
# for i in 1:N
#     y[i] = draw(x[i], σ)
# end
# ytruth = sin.(x) ./ scale .+ offset
# dists = [Uniform(0.0, 2π)]
# gp = RandomGaussianProcessSampler(dists, a=a, σ=σ, b=b, adaptive_normalization=true)

# for i in 1:N
#     addobs!(gp, [x[i]], y[i])
# end

# @time pars, pred, band = gpsample(gp, 16; batch_size=500)
# x_sample = vec(pars)
# y_sample = pred

# xseq = [[xi] for xi in collect(range(0., stop=2π, length=50))]
# pred_eval, band_eval = gpeval(gp, xseq)
# xseq = vcat(xseq...)

# p1 = plot(x, ytruth, label="truth", linestyle=:dash, color="black")
# plot!(p1, vec(xseq), pred_eval, ribbon=1.96 * band_eval, fillalpha=.2, label="fit_0", color="blue")
# plot!(p1, x, gp.y, st = :scatter, label="candidate_0", color="blue", alpha=0.4)
# plot!(p1, x_sample, y_sample, st = :scatter, label="candidate_1", color=:red)
# ylims!(p1, -2.5, 2.5)


##


# #  make sure it works fine!
# using Plots
# function test()
#     N = 200
#     a = 0.5
#     σ = 0.1
#     b = 1.0
#     x = collect(range(0., stop=2π, length=N))
#     y = zeros(N)
#     seen = zeros(Bool, N)
#     seen[[1, div(N, 2), N]] .= true
#     draw(x::Float64, σ::Float64) = sin(x) + σ * randn()
#     for i in findall(seen)
#         y[i] = draw(x[i], σ)
#     end
#     ytruth = sin.(x)
#     gp = GaussianProcessSampler(x, a=a, σ=σ, b=b)
#     addobs!(gp, findall(seen), y[seen])
#     f, band = gpeval(gp) 
#     p1 = plot(x, ytruth, label="truth", linestyle=:dash, color="black")
#     plot!(p1, x, f, ribbon=1.96 * band, fillalpha=.2, label="fit_0", color="blue")
#     plot!(p1, x[seen], gp.y[seen], st = :scatter, label="candidate_0", color="blue", alpha=0.4)
#     ylims!(p1, -2.5, 2.5)

#     iu, yu, _, _, _ = gpsample(gp, 5)
#     p2 = plot(x[iu], yu, st = :scatter, label="candidate_1", color="red", alpha=0.4)
#     obs = draw.(x[iu], σ)
#     plot!(p2, x[iu], obs, st = :scatter, label="obs_1", color="blue", alpha=0.4)
#     addobs!(gp, iu, obs)
#     f, band = gpeval(gp) 
#     plot!(p2, x, ytruth, label="truth", linestyle=:dash, color="black")
#     plot!(p2, x, f, ribbon=1.96 * band, fillalpha=.2, label="fit_1", color="red")
#     ylims!(p2, -2.5, 2.5)

#     iu, yu, _, _, _ = gpsample(gp, 5)
#     p3 = plot(x[iu], yu, st = :scatter, label="candidate_2", color="red", alpha=0.4)
#     obs = draw.(x[iu], σ)
#     plot!(p3, x[iu], obs, st = :scatter, label="obs_2", color="blue", alpha=0.4)
#     addobs!(gp, iu, obs)
#     f, band = gpeval(gp) 
#     plot!(p3, x, ytruth, label="truth", linestyle=:dash, color="black")
#     plot!(p3, x, f, ribbon=1.96 * band, fillalpha=.2, label="fit_2", color="red")
#     ylims!(p3, -2.5, 2.5)

#     iu, yu, _, _, _ = gpsample(gp, 5)
#     p4 = plot(x[iu], yu, st = :scatter, label="candidate_3", color="red", alpha=0.4)
#     obs = draw.(x[iu], σ)
#     plot!(p4, x[iu], obs, st = :scatter, label="obs_3", color="blue", alpha=0.4)
#     addobs!(gp, iu, obs)
#     f, band = gpeval(gp) 
#     plot!(p4, x, ytruth, label="truth", linestyle=:dash, color="black")
#     plot!(p4, x, f, ribbon=1.96 * band, fillalpha=.2, label="fit_3", color="red")
#     ylims!(p4, -2.5, 2.5)

#     return plot(p1, p2, p3, p4, layout=(2, 2), size=(1000, 600))
# end
# test()
# savefig(test(), "bayesian_optimization_example.png")