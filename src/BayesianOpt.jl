using Distributions
using LinearAlgebra
using Distances


function rbfkernel(X::AbstractArray{Float64}, a::Float64)
    exp.(- a .* pairwise(SqEuclidean(), X, dims=2))
end


struct GaussianProcessSampler
    X::AbstractMatrix{Float64}
    y::AbstractVector{Float64}
    tested::AbstractVector{Bool}
    a::Float64
    σ::Float64
    K::Matrix{Float64}
    b::Float64

    function GaussianProcessSampler(
            X::Array{Float64},
            y::Vector{Float64};
            tested::Union{Nothing, AbstractVector{Bool}}=nothing,
            σ::Float64=0.001,
            a::Float64=0.5,
            b::Float64=1.0)
        N = length(y)
        X = reshape(X, :, N)
        if tested === nothing
            tested = zeros(Bool, N)
        end
        K = b * rbfkernel(X, a)
        new(X, y, tested, a, σ, K, b)
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
    @assert n <= length(gp.y)
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
    #
    f = K[:, s] * ((K[s, s] + (σ2 + 1e-12)*I) \ y[s])
    Σ = K - K[:, s] * ((K[s, s] + σ2*I) \ K[s, :]) 
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
    return idx, vals, Xidx
end


function gpeval(gp::GaussianProcessSampler)
    # gp regression mean and variance
    σ2 = gp.σ^2
    s = gp.tested
    K = gp.K
    y = gp.y
    N = length(gp.y)
    #
    f = K[:, s] * ((K[s, s] + (σ2 + 1e-12)*I) \ y[s])
    Σ = K - K[:, s] * ((K[s, s] + σ2*I) \ K[s, :]) 
    band = diag(Σ)
    return f, band
end


#  make sure it works fine!
# using Plots
# function test()
#     N = 200
#     a = 0.5
#     σ = 0.5
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
#     gp = GaussianProcessSampler(
#         x, y, tested=copy(seen),
#         a=a, σ=σ, b=b)
#     f, band = gpeval(gp) 
#     p1 = plot(x, ytruth, label="truth", linestyle=:dash, color="black")
#     plot!(p1, x, f, ribbon=1.96 * band, fillalpha=.2, label="fit_0", color="blue")
#     plot!(p1, x[seen], gp.y[seen], st = :scatter, label="candidate_0", color="blue", alpha=0.4)
#     ylims!(p1, -2.5, 2.5)
    
#     iu, yu, _ = gpsample(gp, 5)
#     p2 = plot(x[iu], yu, st = :scatter, label="candidate_1", color="red", alpha=0.4)
#     obs = draw.(x[iu], σ)
#     plot!(p2, x[iu], obs, st = :scatter, label="obs_1", color="blue", alpha=0.4)
#     addobs!(gp, iu, obs)
#     f, band = gpeval(gp) 
#     plot!(p2, x, ytruth, label="truth", linestyle=:dash, color="black")
#     plot!(p2, x, f, ribbon=1.96 * band, fillalpha=.2, label="fit_1", color="red")
#     ylims!(p2, -2.5, 2.5)

#     iu, yu, _ = gpsample(gp, 5)
#     p3 = plot(x[iu], yu, st = :scatter, label="candidate_2", color="red", alpha=0.4)
#     obs = draw.(x[iu], σ)
#     plot!(p3, x[iu], obs, st = :scatter, label="obs_2", color="blue", alpha=0.4)
#     addobs!(gp, iu, obs)
#     f, band = gpeval(gp) 
#     plot!(p3, x, ytruth, label="truth", linestyle=:dash, color="black")
#     plot!(p3, x, f, ribbon=1.96 * band, fillalpha=.2, label="fit_2", color="red")
#     ylims!(p3, -2.5, 2.5)

#     iu, yu, _ = gpsample(gp, 5)
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