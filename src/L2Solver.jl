# Implements a simple versino of the Kalman smoother

using DataStructures

function filter1Dl2!(
    β::AbstractVector{Float64},
    y::AbstractVector{Float64},
    λ::AbstractVector{Float64},
    from::Int,
    to::Int
)
    # initialise with improper prior
    β[from] = y[from]
    ψstack = Stack{Float64}()
    push!(ψstack, 1.0)
    
    @simd for t in from:(to - 1)
        ψ = top(ψstack)
        ω = λ[t] * ψ / (λ[t] + ψ)
        β[t + 1] = (y[t + 1] + ω * β[t]) / (1.0 + ω)
        push!(ψstack, 1.0 + ω)
    end
    pop!(ψstack)
    
    ϕ = 1.0
    η = y[to]
    @simd for t in reverse(from:(to - 1))
        ψ = pop!(ψstack)
        ω = ϕ * λ[t] / (ϕ + λ[t])
        β[t] = (ψ * β[t] + ω * η) / (ψ + ω) 
        ϕ =  1.0 + ω
        η = (y[t] + ω * η) / ϕ
    end
end

function filter1Dl2(
    y::AbstractVector{Float64},
    λ::AbstractVector{Float64},
    from::Int,
    to::Int
)
    β = zeros(to - from + 1)
    filter1Dl2!(β, y, λ, from, to)
    return β
end

function filter1Dl2!(
    β::AbstractVector{Float64},
    y::AbstractVector{Float64},
    λ::AbstractVector{Float64}
)
    filter1Dl2!(β, y, λ, 1, length(y))
end

function filter1Dl2(
    y::AbstractVector{Float64},
    λ::AbstractVector{Float64}
)
    filter1Dl2(y, λ, 1, length(y))
end