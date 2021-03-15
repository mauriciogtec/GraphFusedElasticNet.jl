using Pkg; Pkg.activate(".")
include("../src/ARS.jl")
using Plots


μ, σ = 0.0, 5.0
support = (-10.0, 1.0)

f(x) = exp(-0.5(x - μ)^2 / σ^2) / sqrt(2pi * σ^2) 

# Build the sampler and simulate 10,000 samples
sampler = RejectionSampler(
    f,
    support,
    (-3.0, -2.0),
    max_segments = 25,
    use_secants=true
)
@time sim = run_sampler!(sampler, 100_000);

x = range(-10.0, 10.0, length=100)
env = sampler.envelop
majorant = eval_envelop(sampler, x)
minorant = eval_secants(sampler, x)
target = f.(x)

histogram(sim, normalize = true, label = "Histogram", fillalpha=0.5)
plot!(x, [target majorant minorant], width = 5, label = ["Target" "Majorant" "Minorant"])


##

# ----------------

f2(x) = begin
    δ = x - μ
    fout = - 0.5δ^2 / σ^2
    gout = - δ / σ^2
    fout, gout
end

# Build the sampler and simulate 10,000 samples
sampler = RejectionSampler(
    f2,
    support,
    (-3.0, -2.0),
    autograd=false,
    apply_log=false,
    max_segments = 25,
    use_secants=true
)
@time sim = run_sampler!(sampler, 100_000);

x = range(-10.0, 10.0, length=100)
env = sampler.envelop
majorant = eval_envelop(sampler, x)
minorant = eval_secants(sampler, x)
target = f.(x)

histogram(sim, normalize = true, label = "Histogram", fillalpha=0.5)
plot!(x, [target majorant minorant], width = 5, label = ["Target" "Majorant" "Minorant"])