
using Random # Random stdlib
import ForwardDiff: derivative
import StatsBase: sample, weights # To include the basic sample from array function
# export Line, Envelop, RejectionSampler # Structures/classes
# export run_sampler!, sample_envelop, eval_envelop, add_segment! # Methods

"""
    Line(slope::Float64, intercept::Float64)
Basic tangent ensamble-unit for an envelop.
"""
struct Line
    slope::Float64
    intercept::Float64
end
eval_line(l::Line, x::Float64) = l.slope * x + l.intercept

"""
    intersection(l1::Line, l2::Line)
Finds the horizontal coordinate of the intersection between lines
"""
function intersection(l1::Line, l2::Line)
    @assert l1.slope ≥ l2.slope "slopes should be weakly ordered (l1: $(l1.slope), l2: $(l2.slope))"
    @assert l1.slope != l2.slope
    - (l2.intercept - l1.intercept) / (l2.slope - l1.slope)
end

"""
    exp_integral(l::Line, x1::Float64, x2::Float64)
Computes the integral
    ``LaTeX \\int_{x_1} ^ {x_2} \\exp\\{ax + b\\} dx. ``
The resulting value is the weight assigned to the segment [x1, x2] in the envelop
"""
function exp_integral(l::Line, x1::Float64, x2::Float64)::Tuple{Float64, Float64}
    (x1 == x2) && (return (0.0, 0.0))
    @assert x1 ≤ x2 "invalid x1=$x1, x2=$x2, $l"
    a, b = l.slope, l.intercept
    # for numerical stability: multiply by 0.5 * a(x1 - x2)
    if a != 0.0
        z1, z2 = eval_line(l, x1), eval_line(l, x2)
        scale = max(z1, z2)
        out = (exp(z2 - scale) - exp(z1 - scale)) / a
    else
        scale = 0.0
        out = (x2 - x1) * exp(b)
    end
    @assert out ≥ 0 "negative weight (out: $out, x1: $x1, x2: $x2, l: $l, scale: $scale"
    # debug info
    #  println("  @exp_integral:")
    #  println("  |-- line: $l")
    #  println("  |-- x1: $x1") 
    #  println("  |-- x2: $x2")
    #  println("  |-- exp(b + a * x2): $(exp(b + a * x2))")
    #  println("  |-- exp(b + a * x1): $(exp(b + a * x1))")
    #  println("  |-- exp(a * x2): $(exp(a * x2))")
    #  println("  |-- exp(a * x1): $(exp(a * x1))")
    #  println("  |-- out: $out")
    return out, scale
end

"""
    Envelop(lines::Vector{Line}, support::Tuple{Float64, Float64})
A piecewise linear function with k segments defined by the lines `L_1, ..., L_k` and cutpoints
`c_1, ..., c_k+1` with `c1 = support[1]` and `c2 = support[2]`. A line L_k is active in the segment
[c_k, c_k+1], and it's assigned a weight w_k based on [exp_integral](@exp_integral). The weighted integral
over c_1 to c_k+1 is one, so that the envelop is interpreted as a density.
"""
struct Envelop
    lines::Vector{Line}
    cutpoints::Vector{Float64}
    weights::Vector{Tuple{Float64, Float64}}
    secant_knots::Vector{Float64}
    secant_values::Vector{Float64}

    Envelop(
        lines::Vector{Line},
        support::Tuple{Float64, Float64},
        secant_knots::Vector{Float64} = Float64[],
        secant_values::Vector{Float64} = Float64[]
    ) = begin
        @assert length(secant_knots) == length(secant_values)
        #  println("lines: $lines")
        @assert issorted([l.slope for l in lines], rev = true) "line slopes must be decreasing"
        intersections = [intersection(lines[i], lines[i + 1]) for i in 1:(length(lines) - 1)]
        #  println("intersections: $intersections")
        cutpoints = [support[1]; intersections; support[2]]
        #  println("cutpoints: $cutpoints")
        weights = [exp_integral(l, cutpoints[i], cutpoints[i + 1]) for (i, l) in enumerate(lines)]
        # println("weights: $weights")
        new(lines, cutpoints, weights, secant_knots, secant_values)
    end
end
Base.length(e::Envelop) = length(e.lines)


function eval_secants(e::Envelop, x::Float64; log::Bool=false)
    pos = searchsortedfirst(e.secant_knots, x)
    if pos == 1 || pos == length(e) + 1
        out =- Inf
    else
        x1 = e.secant_knots[pos - 1]
        y1 = e.secant_values[pos - 1]
        x2 = e.secant_knots[pos]
        y2 = e.secant_values[pos]
        out = (y2 - y1) / (x2 - x1) * (x - x1) + y1
    end
    !log && (out = exp(out))
    return out
end


# struct SecantBuffer
#     secants::Vector{Buffer}
# end 
# len(s::SecantBuffer) = length(s.secants)

# function eval_secant_buffer(s::SecatBuffer, x::Float64)
#     for sec in s.secants
#         (sec.x1 ≥ x) && (return eval_secant(sec, x))
#     end
#     error("point $x outside secant range $(s.secants[1].x1, s.secants[end].x2)")
# end


"""
    add_segment!(e::Envelop, l::Line)
Adds a new line segment to an envelop based on the value of its slope (slopes must be decreasing
always in the envelop). The cutpoints are automatically determined by intersecting the line with
the adjacent lines.
"""
function add_segment!(e::Envelop, l::Line)
    #  println("@add_segment")
    #  println("|-- adding segment $l")
    # printstyled("|-- envelop: $e\n", color=:blue)
    # Find the position in sorted array with binary search
    pos = searchsortedfirst([-line.slope for line in e.lines], -l.slope)
    # printstyled("|-- adding $l to position $pos\n", color=:green)
    # remember cutpoints has the support at the beginning and end!
    # Find the new cutpoints
    if pos == 1
        new_cut = intersection(l, e.lines[pos])  # smaller than all values
        # printstyled("|-- adding cut $(new_cut)\n", color=:yellow)
        # Insert in second position, first one is the support bound
        # 1. measure weight from support to new first cut
        # 2. measure weight from new cut to previous first cut after support
        ws1 = exp_integral(l, e.cutpoints[1], new_cut)  # support to new
        ws2 = exp_integral(e.lines[1], new_cut, e.cutpoints[2])  # new to next
        insert!(e.cutpoints, 2, new_cut)
        splice!(e.weights, 1, [ws1, ws2])
        # printstyled("|-- new weights: ($ws1, $ws2)\n", color=:green)
        # printstyled("|-- new envelop: $e\n", color=:blue)
        # error("early exit")
        # must update weights for first two segments only
    elseif pos == length(e) + 1  # greater than all values
        new_cut = intersection(e.lines[end], l)
        # printstyled("|-- adding cut $(new_cut)\n", color=:yellow)
        # 1. measure weight from new last cut to support
        # 2. measure weight from new cut to previous first cut after support
        ws1 = exp_integral(e.lines[end], e.cutpoints[end - 1], new_cut)
        ws2 = exp_integral(l, new_cut, e.cutpoints[end])
        insert!(e.cutpoints, length(e) + 1, new_cut)  # just before support end
        splice!(e.weights, length(e), [ws1, ws2])
        # printstyled("|-- new weights: ($ws1, $ws2)\n", color=:green)
        # printstyled("|-- new envelop: $e\n", color=:blue)
        # error("early exit")
    else
        (l.slope == e.lines[pos].slope) && return -1 # can't add segment
        new_cut1 = intersection(e.lines[pos - 1], l)
        new_cut2 = intersection(l, e.lines[pos])
        supp = e.cutpoints[1], e.cutpoints[end]
        @assert supp[1] ≤ new_cut1 ≤ new_cut2 "new_cut1 ≤ supp[2]: $new_cut1, new_cut2: $new_cut2, supp: $supp"
        # println("Current cuts: $(e.cutpoints)")
        # printstyled("|-- adding cuts $(new_cut1) and $(new_cut2) from tangent $(l.tangent_point)\n", color=:yellow)
        # must update three weights!
        ws1 = exp_integral(e.lines[pos - 1], e.cutpoints[pos - 1], new_cut1)
        ws2 = exp_integral(l, new_cut1, new_cut2)
        ws3 = exp_integral(e.lines[pos], new_cut2, e.cutpoints[pos + 1])
        # printstyled("|-- new weights: ($ws1, $ws2, $ws3)\n", color=:green)
        splice!(e.cutpoints, pos, [new_cut1, new_cut2])
        splice!(e.weights, (pos - 1):pos, [ws1, ws2, ws3])
        # printstyled("|-- new envelop: $e\n", color=:blue)
        # error("early exit")
        # @assert issorted(e.cutpoints)  "incompatible line: resulting intersection points aren't sorted"
    end
    # Insert the new line
    insert!(e.lines, pos, l)
    return pos
    # e.weights = [exp_integral(line, e.cutpoints[i], e.cutpoints[i + 1]) for (i, line) in enumerate(e.lines)]
end

"""
    sample_envelop(rng::AbstractRNG, e::Envelop)
    sample_envelop(e::Envelop)
Samples an element from the density defined by the envelop `e` with it's exponential weights.
See [`Envelop`](@Envelop) for details.
"""
function sample_envelop(rng::AbstractRNG, e::Envelop)
    # Randomly select a segment from the weights
    # scale each weight accordingly for numerical stability
    max_scale = maximum(s for (_, s) in e.weights)
    rel_wts = weights([w * exp(s - max_scale) for (w, s) in e.weights])
    i = sample(rng, 1:length(e), rel_wts)
    
    # use the inverse CDF method to sample within segment
    #  println("Sampling $i from current weights $(e.weights)")
    # println("Sampling with weights $rel_wts, max_scale: $max_scale, x1: $x1")
    l = e.lines[i]
    denom, scale = e.weights[i]
    a, b = e.lines[i].slope, e.lines[i].intercept
    x1, x2 = e.cutpoints[i:(i + 1)]
    u = rand(rng)
    if a == 0.0
        out = x1 + u * (x2 - x1)
    else
        # out = (log(a * U * denom + exp(eval_line(e.lines[i], x1) - scale)) - b + scale) / a
        # out = (
        #     log(exp(-b + scale) * u * denom * a + exp(eval_line(l, x1) - scale))
        #     - b + scale
        # ) / a
        out = (
            log(u * e.weights[i][1] * a + exp(a * x1 + b - scale))
            -b + scale
        ) / a

    end

    #  println("Sampled $out from segment [$(e.cutpoints[i]), $(e.cutpoints[i+1])]")
    return out
end

function sample_envelop(e::Envelop)
    sample_envelop(Random.GLOBAL_RNG, e)
end


"""
    eval_envelop(e::Envelop, x::Float64)
Eval point a point `x` in the piecewise linear function defined by `e`. Necessary for evaluating
the density assigned to the point `x`.
"""
function eval_envelop(e::Envelop, x::Float64; log::Bool = false)
    # searchsortedfirst is the proper method for and ordered list
    pos = searchsortedfirst(e.cutpoints, x)
    if pos == 1 || pos == length(e.cutpoints) + 1
        return log ? - Inf : 0.0
    else
        pos = searchsortedfirst(e.cutpoints, x)
        logv = eval_line(e.lines[pos - 1], x)
        return log ? logv : exp(logv)
    end
end


# --------------------------------


"""
    RejectionSampler(f::Function, support::Tuple{Float64, Float64}[ ,δ::Float64])
    RejectionSampler(f::Function, support::Tuple{Float64, Float64}, init::Tuple{Float64, Float64})
An adaptive rejection sampler to obtain iid samples from a logconcave function `f`, supported in the
domain `support` = (support[1], support[2]). To create the object, two initial points `init = init[1], init[2]`
such that `logf'(init[1]) > 0` and `logf'(init[2]) < 0` are necessary. If they are not provided, the constructor
will perform a greedy search based on `δ`.

The argument `support` must be of the form `(-Inf, Inf), (-Inf, a), (b, Inf), (a,b)`, and it represent the
interval in which f has positive value, and zero elsewhere.

## Keyword arguments
- `max_segments::Int = 10` : max size of envelop, the rejection-rate is usually slow with a small number of segments
- `max_failed_factor::Float64 = 0.001`: level at which throw an error if one single sample has a rejection rate
    exceeding this value
"""
struct RejectionSampler
    objective::Function
    envelop::Envelop
    max_segments::Int
    max_failed_rate::Float64
    support::Tuple{Float64, Float64}
    use_secants::Bool
    # Constructor when initial points are provided
    RejectionSampler(
        f::Function,
        support::Tuple{Float64, Float64},
        init::Tuple{Union{Float64, Nothing}, Union{Float64, Nothing}};
        apply_log::Bool = true,
        max_segments::Int = 25,
        max_failed_rate::Float64 = 0.001,
        autograd::Bool = true,
        use_secants::Bool = false
    ) = begin
        # check support
        logf = apply_log ? (x -> log(f(x))) : f
        objective = autograd ? (x -> (logf(x), derivative(logf, x))) : logf
        @assert support[1] < support[2] "invalid support, not an interval"

        # check initial cutpoints
        x1, x2 = init
        isnothing(x1) && (x1 = support[1])
        isnothing(x2) && (x2 = support[2])

        @assert !isinf(x1) "initial left point can't be infinite if support unbounded below"
        @assert !isinf(x2) "initial right point can't be infinite if support unbounded above"
        @assert x1 < x2 "Initial points ($x1, $x2) must be ordered"

        # validate slopes
        lines = Line[]
        secant_knots = Float64[]
        secant_values = Float64[]

        if x1 != support[1]
            # println(objective(x1))
            y1, a1 = objective(x1)
            @assert !isinf(support[1]) || a1 > 0.0 "slope must be positive in left init when unbounded below"
            b1 = y1 - a1 * x1
            line1 = Line(a1, b1)
            push!(lines, line1)
            if use_secants
                push!(secant_knots, x1)
                push!(secant_values, y1)
            end
        end
        if x2 != support[2]
            y2, a2 = objective(x2)
            @assert !isinf(support[2]) || a2 < 0.0 "slope must be positive in right init when unbounded above"
            b2 = y2 - a2 * x2
            line2 = Line(a2, b2)
            push!(lines, line2)
            if use_secants
                push!(secant_knots, x2)
                push!(secant_values, y2)
            end
        end
            
        if isempty(lines)
             # special case when starting with support only bounds
            @assert !isinf(support[1]) && !isinf(support[2]) "when not providing initial points, support must be bounded"
            xmid = 0.5(support[1] + support[2])
            ymid, amid = objective(xmid)
            bmid = ymid - amid * xmid
            line12 = Line(amid, bmid)
            push!(lines, line12)
            if use_secants
                push!(secant_knots, xmid)
                push!(secant_values, ymid)
            end
        end
        envelop = Envelop(lines, support, secant_knots, secant_values)
        # println("initial envelop: $envelop")
        new(
            objective,
            envelop,
            max_segments,
            max_failed_rate,
            support,
            use_secants
        )
    end

    # Constructor for greedy search of starting points
    RejectionSampler(
            f::Function,
            support::Tuple{Float64, Float64},
            δ::Float64;
            search_range::Tuple{Float64, Float64} = (-10.0,10.0),
            apply_log::Bool = true,
            grad::Union{Function, Nothing} = nothing,
            autograd::Bool = true,
            kwargs...
    ) = begin
        logf = apply_log ? (x -> log(f(x))) : f
        objective = autograd ? (x -> (logf(x), derivative(logf, x))) : logf
        grid_lims = max(search_range[1], support[1]), min(search_range[2], support[2])
        grid = grid_lims[1]:δ:grid_lims[2]
        i1, i2 = findfirst(x -> objective(x)[2] > 0.0, grid), findfirst(objective -> f(x)[2] < 0., grid)
        @assert !isnothing(i1) &&  !isnothing(i2) "couldn't find initial points, please provide them or change `search_range`"
        x1, x2 = grid[i1], grid[i2]
        # pass autograd false since we already checked it
        RejectionSampler(objective, support, (x1, x2); apply_log=false, autograd=false, kwargs...)
    end
end

"""
    run_sampler!(rng::AbstractRNG, sampler::RejectionSampler, n::Int)
    run_sampler!(sampler::RejectionSampler, n::Int)
It draws `n` iid samples of the objective function of `sampler`, and at each iteration it adapts the envelop
of `sampler` by adding new segments to its envelop.
"""
function run_sampler!(rng::AbstractRNG, s::RejectionSampler, n::Int)
    i = 0
    failed, max_failed = 0, trunc(Int, n / s.max_failed_rate)
    out = zeros(n)
    env = s.envelop
    while i < n
        c = sample_envelop(rng, env)
        u = rand(rng)
        majorant = eval_envelop(env, c, log=true)
        if s.use_secants # squeeze test
            if u < exp(eval_secants(env, c, log=true) - majorant)
                i += 1
                out[i] = c
                continue
            end
        end
        failed += 1
        @assert failed < max_failed "max_failed_factor reached"
        y, a = s.objective(c)
        if u < exp(y - majorant) # evalf
            i += 1
            out[i] = c
            (i == n) && break
            # println(c)
            # error("early stop")
        end
        # add to segments
        if length(env) <= s.max_segments
            b = y - a * c
            pos = add_segment!(env, Line(a, b))
            if pos > 0 && s.use_secants
                insert!(env.secant_knots, pos, c)
                insert!(env.secant_values, pos, y)
            end
        end
    end
    out
end

function run_sampler!(s::RejectionSampler, n::Int)
    run_sampler!(Random.GLOBAL_RNG, s, n)
end


function eval_envelop(s::RejectionSampler, x::Float64, kwargs...)
    eval_envelop(s.envelop, x, kwargs...)
end
function eval_envelop(s::RejectionSampler, x::AbstractVector{Float64}, kwargs...)
    [eval_envelop(s.envelop, xᵢ, kwargs...) for xᵢ in x]
end
function eval_secants(s::RejectionSampler, x::Float64, kwargs...)
    eval_secants(s.envelop, x, kwargs...)
end
function eval_secants(s::RejectionSampler, x::AbstractVector{Float64}, kwargs...)
    [eval_secants(s.envelop, xᵢ, kwargs...) for xᵢ in x]
end
