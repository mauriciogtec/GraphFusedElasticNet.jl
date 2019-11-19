# This code is my Julia version of the taut string 1d solver of Barbero and Sra
# The original code of the authors is in https://github.com/albarji/proxTV/blob/master/src/TVL1opt_tautstring.cpp
# Copyright (c) 2013, Álvaro Barbero Jiménez and Suvrit Sra. All rights reserved.

import Base.getindex
import Base.setindex!
import Base.show

mutable struct PointXY
    x::Int
    y::Float64
end

mutable struct SegmentBuffer
    Δx::Vector{Int}
    Δy::Vector{Float64}
    slope::Vector{Float64}
    first::Int
    last::Int
    SegmentBuffer(n::Int) = new(zeros(Int, n), zeros(n), zeros(n), 1, 0)
end

# pretty printer of segment buffers for debugging purposes only
function Base.show(io::IO, sb::SegmentBuffer)
    nsegments = sb.last - sb.first + 1
    print(io, "buffer with ", sb.last - sb.first + 1, " segment(s)")
    (nsegments >= 1) && print(io, "\n",
        join(collect(zip(sb.Δx[sb.first:sb.last], sb.Δy[sb.first:sb.last])), "-"))
end

@inline getindex(sb::SegmentBuffer, i::Int) = (sb.Δx[i], sb.Δy[i], sb.slope[i])

@inline setindex!(sb::SegmentBuffer, val::Tuple{Int, Float64, Float64}, i::Int) =
    ((sb.Δx[i], sb.Δy[i], sb.slope[i]) = val; nothing)

@inline num_segments(sb::SegmentBuffer) = sb.last - sb.first + 1

@inline remove_first_segment!(sb::SegmentBuffer) = (sb.first += 1; nothing)

@inline pop_first_segment!(sb::SegmentBuffer) = (sb.first += 1; sb[sb.first - 1])

@inline first_segment(sb::SegmentBuffer) = sb[sb.first]

@inline last_segment(sb::SegmentBuffer) = sb[sb.last]

@inline first_slope(sb::SegmentBuffer) = sb.slope[sb.first]

@inline last_slope(sb::SegmentBuffer) = sb.slope[sb.last]

@inline pop_last_segment!(sb::SegmentBuffer) = (sb.last -= 1; sb[sb.last + 1])

@inline push_to_last!(sb::SegmentBuffer, Δx::Int, Δy::Float64) =
    (sb.last += 1; sb[sb.last] = (Δx, Δy, Δy / float(Δx)); nothing)

@inline restart_from_state!(sb::SegmentBuffer) =
    (sb.first = sb.last + 1; nothing)
    # (sb.first = sb.last; nothing)
    # (sb.first = sb.last; sb.last -= 1; nothing)

@inline function add_segment_to_minorant!(
    minorant::SegmentBuffer,
    Δx::Int,
    Δy::Float64
)
    δx, δy = Δx, Δy
    if δy < float(δx) * last_slope(minorant) # if not convex
        auxi = num_segments(minorant)
        while true # do-while block
            # Merge last two segments
            dx, dy, s = pop_last_segment!(minorant)
            δx += dx
            δy += dy

            # Check if convex again
            auxi -= 1
            (auxi < 1 || δy >= float(δx) * last_slope(minorant)) && break
        end
    end
    push_to_last!(minorant, δx, δy)
end

@inline function add_segment_to_majorant!(
    majorant::SegmentBuffer,
    Δx::Int,
    Δy::Float64
)
    δx, δy = Δx, Δy
    if δy > float(δx) * last_slope(majorant) # if not concave
        auxi = num_segments(majorant)
        while true # do-while block
            # Merge last two segments
            dx, dy, s = pop_last_segment!(majorant)
            δx += dx
            δy += dy

            # Check if concave again
            auxi -= 1
            (auxi < 1 || δy <= float(δx) * last_slope(majorant)) && break
        end
    end
    push_to_last!(majorant, δx, δy)
end

@inline function new_knot!(
    minorant::SegmentBuffer,
    majorant::SegmentBuffer,
    λ::Float64,
    origin::PointXY,
    last_explored::PointXY
)
    # First segments
    Δxmin, Δymin, slopemin = first_segment(minorant)
    Δxmaj, Δymaj, slopemaj = first_segment(majorant)

    # Shortest segment defines the new knot
    if Δxmin < Δxmaj
        # Remove first segment, rest of the minorant is still valid
        remove_first_segment!(minorant)

        # Majorant is a single segment from new knot to last original point
        restart_from_state!(majorant)
        Δx = last_explored.x - origin.x - Δxmin
        Δy = last_explored.y - origin.y - Δymin - λ
        push_to_last!(majorant, Δx, Δy)

        return Δxmin, Δymin, slopemin
    else # Left-most majorant touching point is the new knot
        remove_first_segment!(majorant)

        # Minorant is a single segment from new knot to last original point
        restart_from_state!(minorant)
        Δx = last_explored.x - origin.x - Δxmaj
        Δy = last_explored.y - origin.y - Δymaj + λ
        push_to_last!(minorant, Δx, Δy)

        return Δxmaj, Δymaj, slopemaj
    end
end

function filter1D!( # Weighted version
    β::AbstractVector{Float64},
    y::AbstractVector{Float64},
    λ::AbstractVector{Float64},
    from::Int,
    to::Int
)
    # problem size
    n = to - from + 1
    # @assert length(y) == length(λ) + 1
    # @assert all([λᵢ >= 0 for λᵢ ∈ λ])

    # @fastmath @inbounds begin # acceleration

    # Initialise memory structures
    majorant = SegmentBuffer(n)
    minorant = SegmentBuffer(n)
    push_to_last!(majorant, 1, y[from] - λ[from])
    push_to_last!(minorant, 1, y[from] + λ[from])

    # Initial point of taut-string and last explored
    origin = PointXY(from - 1, 0.)
    last_explored = PointXY(from, y[from])

    # Iterate along the signal length
    for i = (from + 1):(from + n - 2)
        # Update majorant, minorant and last explored point
        # Take into account difference in lambdas
        Δλ = λ[i] - λ[i - 1]
        add_segment_to_majorant!(majorant, 1, y[i] - Δλ)
        add_segment_to_minorant!(minorant, 1, y[i] + Δλ)
        last_explored.x += 1
        last_explored.y += y[i]

        # Check for slope crossings at the first point
        while first_slope(minorant) < first_slope(majorant)
            # Crossing detected
            Δx, Δy, slope = new_knot!(minorant, majorant, λ[i], origin, last_explored)

            # Write to solution
            @simd for j in 1:Δx
                β[origin.x + j] = slope
            end

            # Update origin
            origin.x += Δx
            origin.y += Δy
        end
    end

    # Update majorant and minorant with last segments
    add_segment_to_majorant!(majorant, 1, y[from + n - 1] + λ[from + n - 2]) # Δλ = 0. - λ[n - 1]
    add_segment_to_minorant!(minorant, 1, y[from + n - 1] - λ[from + n - 2])

    # At this point, because the endpoint of the tube is the same
    # for both majorant and minorant, either the majorant or the minorant
    # is a single straight segment while the other can be larger.
    # The remaining of the taut-string must be the multi-segment component.
    largest_buffer = (num_segments(majorant) > num_segments(minorant) ? majorant : minorant)
    for i in 1:num_segments(largest_buffer)
        # Write to solution
        Δx, Δy, slope = pop_first_segment!(largest_buffer)
        @simd for j in 1:Δx
            β[origin.x + j] = slope
        end
        origin.x += Δx
    end

    # end # end fastmath and inbounds
end

# Unweighted version
function filter1D!(
    β::AbstractVector{Float64},
    y::AbstractVector{Float64},
    λ::Float64,
    from::Int,
    to::Int
)
    filter1D!(β, y, fill(λ, length(y)), from, to)
end

function filter1D!( #
    β::AbstractVector{Float64},
    y::AbstractVector{Float64},
    λ::Union{AbstractVector{Float64}, Float64}
)
    filter1D!(β, y, λ, 1, length(y))
end

# non-inplace version
function filter1D(
    y::AbstractVector{Float64},
    λ::Union{AbstractVector{Float64}, Float64}
)
    # trivial cases
    λ == 0. && (return y)
    β = zeros(length(y))
    filter1D!(β, y, λ)
    β
end
