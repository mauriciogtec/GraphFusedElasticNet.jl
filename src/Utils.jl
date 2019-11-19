using DataStructures
import LightGraphs: SimpleGraph, add_edge!, neighbors

sorted(i::Int, j::Int) = (min(i, j), max(i, j))

function invcount(ptr::Vector{Int}, brks::Vector{Int})
    m = length(ptr) - 1
    cnt = counter([sorted(ptr[i], ptr[i + 1]) for i in 1:m  if i + 1 ∉ brks])
    edgecnts = [float(cnt[sorted(ptr[i], ptr[i + 1])]) for i in 1:m]
    edgewt = [x <= 1. ? x : 1 ./ x for x in edgecnts]
    push!(edgewt, 1.)
    edgewt
end

function plateaus(
        β::Vector{Float64},
        g::SimpleGraph,
        tol::Float64)
    n = length(β)
    # plateaus = Vector{Vector{Int}}

    # checked if we have already walked through node, to_check has missing nodes
    nplateaus = 0.
    checked = falses(n)
    to_check = collect(1:n)

    while !isempty(to_check)
        # find next unchecked index
        idx = pop!(to_check)
        while !isempty(to_check) && checked[idx]
            idx = pop!(to_check)
        end

        # edge case, exit loop if there's no unchecked element
        checked[idx] && break

        # current plateau and member conditions
        cur_unchecked = [idx]

        # check every possible boundary of the plateau
        while !isempty(cur_unchecked)
            idx = pop!(cur_unchecked)

            # check the index of unchecked neighbours
            for local_idx in neighbors(g, idx)
                if !checked[local_idx] && abs(β[idx] - β[local_idx]) < tol
                    # check neighbour
                    checked[local_idx] = true

                    # add it to current plateau
                    push!(cur_unchecked, local_idx)
                end
            end
        end
        nplateaus += 1.
    end

    nplateaus
end




# ----------------
function bincount(
    y::Vector{Float64},
    node::Vector{Int},
    splits::Vector{Float64};
    priorcount::Float64 = 0.,
    numnodes::Int = maximum(node)
)
    n, m = length(y), length(splits) - 1
    countmat = fill(priorcount, numnodes, m)
    for j in 1:m
        @simd for i in 1:n
            if splits[j] <= y[i] <  splits[j + 1]
                countmat[node[i], j] += 1.
            end
        end
    end
    countmat
end

function generate_bins2(levels::Int)
    bins = Tuple{Int, Int}[]
    for l = 1:levels
        currlevel_size = 2^(l - 1)
        currlevel_width = 2^(levels - l + 1)
        for k = 1: currlevel_size
            left = currlevel_width * (k - 1) + 1
            right = currlevel_width * k + 1
            push!(bins, (left, right))
        end
    end
    bins
end

function generate_bins(levels::Int)
    bins = Tuple{Int, Int}[]
    for l = 1:levels
        currlevel_size = 2^(l - 1)
        currlevel_width = 2^(levels - l + 1)
        for k = 1: currlevel_size
            left = currlevel_width * (k - 1) + 1
            right = currlevel_width * k
            push!(bins, (left, right))
        end
    end
    bins
end

function pred(k::Int)
    @assert k >= 1 "k must be positive"
    parent, direction = k ÷ 2, k % 2 == 1
end
function allpred(k::Int)
    if k == 1
        Tuple{Int,Int}[]
    else
        p, d = pred(k)
        [allpred(p); (p, d)]
    end
end
function level_numnodes(l::Int)
    [2^(l - 1) + k for k in 0:2^(l - 1) - 1]
end
# -------------------
