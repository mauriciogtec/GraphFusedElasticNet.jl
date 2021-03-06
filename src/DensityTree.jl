using Distributed
using Printf

mutable struct Node{T<:AbstractFloat}
    split::T
    low::T
    up::T
    logit::Union{Nothing, T}
    left::Union{Node, Nothing}
    right::Union{Node, Nothing}
    function Node(
        split::T, low::T, up::T, logit::Union{T, Nothing} = nothing
    ) where T<:AbstractFloat
        new{T}(split, low, up, logit, nothing, nothing)
    end
end

##
isleaf(v::Node) = isnothing(v.left) && isnothing(v.right)
descend(v::Node{T}, x::T) where T<:AbstractFloat = (x ≤ v.split) ? (v.left, 0) : (v.right, 1)

# tostr(x::Union{Node, Float64, Nothing}; add_children=false) = begin
#     if isnothing(x) 
#         return "(*)"
#     elseif isa(x, Float64)
#         return @sprintf("%.2f", x)
#     else
#         s = "Node($(tostr(x.split)), $(tostr(x.logit)))\n"
#         if add_children
#             s *= "|--$(tostr(x.left))\n"
#             s *= "|--$(tostr(x.right))"
#         end
#         return s
#     end
# end
# Base.show(io::IO, v::Node) = show(io, "Node(id=$(objectid(v)))")
# Base.print(io::IO, v::Node) = print(io, tostr(v, add_children=true))

function getval(v::Node{T}, x::T) where {T<:AbstractFloat}
    @assert !isnothing(v.logit) "can only evaluate nodes with values, use bfs_fill_tree"
    (x ≤ v.split) ? v.logit : -v. logit
end


function uniform_binary_splits(xmin::T, xmax::T, depth::Int) where T <: AbstractFloat
    lows = T[]
    mids = T[]
    ups = T[]
    levs = Int[] 
    splitvals = collect(LinRange(xmin, xmax, 2^depth + 1))
    for d in 1:depth
        width = 2 ^ (depth - d)
        num_intervals = 2 ^ (d - 1)
        for j in 1:num_intervals
            start = 2 * (j - 1) * width + 1
            push!(lows, splitvals[start])
            push!(mids, splitvals[start + width])
            push!(ups, splitvals[start + 2width])
            push!(levs, d - 1)
        end
    end
    (lows=lows, mids=mids, ups=ups, levs=levs, splitvals=splitvals)
end


function eval_logprob(root::Node{T}, x::T) where T<:AbstractFloat
    @assert !isleaf(root) "must start from non-leaf"
    logprob = zero(T)
    v = root
    while !isnothing(v)
        β = getval(v, x)
        if β ≥ zero(T)
            logprob += -log(one(T) + exp(-β))
        else
            logprob += β - log(one(T)+ exp(β))
        end
        v, _ = descend(v, x)
    end
    return logprob
end


function eval_logdens(root::Node{T}, x::T) where T<:AbstractFloat
    @assert !isleaf(root) "must start from non-leaf"
    logprob = zero(T)
    v = root
    prev = v
    descdir = 0
    while !isnothing(v)
        β = getval(v, x)
        if β ≥ zero(T)
            logprob += -log(one(T) + exp(-β))
        else
            logprob += β - log(one(T)+ exp(β))
        end
        prev = v
        v, descdir = descend(v, x)
    end
    δ = (descdir == 0) ? prev.split - prev.low : prev.up - prev.split
    return logprob - log(δ)
end


##
function print_bfs(v::Node)
    visited = Queue{Node}()
    enqueue!(visited, v)
    while !isempty(visited)
        curr = dequeue!(visited)
        println(curr)
        !isnothing(curr.left) && enqueue!(visited, curr.left)
        !isnothing(curr.right) && enqueue!(visited, curr.right)      
    end
end

##

function make_tree_from_bfs(
    lows::Vector{T},
    mid::Vector{T},
    ups::Vector{T},
    logits::Vector{S} = fill(nothing, length(mid))
) where {T<:AbstractFloat, S<:Union{Nothing, AbstractFloat}}
    num_nodes = length(mid)
    next_parent = Queue{Node}()
    root = Node(mid[1], lows[1], ups[1], logits[1])
    curr = root  # current parent
    for i in 2:num_nodes
        v = Node(mid[i], lows[i], ups[i], logits[i])
        while !(ups[i] ≈ curr.split || lows[i] ≈ curr.split)
            @assert !isempty(next_parent)  "disconnected tree!"
            curr = dequeue!(next_parent)
        end
        if ups[i] ≈ curr.split
            curr.left = v
        elseif lows[i] ≈ curr.split
            curr.right = v
        end
        enqueue!(next_parent, v)
    end
    return root
end

function make_tree(
    lows::Vector{T},
    mid::Vector{T},
    ups::Vector{T},
    logits::Vector{S} = fill(nothing, length(mid))
) where {T<:AbstractFloat, S<:Union{Nothing, AbstractFloat}}
    num_nodes = length(mid)
    nodes = [Node(mid[i], lows[i], ups[i], logits[i]) for i in 1:num_nodes]
    is_root = Set{Int}(1:num_nodes)
    for i in 1:num_nodes
        # serch for parent upwards
        child = nodes[i]
        child_lower, child_upper = lows[i], ups[i]
        for j in 1:num_nodes
            parent = nodes[j]
            parent_mid = mid[j]
            parent_lower, parent_upper = lows[j], ups[j]
            if child_lower ≈ parent_mid && child_upper ≈ parent_upper
                parent.right = child
                pop!(is_root, i)
                break
            elseif child_upper ≈ parent_mid && child_lower ≈ parent_lower
                parent.left = child
                pop!(is_root, i)
                break
            end
        end
    end
    if isempty(is_root)
        error("circular graph")
    elseif length(is_root) > 1
        error("multiple roots")
    end
    nodes[pop!(is_root)]
end