using Distributed


mutable struct Node
    split::Float64
    logit::Union{Nothing, Float64}
    left::Union{Node, Nothing}
    right::Union{Node, Nothing}
    Node(split::Float64, logit::Union{Float64, Nothing} = nothing) = begin
        new(split, logit, nothing, nothing)
    end
end

##
isleaf(v::Node) = isnothing(v.left) && isnothing(v.right)
descend(v::Node, x::Float64) = (x ≤ v.split) ? v.left : v.right
tostr(x::Union{Node, Float64, Nothing}; add_children=false) = begin
    if isnothing(x) 
        return "(*)"
    elseif isa(x, Float64)
        return @sprintf("%.2f", x)
    else
        s = "Node($(tostr(x.split)), $(tostr(x.logit)))\n"
        if add_children
            s *= "|--$(tostr(x.left))\n"
            s *= "|--$(tostr(x.right))"
        end
        return s
    end
end
Base.show(io::IO, v::Node) = show(io, "Node(id=$(objectid(v)))")
Base.print(io::IO, v::Node) = print(io, tostr(v, add_children=true))

function getval(v::Node, x::Float64) 
    @assert !isnothing(v.logit) "can only evaluate nodes with values, use bfs_fill_tree"
    (x ≤ v.split) ? v.logit : -v. logit
end


function eval_logprob(root::Node, x::Float64)
    @assert !isleaf(root) "must start from non-leaf"
    logprob = 0.0
    v = root
    while !isnothing(v)
        β = getval(v, x)
        if β ≥ 0
            logprob += -log(1.0 + exp(-β))
        else
            logprob += β - log(1.0 + exp(β))
        end
        v = descend(v, x)
    end
    return logprob
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
    lows::Vector{Float64},
    mid::Vector{Float64},
    ups::Vector{Float64},
    logits::Vector{T} = fill(nothing, length(mid))
) where T <: Union{Nothing, Float64}
    num_nodes = length(mid)
    next_parent = Queue{Node}()
    root = Node(mid[1], logits[1])
    curr = root  # current parent
    for i in 2:num_nodes
        v = Node(mid[i], logits[i])
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
    root
end
