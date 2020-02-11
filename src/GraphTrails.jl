import Base: length, push!, append!, isempty, collect, first, last, show
using DataStructures
using LightGraphs


mutable struct DoublyLinkedNode{T}
    data::Union{Nothing, T}
    next::DoublyLinkedNode{T}
    prev::DoublyLinkedNode{T}
    function DoublyLinkedNode{T}() where T
        x = new()
        x.data = nothing
        x.next = x
        x.prev = x
        x
    end
    function DoublyLinkedNode{T}(
            data::T, 
            next::DoublyLinkedNode{T},
            prev::DoublyLinkedNode{T}) where T
        new(data, next, prev)
    end
end


function show(io::IO, x::DoublyLinkedNode)
    print(io, "LinkedNode($(x.data))") 
end


mutable struct DoublyLinkedList{T}
    head::DoublyLinkedNode{T}
    size::Int
    function DoublyLinkedList{T}() where T
        new(DoublyLinkedNode{T}(), 0)
    end
end


function print_next(x::DoublyLinkedList, n::Int)
    v = x.head
    for i in 1:n         
        print(v.data)
        v = v.next
        print("-")
    end
    print("\n")
end


function print_prev(x::DoublyLinkedList, n::Int)
    v = x.head
    for i in 1:n         
        print(v.data)
        v = v.prev
        print("-")
    end
    print("\n")
end


function show(io::IO, x::DoublyLinkedList)
    print(io, "LinkedList{$(typeof(x.head))} of size $(x.size)") 
end


function push!(l::DoublyLinkedList{T}, x::T) where T
    curr_last = l.head.prev
    new_node = DoublyLinkedNode{T}() 
    new_node.data = x
    new_node.next = l.head
    new_node.prev = curr_last
    l.head.prev = new_node
    curr_last.next = new_node
    l.size += 1
    return
end


function shift!(l::DoublyLinkedList{T}, x::T) where T
    curr_first = l.head.next
    new_node = DoublyLinkedNode{T}() 
    new_node.data = x
    new_node.next = curr_first
    new_node.prev = l.head
    curr_first.prev = new_node
    l.head.next = new_node
    l.size += 1
    return
end


function delete_last!(l::DoublyLinkedList{T}) where T
    l.head.prev.prev.next = l.head
    l.head.prev = l.head.prev.prev
    l.size -= 1
    return
end


function collect(l::DoublyLinkedList{T}) where T
    out = T[]
    v = l.head.next
    for i in 1:l.size
        push!(out, v.data)
        v = v.next
    end
    out
end


function reverse!(l::DoublyLinkedList{T}) where T
    v = l.head
    for i in 1:l.size + 1
        w = v.next
        v.next, v.prev = v.prev, v.next
        v = w
    end
end


function reversed(l::DoublyLinkedList{T}) where T
    rev = DoublyLinkedList{T}()
    rev.size = l.size
    v = rev.head
    w = l.head.prev
    for i in 1:l.size
        v.next = DoublyLinkedNode{T}(w.data, w.prev, w.next)
        v.next.prev = v
        v = v.next
        w = w.prev
    end
    v.next = rev.head
    rev.head.prev = v
    rev
end


function DoublyLinkedList(x::AbstractVector{T}) where T
    l = DoublyLinkedList{T}()
    for xi in x
        push!(l, xi)
    end
    l
end


function append!(
        l::DoublyLinkedList{T},
        l2::DoublyLinkedList{T}) where T
    # let's deal with trivial cases
    if l2.size == 0
        return
    elseif l.size == 0
        l.head = l2.head
        l.size = l2.size
        return
    else
        l.head.prev.next = l2.head.next
        l2.head.next.prev = l.head.prev
        l.head.prev = l2.head.prev
        l2.head.prev.next = l.head
        l.size += l2.size
        return
    end
end


function cat(
        l::DoublyLinkedList{T},
        l2::DoublyLinkedList{T}) where T
    lnew = DoublyLinkedList{T}()
    lnew = deepcopy(l)
    append!(lnew, l2)
end


function first(l::DoublyLinkedList{T}) where T
    l.head.next.data
end


function last(l::DoublyLinkedList{T}) where T
    return l.head.prev.data
end


struct LinkedGraphNode
    id::Int
    nbrs::Vector{LinkedGraphNode}
    inv_nbrs::Vector{Int} # stores the position in the neighbor of the edge to quickly mark as visited
    visited::BitVector
    function LinkedGraphNode(id::Int)
        new(id, LinkedGraphNode[], Int[], falses(0))
    end
end


function show(io::IO, v::LinkedGraphNode)
    msg = "Node($(v.id), $(sum(v.visited))/$(length(v.nbrs)))"
    print(io, msg)
end


struct LinkedGraph
    root::LinkedGraphNode
    num_nodes::Int
    num_edges::Int
end


function show(io::IO, g::LinkedGraph)
    nv = g.num_nodes
    ne = g.num_edges
    msg = "LinkedGraph with $(nv) nodes and $(ne) edges"
    print(io, msg)
end


function find_unvisited(v::LinkedGraphNode)
    n = length(v.nbrs)
    i = 1
    pos_free = n + 1
    num_free = 0
    while i <= n
        if !v.visited[i]
            pos_free = min(pos_free, i)
            num_free += 1
            num_free >= 2 && break
        end
        i += 1
    end
    pos_free, num_free
end


function mark_as_visited!(v::LinkedGraphNode, i::Int)
    v.visited[i] = true
    v.nbrs[i].visited[v.inv_nbrs[i]] = true
end


function LinkedGraph(g::SimpleGraph, root::Int = 1)
    @assert length(connected_components(g)) == 1
    nodes = [LinkedGraphNode(i) for i in 1:nv(g)]
    for e in edges(g)
        i, j = src(e), dst(e)
        κ, η = length(nodes[i].nbrs), length(nodes[j].nbrs)
        push!(nodes[i].nbrs, nodes[j])
        push!(nodes[i].inv_nbrs, η + 1)
        push!(nodes[i].visited, false)
        push!(nodes[j].nbrs, nodes[i])
        push!(nodes[j].inv_nbrs, κ + 1)
        push!(nodes[j].visited, false)
    end
    LinkedGraph(nodes[root], nv(g), ne(g))
end

mutable struct Branch
    path::DoublyLinkedList{LinkedGraphNode}
    deletable::Int
end


function show(io::IO, b::Branch)
    msg = "Branch(size=$(b.path.size), del=$(b.deletable))"
    print(io, msg)
end


function is_cycle(
        v::LinkedGraphNode,
        τ::DoublyLinkedList{LinkedGraphNode})
    v == first(τ) 
end


function merge_branches_at_fork!(
    v::LinkedGraphNode,
    branches::Stack{Branch},
    forks::Stack{LinkedGraphNode},
)
    subpaths = DoublyLinkedList{LinkedGraphNode}[]
    deletables = Int[]
    while true
        branch = pop!(branches)
        pop!(forks)
        push!(subpaths, branch.path)
        push!(deletables, branch.deletable)
        # if no more branches or next branch belongs to a different fork
        (isempty(branches) || top(forks) != v) && break
    end

    merged = DoublyLinkedList{LinkedGraphNode}()

    largest = 0
    maxlen = 0
    for (i, τ) in enumerate(subpaths)
        len = τ.size - deletables[i]
        if len > maxlen
            largest = i
            maxlen = len
        end
    end
    
    τ = subpaths[largest]    
    deletable = deletables[largest] + (is_cycle(v, τ) ?  0 : maxlen)
    if !is_cycle(v, τ)
        push!(merged, v)
        append!(merged, reversed(τ)) 
        delete_last!(merged)
    end  
    append!(merged, τ)

    for (i, τ) in enumerate(subpaths)
        if i != largest
            if !is_cycle(v, τ)
                push!(merged, v)
                append!(merged, reversed(τ)) 
                delete_last!(merged)
            end
            append!(merged, τ)
        end
    end
    
    branch = Branch(merged, deletable)

    return branch
end


function quicktour(g::LinkedGraph)
    seglens = Stack{Int}()
    branches = Stack{Branch}()
    forks = Stack{LinkedGraphNode}()
    path = Stack{LinkedGraphNode}()
    
    curr_len = 1
    v = g.root
    push!(path, v)
    push!(seglens, 1)

    while !isempty(seglens)
        push!(path, v)
        i_free, num_free = find_unvisited(v)
        if num_free >= 2
            # case a: commit as fork an advance
            push!(seglens, curr_len)
            curr_len = 1
        elseif num_free == 1
            # case b: proceed to nbr
            curr_len += 1
        else
            # case 3: dead end found, add to branches and check if done with fork
            # flush to branch
            branch_path = DoublyLinkedList{LinkedGraphNode}()
            for i in 1:curr_len
                v = pop!(path)    
                push!(branch_path, v)
            end 
            push!(branches, Branch(branch_path, 0))
            v = top(path) # now v == curr_fork
            push!(forks, v)

            # if done with fork merge segments, otherwise continue on available
            i_free, num_free = find_unvisited(v)
            if num_free == 0
                # branch merging until finding available path to move
                pop!(path)
                while true 
                    #! TODO! DELETE ON THE LEFT AS WELL
                    branch = merge_branches_at_fork!(v, branches, forks)
                                        
                    # move to next fork
                    curr_len = pop!(seglens)
                    if isempty(seglens)
                        τ = collect(branch.path)
                        sol = [x.id for (i, x) in enumerate(τ) if i > branch.deletable]
                        return sol
                    end
                    for i = 1:curr_len
                        push!(branch.path, v)                        
                        v = pop!(path)
                        shift!(branch.path, v)
                        branch.deletable += 1
                    end 
                    push!(forks, v)
                    push!(branches, branch)

                    i_free, num_free = find_unvisited(v)
                    num_free > 0 && break
                end
                push!(path, v)
            end
            
            curr_len = 1
        end 

        # advance
        mark_as_visited!(v, i_free)
        v = v.nbrs[i_free]
    end
end


function quicktour(g0::SimpleGraph, k::Int)
    g = LinkedGraph(g0, k)
    quicktour(g)
end


function count_unique_edges(tour::Vector{Int})
    return length(unique([sorted(tour[i], tour[i + 1]) for i in 1:length(tour) - 1]))
end


function quicktour(g0::SimpleGraph)
    v = findfirst(k -> isodd(degree(g0, k)), vertices(g0))
    (v == 0) && (v = 1)
    g = LinkedGraph(g0, v)
    quicktour(g)
end


# """
# Implements pseudotour trails decomposition algorithm from Tansey
# """
function pseudotour_trails(g::AbstractGraph; as_trails::Bool=false)
    odds = [i for i in vertices(g) if isodd(degree(g, i))]
    n_odds = length(odds)
    @assert iseven(n_odds) "pseudotour not possible: number of odd vertices must be even"

    trails = Vector{Vector{Int}}(0)
    if n_odds == 0
        # single tour is possible
        append!(trails, [euler(g, 1)])
    else
        # must add a new vertex to all odd vertices
        n = nv(g)
        g_ = copy(g)
        add_vertex!(g_)
        for v in odds
            add_edge!(g_, n + 1, v)
        end

        # now find a large tour
        tour = euler(g_, 1)
        breaks = findall(tour .== n + 1)

        # split the tour based on the artifial vertex
        if breaks[1] != n + 1
            prepend!(tour, n + 1)
            breaks += 1
            prepend!(breaks, 1)
        end
        if breaks[end] != n + 1
            append!(tour, n + 1)
            append!(breaks, length(tour))
        end
        for i in 1:length(breaks) - 1
            append!(trails, [tour[breaks[i] + 1:breaks[i + 1] - 1]])
        end
    end
    trails = [t for t in trails if length(t) > 1]

    return as_trails ? trailvec_to_trails(trails, nv(g)) : trails
end


function trailvec_to_trails(intvec::AbstractVector{Vector{Int}}, n::Int)
    ptr = vcat(intvec...)
    wts = vcat([[ones(length(τ) - 1); 0.] for τ in intvec]...)
    brks = [1; 1 + cumsum(length.(intvec))]
    return Trails(ptr, brks, wts, n)
end


function euler(g::AbstractGraph, to::Int)
    # check input ---
    @assert has_vertex(g, to) "start is not in g"

    # neighbour map
    g_ = copy(g)
    edge_count = degree(g_)

    # eulerian?
    @assert length(connected_components(g)) == 1 "the graph is not connected"
    n_odds = sum(isodd.(edge_count))
    @assert n_odds ∈ (0, 2) "number of odd vertices must be zero or two"
    @assert n_odds == 0 || isodd(edge_count[to]) "the graph has two odd vertices but start is not one of them"
    # checks ready ---

    # stack to keep vertices
    curr_path = Stack{Int}()

    # stores final circuit
    circuit = Int[]

    # start from start
    curr_v = to
    push!(curr_path, to)

    while !isempty(curr_path)
        # There's a remaining edge
        remaining = edge_count[curr_v]
        if remaining > 0
            # add vertex to current path
            push!(curr_path, curr_v)

            # find the next vertex
            next_v = neighbors(g_, curr_v)[1]

            # reduce number of available edges
            edge_count[curr_v] -= 1
            edge_count[next_v] -= 1
            rem_edge!(g_, curr_v, next_v)

            # update current vertex
            curr_v = next_v
        else
            # add vertex without edges into circuit
            push!(circuit, curr_v)

            # backtrack until a vertex has edges left
            curr_v = pop!(curr_path)
        end
    end

    return circuit
end


struct Trails
    ptr::Vector{Int}
    brks::Vector{Int}
    wts::Vector{Float64}
    num_nodes::Int

    # Validate input
    function Trails(
            ptr::Vector{Int},
            brks::Vector{Int},
            wts::Vector{Float64},
            num_nodes::Int)
        @assert issorted(brks)
        @assert brks[1] == 1
        @assert brks[end] == length(ptr) + 1
        @assert length(wts) == length(ptr)
        new(ptr, brks, wts, num_nodes)
    end

    # Obtain weights and num_nodes from ptr
    function Trails(
            ptr::Vector{Int},
            brks::Vector{Int})
        @assert issorted(brks)
        @assert brks[1] == 1
        @assert brks[end] == length(ptr) + 1
        num_nodes = length(unique(ptr))
        cnts = counter(Tuple{Int, Int})
        for (a, b) in zip(ptr[1:end-1], ptr[2:end])
            x, y = min(a, b), max(a, b)
            cnts[x, y] += 1
        end
        wts = zeros(length(ptr))
        for (i, (a, b)) in enumerate(zip(ptr[1:end-1], ptr[2:end]))
            if i + 1 ∉ brks
                x, y = min(a, b), max(a, b)
                wts[i] += 1.0 / cnts[x, y]
            end
        end
        new(ptr, brks, wts, num_nodes)
    end
end

function show(io::IO, trails::Trails)
    print("Trails object with ", num_trails(trails), " trail(s)")
end

function trail(trails::Trails, i::Int)
    trails.ptr[trails.brks[i]:(trails.brks[i + 1] - 1)]
end

length(trails::Trails) = length(trails.ptr)
num_trails(trails::Trails) = length(trails.brks) - 1
num_nodes(trails::Trails) = trails.num_nodes

sorted(e::Edge) = sorted(src(e), dst(e))


function graph_from_edgelist(
        edges::AbstractArray;
        from_zero::Bool=false)
    @assert size(edges, 2) == 2
    n = maximum(vec(edges))
    g = SimpleGraph(n + Int(from_zero))
    for (i, j) in zip(edges[:, 1], edges[:, 2])
        add_edge!(g, i + Int(from_zero), j + Int(from_zero))
    end
    return g
end


function find_trails(g::SimpleGraph; ntrails::Int = 1)
    @time ptr_chinese = quicktour(g)
    N = length(ptr_chinese) + ntrails - 1
    chunk_size = if N % ntrails == 0
        N / ntrails
    else 
        (N + ntrails) ÷ (ntrails) 
    end
    ptr = Int[]
    brks = [1]
    for i in 1:ntrails - 1
        chunk_start = (i - 1) * chunk_size + 1
        chunk_end = i * chunk_size
        append!(ptr, ptr_chinese[chunk_start:chunk_end])
        push!(brks, length(ptr) + 1)
        push!(ptr, ptr[end])
    end
    chunk_start = (ntrails - 1) * chunk_size + 1
    append!(ptr, ptr_chinese[chunk_start:end])
    push!(brks, length(ptr) + 1);
    Trails(ptr, brks)
end


# g0 = Graph(7)
# add_edge!(g0, 1, 2)
# add_edge!(g0, 1, 3)
# add_edge!(g0, 2, 4)
# add_edge!(g0, 2, 5)
# add_edge!(g0, 3, 6)
# add_edge!(g0, 3, 7)
# gplot(g0, nodelabel=1:7)

# g = Graph(18)
# add_edge!(g, 1, 2)
# add_edge!(g, 2, 3)
# add_edge!(g, 2, 4)
# add_edge!(g, 2, 6)
# add_edge!(g, 4, 5)
# add_edge!(g, 5, 7)
# add_edge!(g, 6, 7)
# add_edge!(g, 6, 8)
# add_edge!(g, 8, 9)
# add_edge!(g, 9, 10)
# add_edge!(g, 9, 11)
# add_edge!(g, 10, 13)
# add_edge!(g, 11, 12)
# add_edge!(g, 13, 14)
# add_edge!(g, 13, 15)
# add_edge!(g, 15, 16)
# add_edge!(g, 14, 10)
# add_edge!(g, 13, 17)
# add_edge!(g, 11, 18)
# gplot(g, nodelabel=1:nv(g))
# g2 = deepcopy(g)
# tour = quicktour(g)

# srand(321654)
# for i in 1:1000000
#     seed = rand(1:1000000)
#     srand(seed)
#     g = erdos_renyi(10, .25)
#     if length(connected_components(g)) == 1 
#         tour = quicktour(g)
#         evisited = count_unique_edges(tour)
#         if ne(g) != evisited
#             println("seed = $seed")
#             for l in 1:length(tour) - 1
#                 v, w = tour[l], tour[l + 1]
#                 if !has_edge(g, v, w)
#                     println("$v-$w")
#                 end
#             end
#             gplot(g, nodelabel=vertices(g))
#             break
#         end
#     end
# end