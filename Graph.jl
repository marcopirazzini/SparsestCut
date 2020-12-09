module MyGraph

#export Graph, erdosrenyi, cube, bfs, connected, conductance, update_conductance, sweep

using LinearAlgebra

struct Graph
    al::Dict{Int64, Array{Int64,1}}
    n::Int64
    am::BitArray{2}
    degs::Vector{Int64}
    l::Matrix{Float64}
    function Graph(al::Dict{Int64, Array{Int64,1}})
            n = length(al)
            am = BitArray(zeros(n,n)) 
            degs = zeros(Int64, n)
            for (v, neighb) in al
                am[v, neighb] .= true
                degs[v] = length(neighb)
            end
            d = Diagonal(degs)
            l = Symmetric(I - d^(-1/2) * am * d^(-1/2))

            @assert sum(Diagonal(am)) == 0
            @assert am == am'
        return new(al, n, am, degs, l)
    end
end

function bfs(g::Graph)    
    @assert any(g.degs .> 0)
    vertices = collect(1:g.n)
    reached = BitArray(zeros(g.n))
    visited = BitArray(zeros(g.n))
    to_visit = BitArray(zeros(g.n))

    v0 = vertices[g.degs .> 0][1]
    visited[v0] = true
    reached[v0] = true
    to_visit[v0] = false
    for i in g.al[v0]
        reached[i] = true
        !visited[i] && (to_visit[i] = true)
    end

    while any(to_visit)
        v = vertices[to_visit][1]
        visited[v] = true
        to_visit[v] = false
        for i in g.al[v]
            reached[i] = true
            !visited[i] && (to_visit[i] = true)
        end
        all(reached) && break
    end
    return reached
end

function connected(g::Graph)
    return all(bfs(g))
end

function erdosrenyi(n::Int64, p::Float64)
    p < 0.0 && p > 1.0 && throw(ArgumentError("Invalid probability p: it must be in [0,1]"))
    nodes = collect(1:n)
    al = Dict{Int64, Vector{Int64}}()
    ein = Symmetric(rand(n,n)) .< p
    ein[diagind(ein)] .= false
    for v in 1:n
        al[v] = nodes[ein[v,:]]
    end
    return Graph(al)
end

function cube()
    l = Dict(1=>[2,3,5], 2=>[1,6,4], 3=>[1,4,7], 4=>[2,3,8], 
             5=>[1,6,7], 6=>[2,5,8], 7=>[3,5,8], 8=>[4,6,7])
    return Graph(l)
end


function conductance(g::Graph, S::BitArray{1}) #edge expansion
    E = 0
    den = 0
    for u in keys(g.al)
        S[u] && (den += g.degs[u])
        for v in g.al[u]
            (S[u] != S[v]) && (E += 1)
        end
    end

    den = min(den, sum(g.degs)-den)
    @assert E%2 == 0
    num = E÷2
    return (num, den)
end

function sparsity(g::Graph, S::BitArray{1}) #cut sparsity
    E = 0
    den = sum(S) * (g.n - sum(S))
    for u in keys(g.al)
        for v in g.al[u]
            (S[u] != S[v]) && (E += 1)
        end
    end

    @assert E%2 == 0
    num = E÷2
    return (num, den)
end

function update_conductance(g::Graph, S::BitArray{1}, num::Int64, den::Int64, v::Int64, den_true::Int64)
    #@assert !S[v] 
    #@assert conductance(g,S) == (num,den_true) #uncomment for testing, keep comment for efficiency

    d = den + g.degs[v] #actual denominator is adjusted in sweep function 

    e1 = e2 = 0
    for u in g.al[v]
        S[u] ? (e1 += 1) : (e2 += 1)
    end
    return (num - e1 + e2, d)
end

function update_boundary(g::Graph, S::BitArray{1}, num::Int64, v::Int64) #same as numerator update for conductance, which is the only update necessary for sparsity
    e1 = e2 = 0
    for u in g.al[v]
        S[u] ? (e1 += 1) : (e2 += 1)
    end
    return num - e1 + e2
end

function sweep(g::Graph, f::Vector{Float64}; loss::String="sparsity", verbose::Bool=true)
    @assert g.n == length(f)
    order = sortperm(f)

    #initialization - the initial conductance is always 1
    S = BitArray(zeros(g.n))
    v = order[1]
    S[v] = true
    num = length(g.al[v]) #initial boundary
    if loss == "conductance"
        den = length(g.al[v]) #initial volume
        den_true = min(den, sum(g.degs)-den)
    elseif loss == "sparsity"
        den_true = 1
    else
        throw(ArgumentError("Invalid loss")) 
    end
    best, best_idx = 1.0, 1
    #update
    for i in 2:(g.n-1)
        #@info "Iteration = $(i-1)\n Current best cut = $(S)\n Current num = $(num)\n Current den = $(den)\n Cut $(loss) = $(best)"
        v = order[i]
        if loss == "conductance"
            num, den = update_conductance(g, S, num, den, v, den_true)
            den_true = min(den, sum(g.degs)-den)
        else
            num = update_boundary(g, S, num, v)
            den_true = i * (g.n - i)
        end
        ϕ = num/den_true
        #@info "Added node $(v)\n Updated num = $(num)\n Updated den = $(den_true)\n Updated $(loss) = $(ϕ)\n\n"
        ϕ < best && ((best, best_idx) = (ϕ, i))
        S[v] = true
    end

    cut = BitArray(zeros(g.n))
    cut[order[1:best_idx]] .= true

    # if loss == "conductance"
    #     nf,df = conductance(g, cut)
    # else
    #     nf,df = sparsity(g, cut)
    # end
    # @assert nf/df == best

    if verbose
        println("Final cut: $(order[1:best_idx])")
        println("Cut $(loss): $(best)")
    end
    return cut
end

end #module