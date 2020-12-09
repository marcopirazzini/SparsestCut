include("Graph.jl")

include("Fiedler.jl")
include("SDP.jl")
include("ARV.jl")

using LinearAlgebra

###
# Graph
###

function create_g() #cube in 3 dimensions
    l = Dict(1=>[2,3,5], 2=>[1,6,4], 3=>[1,4,7], 4=>[2,3,8], 
             5=>[1,6,7], 6=>[2,5,8], 7=>[3,5,8], 8=>[4,6,7])
    g = MyGraph.Graph(l)
    #println("Graph: $(g)")
    return g
end

function create_g(n::Int, p::Float64) 
    g = MyGraph.erdosrenyi(n,p)
    return g
end

function test_cond(p = 0.4)
    g = create_g()
    S = rand(g.n) .< p
    println("S = $(S)")
    ϕ = MyGraph.conductance(g,S)
    return ϕ
end

###
# Fiedler
###

function test_fiedler()
    g = create_g()
    if !MyGraph.connected(g)
        println("Disconnected graph!")
        return MyGraph.bfs(g)
    end
    return Fiedler.fiedler(g)
end

function test_fiedler(g::MyGraph.Graph)
    return Fiedler.fiedler(g)
end

function test_fiedler(n::Int, p::Real)
    g = MyGraph.erdosrenyi(n,p)
    return Fiedler.fiedler(g)
end

###
# SDP
###

function test_c()
    g = create_g()
    println("Graph: $(g.am)")
    c = SDP.build_C(g)
    println("Cost matrix: $c")
    return c
end

function test_sdp()
    g = create_g()
    X = SDP.sdp(g)
    return X
end

function test_sdp(n::Int=10, p::Float64=0.3)
    g = create_g(n, p)
    X = SDP.sdp(g)
    return X
end

function check_emb(X::Symmetric{Float64, Matrix{Float64}}, emb::Matrix{Float64})
    n = size(X)[1]
    for i in 1:n, j in 1:n
        @assert abs(X[i,j] - dot(emb[:,i], emb[:,j])) < 1e-10
    end
end

function test_emb()
    X = test_sdp()
    emb = SDP.embedding(X)
    check_emb(X, emb)
    return emb
end

###
# ARV
###

function emb_to_dist()
    emb = test_emb()
    dist = ARV.compute_dist(emb)
    return dist
end

# function test_frechet()
#     dist = emb_to_dist()
#     S = [2,5]
#     f1 = ARV.frechet_embedding(dist, S)
#     f2 = ARV.frechet_embedding1(dist, S)
#     @assert sum(abs.(f1-f2)) < 1e-10
#     return f1, dist
# end

function test_arv()
    g = MyGraph.cube()
    if !MyGraph.connected(g)
        println("Disconnected graph!")
        return MyGraph.bfs(g)
    end
    X = SDP.sdp(g)
    emb = SDP.embedding(X)
    cut = ARV.arv(g, emb)
    return cut
end

function compare_alg(;verbose=true)
    g = MyGraph.erdosrenyi(10,0.3)
    if !MyGraph.connected(g)
       println("Disconnected graph!")
       return MyGraph.bfs(g)
    end
    @info "Fiedler"
    c_f = Fiedler.fiedler(g, verbose=verbose)
    cond_f = MyGraph.conductance(g, c_f)
    spars_f = MyGraph.sparsity(g, c_f)
    @info "ARV"
    X = SDP.sdp(g)
    emb = SDP.embedding(X)
    c_arv = ARV.arv(g, emb, verbose=verbose)
    cond_arv = MyGraph.conductance(g, c_arv)
    spars_arv = MyGraph.sparsity(g, c_arv)
    @info "results"
    println("Fiedler conductance = $(cond_f[1]/cond_f[2]) \nFiedler sparsity = $(spars_f[1]/spars_f[2]) \n\nARV conductance = $(cond_arv[1]/cond_arv[2]) \nARV sparsity = $(spars_arv[1]/spars_arv[2])")
    return (g, c_f, c_arv)
end

