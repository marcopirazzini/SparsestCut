include("Graph.jl")
include("Fiedler.jl")
include("SDP.jl")
include("ARV.jl")

g = MyGraph.erdosrenyi(12,0.4) #be careful in using more than 15 nodes, the algorithm slows down significantly for ARV
if !MyGraph.connected(g)
    println("Disconnected graph!")
    cut = MyGraph.bfs(g)
    println("Final cut = $(collect(1:12)[cut]) \nFinal conductance = $(0.0)")
else
    @info "Fiedler"
    c_f = Fiedler.fiedler(g, verbose=true)
    cond_f = MyGraph.conductance(g, c_f)
    spars_f = MyGraph.sparsity(g, c_f)
    @info "ARV"
    X = SDP.sdp(g)
    emb = SDP.embedding(X)
    c_arv = ARV.arv(g, emb, verbose=true)
    cond_arv = MyGraph.conductance(g, c_arv)
    spars_arv = MyGraph.sparsity(g, c_arv)
    @info "results"
    println("Fiedler conductance = $(cond_f[1]/cond_f[2]) \nFiedler sparsity = $(spars_f[1]/spars_f[2]) \n\nARV conductance = $(cond_arv[1]/cond_arv[2]) \nARV sparsity = $(spars_arv[1]/spars_arv[2])")
end