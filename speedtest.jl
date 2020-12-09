include("Graph.jl")
using .MyGraph
include("Fiedler.jl")
include("SDP.jl")
include("ARV.jl")

using Profile, ProfileView

function run_arv_all(g::MyGraph.Graph)
    X = SDP.sdp(g)
    emb = SDP.embedding(X)
    c_arv = ARV.arv(g, emb, verbose=false)
end


println("Fiedler(0) or ARV(1) ?")
s = readline()

if s == "0"
    #Fiedler speedtest
    @info "warming up (compiling...)"
    g1 = cube()
    g2 = erdosrenyi(100, 0.2)
    cut = Fiedler.fiedler(g1)
    @info "compiling completed"

    @info "running on the cube"
    g1 = cube()
    @time cut = Fiedler.fiedler(g1, verbose=false)
    @time cut = Fiedler.fiedler(g1, verbose=false)
    @time cut = Fiedler.fiedler(g1, verbose=false)
    @time cut = Fiedler.fiedler(g1, verbose=false)
    @info "running on a random graph with 100 nodes"
    g2 = erdosrenyi(100, 0.2)
    @time cut = Fiedler.fiedler(g2, verbose=false)
    @time cut = Fiedler.fiedler(g2, verbose=false)
    @time cut = Fiedler.fiedler(g2, verbose=false)
    @time cut = Fiedler.fiedler(g2, verbose=false)

    @profview Fiedler.fiedler(g2, verbose=false)
else
    @info "warming up (compiling...)"
    g1 = cube()
    g2 = erdosrenyi(10, 0.2)
    emb = run_arv_all(g1)
    @info "compiling completed"
    
    @info "running on the cube"
    g1 = cube()
    @time cut = run_arv_all(g1)
    @time cut = run_arv_all(g1)
    @time cut = run_arv_all(g1)
    @time cut = run_arv_all(g1)
    @info "running on a random graph with 15 nodes" #30 nodes take 15 minutes to solve SDP...
    g2 = erdosrenyi(15, 0.2)
    @time cut = run_arv_all(g2)
    @time cut = run_arv_all(g2)
    @time cut = run_arv_all(g2)
    @time cut = run_arv_all(g2)

    @profview run_arv_all(g2)
end