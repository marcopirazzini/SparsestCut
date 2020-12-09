include("Graph.jl")
include("SDP.jl")
include("ARV.jl")

g = MyGraph.erdosrenyi(12, 0.4) #be careful in using more than 15 nodes, the algorithm slows down significantly
X = SDP.sdp(g)
emb = SDP.embedding(X)
c_arv = ARV.arv(g, emb, verbose=true)