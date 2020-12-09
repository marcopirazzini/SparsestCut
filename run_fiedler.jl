include("Graph.jl")
include("Fiedler.jl")

g = MyGraph.erdosrenyi(50, 0.2)
cut = Fiedler.fiedler(g)