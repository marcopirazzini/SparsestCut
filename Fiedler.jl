module Fiedler

using LinearAlgebra
using ..MyGraph

function fiedler(g::MyGraph.Graph; verbose::Bool=true)
    evals, evecs = eigen(g.l) #eigendecomposition 
    x = evecs[:,2] #Fiedler vector
    cut = MyGraph.sweep(g, x, loss="conductance", verbose=verbose)
    return cut
end

end #module