module SDP

using LinearAlgebra, JuMP, ProxSDP
using ..MyGraph

function build_C(g::MyGraph.Graph)
    c = -1 .* copy(g.am)
    c[diagind(c)] = g.degs
    return c
end

function build_A(n::Int)
    a = -1 .* ones(Int, n, n)
    a[diagind(a)] .= n - 1
    return a
end

function build_B(n::Int, w::Int, u::Int, v::Int)
    @assert (v ≠ u) && (v ≠ w) && (u ≠ w)
    b = zeros(n, n)

    for i in 1:n, j in i:n
        if i == j == w
            b[i,i] = 1
        elseif (i == u && j == v) || (i == v && j == u)    #{i,j}∩{u,v,w} == {u,v}
            b[i,j] = b[j,i] = 0.5
        elseif ((i == u && j == w) || (i == w && j == u)) || ((i == v && j == w) || (i == w && j == v))      #{i,j}∩{u,v,w} == {u,w} || {i,j}∩{u,v,w} == {v,w}
            b[i,j] = b[j,i] = -0.5
        end
    end
    return b
end

function build_B(n::Int, w::Int, u::Int)
    @assert (u ≠ w) 
    b = zeros(n, n)

    for i in 1:n, j in i:n
        if i == j == w
            b[i,i] = 1
        elseif i == j == u
            b[i,i] = 1
        elseif (i == u && j == w) || (i == w && j == u)  #in this case i≠j automatically
            b[i,j] = b[j,i] = -0.5
        end
    end
    return b
end

function sdp(g::MyGraph.Graph)
    n = g.n
    C = build_C(g) #only the objective function depends on the structure of the graph, the constraints are structural properties of the embedded space of vertices
    A = build_A(n)

    model = Model(optimizer_with_attributes(ProxSDP.Optimizer, "log_verbose"=>false, "tol_gap"=>1e-4, "tol_feasibility"=>1e-4))
    @variable(model, X[1:n, 1:n], PSD)
    @objective(model, Min, dot(C, X))
    @constraint(model, dot(A, X) == 1)
    for w in 1:n
        for u in 1:n, v in u:n
            if (u != w) && (v != w) 
                if u != v
                    B = build_B(n, w, u, v)
                    @constraint(model, dot(B, X) >= 0)
                else
                    B = build_B(n, w, u)
                    @constraint(model, dot(B, X) >= 0)
                end
            end
        end
    end
    #model solving
    optimize!(model)
    X_out = Symmetric(value.(model[:X]))
    return X_out
end

function embedding(X::Symmetric{Float64, Matrix{Float64}})
    evals, evecs = eigen(X, sortby=nothing)
    @. evals[abs(evals) < 1e-10] = 0.0 #avoid negative eigenvalues due to numerical imprecision
    emb = evecs * Diagonal(.√evals) * evecs' #psd matrix square root
    return emb
end

function sdp_run(g::MyGraph.Graph)
    X = sdp(g)
    emb = embedding(X)
    return emb
end

end #module