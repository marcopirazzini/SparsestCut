module ARV

using LinearAlgebra, Distributions
using ..MyGraph

function compute_dist(emb::Matrix{Float64}) #ℓ_2^2 metric
    n = size(emb)[1]
    dist = zeros(n,n)
    for i in 1:n, j in (i+1):n
        dist[i,j] = sum((emb[:,i] - emb[:,j]).^2)   #no sqrt
        dist[j,i] = dist[i,j]
    end
    return dist
end

function get_clusters(dist::Matrix{Float64}, r::Float64)
    n = size(dist)[1]
    sizes = [Vector{Int}([]) for i in 1:n]
    for i in 1:n, j in (i+1):n
        # compute ball size
        if dist[i,j] < r
            push!(sizes[i], j)
            push!(sizes[j], i)
        end
    end
    return sizes
end

function best_cluster(clusters::Array{Vector{Int},1})
    n = length(clusters)
    best_size = 0
    best_i = 0
    best_c = Array{Int64}([])
    for i in 1:n
        if length(clusters[i]) > best_size
            best_size = length(clusters[i])
            best_i = i
            best_c = clusters[i]
        end
    end
    return (best_i, best_size, best_c)
end

function check_regions(S::Vector{Int64}, T::Vector{Int64}, dist::Matrix{Float64})
    n = size(dist)[1]
    for i in S
        for j in T
            @assert dist[i,j] > 1/sqrt(log(n))
        end
    end
end

function frechet_embedding(dist::Matrix{Float64}, S::Vector{Int64})
    return vec(minimum(dist[:,S], dims = 2))     #f_S(u) = min_{v∈S} d(u,v)
end

function arv_core(emb::Matrix{Float64}, c::Float64=0.3; verbose::Bool=true)
    n = size(emb)[1]
    # 1 - compute distances from embedding
    dist = compute_dist(emb) .* n^2

    # 2 - check "easy case" by doing DFS from each node
    clust_small = get_clusters(dist, 0.25)
    i0, best_s, best_c = best_cluster(clust_small)
    if best_s ≥ n/4
        verbose && println("easy case - constant approximation")
        S = best_c
    else
        verbose && println("hard case - logn approximation")
        # 3 - main case: gaussian projection and refinement
        clust_large = get_clusters(dist, 2.0)
        i1, best_s1, best_c1 = best_cluster(clust_large)

        d = Normal()
        g = rand(d, n)
        y = vec(g' * emb)    #Gaussian projection: y_i = ⟨X_i,g⟩, or y[i] = ⟨emb[:,i],g⟩
        ord = sortperm(y)
        c_idx = Int(floor(n * c))
        S = ord[1:c_idx]
        T = ord[(n-c_idx):n]

        rem_S = Vector{Int}([])     #far-away regions refinement
        rem_T = Vector{Int}([])
        for i in S
            for j in setdiff(T, rem_T)
                if dist[i,j] ≤ 1/sqrt(log(n))
                    push!(rem_S, i)
                    push!(rem_T, j)
                    break
                end
            end
        end

        filter!(x->x∉rem_S, S)
        filter!(x->x∉rem_T, T)
        check_regions(S, T, dist)
    end

    # 4 - compute frechet embedding - will be used to grow region later
    f = frechet_embedding(dist, S)
    return f
end

function arv(g::MyGraph.Graph, emb::Matrix{Float64}, c::Float64=0.3; verbose=true)
    f = arv_core(emb, c; verbose=verbose)
    cut = MyGraph.sweep(g, f; loss="sparsity", verbose=verbose)
    return cut
end

end #module