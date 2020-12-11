# Sparsest Cut

## Implementation of spectral approximation algorithms for the Sparsest Cut problem

## Final Project for 40973 - Computer Science I (Programming) at Bocconi University 

This project contains the implementation of the most famous approximation algorithms for the [Sparsest Cut](https://en.wikipedia.org/wiki/Cut_(graph_theory)#Sparsest_cut) problem, which is a very famous NP-hard problem in Computer Science. In particular, we implemented the Fiedler algorithm (also known as "spectral method") and the Arora-Rao-Vazirani (ARV) algorithm. The latter is actually state-of-the-art in terms of worst-case guarantees, with a $`O(\sqrt{\log n})`$ approximation ratio.

The sparsest cut problem consists in finding a way to separate the vertex set of a graph (i.e. find a cut) in a way that the minimum number of edges are cut and the maximum number of pairs of vertices are separated. Letting $`G=(V,E)`$ be a graph and $`S`$ /> a cut, the set of edges with one endpoint in $`S`$ and one endpoint in $`V \setminus S`$ is denoted $`E(S,V \setminus S)`$, and the degree of a node $`v`$ is $`d_v`$. The two most widely used notion of sparsity of a cut are the following:
- __Cut Conductance__: $`\phi(S) = \frac{E(S,V \setminus S)}{\sum_{v \in S}d_v}`$
- __Uniform Sparsity__: $`usc(S) = \frac{E(S,V \setminus S)}{|S||V \setminus S|}`$

The Fiedler algorithms finds a cut that approximates the one with lowest conductance in the graph, while the ARV algorithm finds a cut that approximates the one with lowest sparsity.

We tried to rely as little as possible on existing Julia code, and wrote from scratch things that are probably already implemented more efficiently by the community (i.e. a graph object and basic operations on graphs) to make the project more complete and self-contained.

---

### Instruction for running the final product

There are 3 files that whose execution in a Julia environment allow the user to see the algorithms in action simply by running ```include("filename.jl")``` (e.g. ```include("run_both.jl")```).

1. **[run_fiedler.jl](run_fiedler.jl)**: This runs the Fiedler algorithm on a randomly generated graph with 50 vertices.
2. **[run_arv.jl](run_arv.jl)**: This runs the ARV algorithm on a randomly generated graph with 12 vertices.
3. **[run_both.jl](run_both.jl)**: This runs both algorithms on a randomly generated graph with 12 vertices. The algorithms minimize different (but in a way equivalent) loss functions, so we also print out how the results compare under both losses. Unsurprisingly, generally each algorithm outperforms the other with respect to the loss over which it is optimized, but the results vary. This is also due to testing on very small graphs (12 nodes), but we defer this discussion to the **Performance bottleneck and final considerations** section.

### Main code description

The core of the project is divided in 6 files:

1. **[Graph.jl](Graph.jl)**: This contains the ```MyGraph``` module, which contains the main object used throughout the project: ```MyGraph.Graph``` (from here on, we will omit the module prefix because it is clear from context). The latter stores a graph through both adjacency list and adjacency matrix representations, along with other useful attributes (degree vector and normalized [Laplacian matrix](https://en.wikipedia.org/wiki/Laplacian_matrix)). The basic constructor is built upon the adjacency list representation. The module also contains the following basic operations on graphs that will be useful in the project:
    - ```erdosrenyi(n::Int, p::Float64)``` builds a random graph according to the [Erdos-Renyi model](https://en.wikipedia.org/wiki/Erd%C5%91s%E2%80%93R%C3%A9nyi_model) and is used for testing.
    - ```bfs(g::Graph)``` and ```connected(g::Graph)``` perform breadth-first-search to check whether a given graph is connected, and deal with trivial cut problems.
    - ```conductance(g::Graph, S::BitArray{1})``` and ```sparsity(g::Graph, S::BitArray{1})``` compute the [conductance](https://en.wikipedia.org/wiki/Conductance_(graph)) and the sparsity of a given cut ```S```, respectively. The conductance is the loss function minimized by the Fiedler algorithm, while the sparsity is the loss function minimized by the ARV algorithm. In the regular case, the conductance is called edge expansion, and it is asymptotically equivalent to the uniform sparsity. This means that they are within a constant factor of each other. More precisely, the following relation holds: $`\phi(G) \le \frac{n}{d}usc(G) \le 2\phi(G)`$. See [1, Chapter 10] for a more detailed discussion.
    - ```update_conductance(g::Graph, S::BitArray{1}, num::Int64, den::Int64, v::Int64, den_true::Int64)``` updates the conductance of cut ```S``` by adding vertex ```v``` efficiently, looking only at the neighborhood of the new vertex instead of the whole graph. ```update_boundary(g::Graph, S::BitArray{1}, num::Int64, v::Int64)``` performs a similar update for the boundary of the graph, which is the only update necessary for the sparsity.
    - ```sweep(g::Graph, f::Vector{Float64}; loss::String="sparsity", verbose::Bool=true)``` is the procedure at the heart of both algorithms. Given a graph and a vector representing an embedding of the vertices on the real line, sort the vertices according to the increasing values of the vector and sequentially add them to a growing cut (which goas from a single vertex to the whole vertex set). The best cut chosen is the one among these ```|V|-1``` cuts with lower ```loss```. The previous update functions make the sweep algorithm very fast, which runs in linear time.


2. **[Fiedler.jl](Fiedler.jl)**: This contains the Fiedler algorithm in the ```Fiedler``` module. Most of the work is done by the ```Graph``` module, because we simply compute the [eigenvector](https://en.wikipedia.org/wiki/Eigenvalues_and_eigenvectors) corresponding to the second smallest eigenvalue of the normalized Laplacian matrix of a graph and feed it to the ```sweep``` algorithm.
3. **[SDP.jl](SDP.jl)**: This contains the ```SDP``` module, which computes a specific relaxation for the sparsest cut problem, called ARV relaxation. In particular, it embeds a graph in a negative-type metric space through [semidefinite programming](https://en.wikipedia.org/wiki/Semidefinite_programming). The main idea is that vertices are mapped to an ```n```-dimensional Euclidean space where the _square_ of the Euclidean norm is itself a valid semi-metric. This idea is a combination of two famous relaxations of the problem with a more intuitive meaning. In practice, the program has ```|V|``` variables, each representing a vertex in the original graph, and ```|V|*|V|*(|V|-1)/2 + 1``` linear constraints, one fixing the value of the denominator of the relaxed loss function (it is scale invariant), and the remaining imposing the triangle inequality of the squared norms among each triplet of vertices. The final product of this module is a function that transforms a ```Graph``` object in a matrix whose columns are ```n```-dimensional embeddings of the vertices. This description is a very high-level summary, and casting the relaxation as a semidefinite program is a quite technical step that requires knowledge about characterizations of [positive semidefinite matrices](https://en.wikipedia.org/wiki/Definite_symmetric_matrix), which is beyond the scope of this project. See [1, Chapters 10-12] for a more detailed explanation of the relaxation and the translation of the ARV relaxation into a valid semidefinite program. To solve the semidefinite program, we relied on Julia's main mathematical progamming library, [JuMP](https://github.com/jump-dev/JuMP.jl), and the [ProxSDP](https://github.com/mariohsouto/ProxSDP.jl) solver. 
4. **[ARV.jl](ARV.jl)**: This file contains the ```ARV``` module, which computes the Arora-Rao-Vazirani algorithm from the embedding produced by ```SDP.jl```. In steps, the algorithm works as follows:
    0. Check whether the graph is disconnected, which would produce a trivial cut of zero sparsity.
    1. If there is a node ```i0``` in the embedded space with a large fraction of vertices concentrated in a ball of radius 0.25 around it, let ```S``` be ```B(i0, 0.25)``` and skip to Step 4.
    2. Otherwise, let ```i1``` be the vertex with the largest ball of radius 2, where "largest" is intended as containing the most vertices. Project ```B(i1, 2)``` on a Gaussian space, and let ```S``` be the 30% fraction of the nodes with smallest projection value and ```T``` be the 30% fraction of nodes with the largest projection value.
    3. Refine the two regions by eliminating pairs of vertices with distance $`\le 1/\sqrt{\log n}`$ in the original embedding (not the projection!).
    4. Compute the Frechet embedding of ```S```, i.e. produce a vector ```f_S``` such that assigns to each vertex its distance from ```S``` in the embedded space: ```f_S(v) = dist(v, S)```.
    5. Finally, feed the Frechet embedding to the ```sweep``` algorithm and output that cut.
With high probability, this algorithms outputs a cut whose sparsity is within a root-logarithmic factor of the optimum.
5. **[test.jl](test.jl)**: This file is basically a workshop for the project, and has been used to gradually test the correctness of the code as it was created. Since the final test functions have been moved to the main files of the previous sections, the only purpose of this file is to give a partial history of the project and it can be ignored.
6. **[speedtest.jl](speedtest.jl)**: This file has been used for profiling the code throughout the project, and brought significant efficiency improvements. Profiling ```SDP.jl``` pointed out that there is a _huge computational bottleneck in solving the semidefinite program that generates the graph embedding_. More about this will be discussed in the final section.

### Coding considerations

The project has been built piece-by-piece through constant testing. A sketch of the evolution of the project can be found by looking at the remaining tests in [test.jl](test.jl). Throughout the code we used several ```@assert``` statement to check important properties, which are commented out in the final product to reduce computational cost. Aside from testing the correctness of the functions, we also spent some care on testing their efficiency. First of all, we used Julia's ```@code_warntype``` macro quite extensively to avoid any type instability issues. This led to the use of specialized matrix types, for example since the undirected Laplacian matrix is symmetric we saved it as a ```Symmetric``` object, which led to more efficient eigenvalue computation. The only remaining type instability occurs in the [```SDP.sdp(g::MyGraph.Graph)```](SDP.jl) function, and it is related to the [_variable_](https://jump.dev/JuMP.jl/stable/variables/) of the semidefinite program. Even though it is specified as a PSD matrix in the model, the type of the entries of the matrix remains unspecified, so its concrete type cannot be inferred a priori. However, this is not a problem because there is a significant computational speed-up after the first run, so the code seems to be compiled in a way such that everything is apparently statically typed. As mentioned above, we also profiled the main functions and their early versions in [speedtest.jl](speedtest.jl) to understand the computational bottlenecks. Careful analysis at the computational trees brought about significant improvements. For example, an initial version of [SDP.jl](SDP.jl) used a lot of ```Set``` objects to create constraint matrices. This made the code more readable but created a huge memory allocation overhead, which was improved by substituting sets with explicit Boolean conditions.

Since the object ```MyGraph.Graph``` is pervasive throughout the project, we structured the modules so that ```MyGraph``` is a submodule of ```Fiedler```, ```SDP``` and ```ARV``` by creating explicit paths among modules. This is convenient because it avoids including multiple identical versions of ```MyGraph``` in the [main file](run_both.jl). Since this was not explicitly discussed during the course, I think a short tutorial might be interesting.
```julia 
# inside main scope 
include("Graph.jl")
include("Fiedler.jl")
include("SDP.jl")
include("ARV.jl")
# do stuff
```
```julia 
# inside MyGraph submodule 
module MyGraph
# do stuff
```
```julia 
# inside any of Fiedler, SDP, or ARV - call it ModuleName
module ModuleName
using ..MyGraph
# do stuff
```
The ```using ..MyGraph``` syntax means "look for the ```MyGraph``` module in the enclosing module of ```ModuleName```", which corresponds to the main scope above. Of course, this whole organization could have been avoided by collapsing the project in a single ```SparsestCut``` module, but we decided to keep separate modules for functions that perform conceptually different operations.

For a more general description of this technique, we refer the reader to the [Julia Documentation](https://docs.julialang.org/en/v1/manual/modules/#Relative-and-absolute-module-paths) and [this discussion](https://discourse.julialang.org/t/organisation-into-submodules-and-multiple-dispatch/40641).

### Performance bottleneck and final considerations

The _Achilles heel_ of this project is the practical infeasibility of the Arora-Rao-Vazirani, which is due to the huge computational cost of solving the semidefinite program. Profiling the code shows that most of the time is spent on the ```optimize!()``` function of the ```JuMP``` module, which is outside our control. For this reason, this project is not suitable for a large-scale implementation, and the results presented are just toy-problems. The main problem is that the ARV semidefinite program has a **cubic number** of constraints. This has a lot of impact on the quality of the results, because the ARV algorithm has a theoretical upper-hand over the Fiedler algorithm only **asymptotically**, and certainly not on the small graphs analyzed. Moreover, the constants used in Step 1 and Step 2 of ```ARV.jl``` are not optimized in practice, they are only sufficient to prove the desired asymptotic bound. 

### Possible improvements and follow-up works

While this project is quite packed, it is certainly not complete and can benefit from the following improvements:
1. Provide some graphical representation of graphs and cuts for a more intuitive geometric understanding of what the algorithms do.
2. The SDP approach is useful in proving theoretical properties of the algorithm, but it is not efficient in practice, and this project confirmed this. There exist equivalent formulations of the ARV relaxation that rely on expander flows and are faster, so their implementation is a promising direction (see [1]).

## References:
[1] Arora S, Rao S, Vazirani U (2008). ["Expander flows, geometric embeddings and graph partitioning"](https://www.cs.princeton.edu/~arora/pubs/arvfull.pdf). Journal of the ACM, 56: 1-37. 

[2] Trevisan L (2016). ["Lecture Notes on Graph Partitioning, Expanders and Spectral Methods"](https://people.eecs.berkeley.edu/~luca/books/expanders-2016.pdf)# SparsestCut
