using ArgParse
using Arpack
using CSV, DataFrames
using LightGraphs
using LinearMaps
using LinearAlgebra, Random, Statistics
using SparseArrays

include("EvolvingNetworks.jl")
include("Utils.jl")


"""
	get_min_ratio(λs, n_skip)

Compute the ratios of successive eigenvalues and return the vector of eigvals
sorted in decreasing order, as well as the index of the minimizing ratio.
"""
function get_min_ratio(λs, n_skip)
	sort!(λs, rev=true); ratios = λs[2:end] ./ λs[1:(end-1)]
	rIdx = first(filter(x -> x > n_skip, sortperm(ratios)))
	return λs, rIdx
end


"""
	update_eigvals(Anew, Vnew, λs, γ, ϵ, qPred, r, use_bk, skip_last) -> (rIdx, Lall)

Perform phase 2 of the eigenvalue estimation algorithm, applying randomized
subspace or block krylov (if `use_bk == true`) on the deflated matrix
`(I - Vnew * Vnew') * Anew * (I - Vnew * Vnew')`, given the dominant eigenvalues
`λs`, the assumed decay factor `γ`, a target accuracy `ϵ`, the number of higher
order eigenvalues `r` and the number `skip_last` of eigenvalue ratios to avoid
considering.
"""
function update_eigvals(Anew, Vnew, Vold, λs, γ, ϵ, nHi, use_bk, skip_last)
    n = size(Anew)[1]; ℓ = nHi + 6  # oversampling factor
    if use_bk
		Vold[:, 1:nHi], λHi, qIter = Utils.bkvals(Anew, nHi, Vold, ϵ,
												  getVecs=true, Vp=Vnew)
    else
		Vold[:, 1:nHi], λHi, qIter = Utils.sivals(Anew, ℓ, ϵ, nconv=nHi, V0=Vold,
												  getVecs=true, Vp=Vnew)
    end
	λAll, rIdx = get_min_ratio(abs.(vcat(λs, λHi)), skip_last)
	return rIdx, λAll, Vold, qIter
end


# empty!(E): replace all elements of `E` with zeros.
empty!(E::SparseMatrixCSC) = begin
	fill!(E, zero(eltype(E))); dropzeros!(E)
end


# pert_norm(graph, α): bound predicted by theory given `α`-sparse update.
pert_norm(graph, α) = begin
	d = degree(graph); κ = maximum(d) / minimum(d)
	return α + (α / (1 + α))^2 + α * sqrt(κ)
end


"""
    generate_perturbation(graph, α, nEdits)

Generate a perturbation corresponding to `nEdits` edge additions or deletions,
such as the ratio of edits-over-degree for each node is kept below a provided
threshold `α`.
"""
function generate_perturbation!(graph, α, nEdits, E::SparseMatrixCSC)
	eDone = 0; n, _ = size(E); D = degree(graph); empty!(E)
	while eDone < nEdits
		# generate v-pair
		i = rand(1:(n-1)); j = rand((i+1):n)
		iRatio = (sum(abs.(E[i, :])) + 1) / D[i]
		jRatio = (sum(abs.(E[j, :])) + 1) / D[j]
		((iRatio > α) || (jRatio > α)) && continue  # skip if breaking change
		E[i, j] = E[j, i] = (has_edge(graph, i, j) ? -1 : 1)
		eDone += 1
	end
	return E
end


function track_sbm(n, num_comms, num_edits, monitor_freq;
				   p, q, α, γ, δ, ϵ, use_bk)
    graph = EvolvingNetworks.genSBM(n, num_comms, p=p, q=q)
    Anorm = EvolvingNetworks.norm_adjacency(graph)
    # preallocate E matrix
    Emat = SparseMatrixCSC{Int64, Int64}(sparse([], [], [], n, n))
    @info("Preallocated E")
    # get initial evals / evecs
    max_dim = 15; nHi = 5
	D, V0, _ = eigs(Anorm, nev=max_dim); VHi = Utils._qrDirect(randn(n, nHi))
	λs, rIdx = get_min_ratio(D, 1); λ1, λr, λr₊ = λs[[1, rIdx, rIdx+1]]
	V0 = V0[:, 1:rIdx]; VHi = Utils._qrDirect(randn(n, nHi))
    dsTheo = []; dsPrac = []   # theoretical and measured subspace dists
	pracSteps = []; theoSteps = []; perfSteps = []  # steps (bounds & actual)
	for i = 1:num_edits
		generate_perturbation!(graph, α, monitor_freq, Emat)
		αCurr = maximum(abs.(sum(Emat, dims=2)) ./ degree(graph))
		@info("αCurr: $(αCurr)")
		# update edges based on Emat
		for (i, j, val) in zip(findnz(Emat)...)
			if val == 1
				add_edge!(graph, i, j)
			else
				rem_edge!(graph, i, j)
			end
		end
		Anew = EvolvingNetworks.norm_adjacency(graph)
		E = LinearMap(X -> Anew * X - Anorm * X, n, n, issymmetric=true)
		Enorm = abs(first(eigs(E, which=:LM, nev=1, tol=ϵ^2)[1]))
		# add perturbation norms
		push!(dsTheo, pert_norm(graph, αCurr)); push!(dsPrac, Enorm)
		@show("|E|_2: $(dsTheo[end]) vs. $(dsPrac[end])")
		# use dsTheo to compute iteration upper bounds
		Δtheo = 2 * (dsTheo[end]) / (λr - λr₊ - (3 * ϵ^2))
		Δprac = Utils.davKahProxy(V0, E, Enorm, ϵ, λ1, λr, λr₊)
		# prediction using a-priori bounds
		nPredBK, nPredSI = Utils.getIterBounds(Δprac, ϵ, λr, λr₊, Enorm, γ, n)
		nTheoBK, nTheoSI = Utils.getIterBounds(Δtheo, ϵ, λr, λr₊, Enorm, γ, n)
		if use_bk
			Vnew, λs, nIter = Utils.block_krylov(Anew, rIdx, V0, ϵ,
												 maxiter=nPredBK)
		else
			Vnew, λs, nIter = Utils.subspace_iteration(Anew, rIdx, nPredSI, ϵ,
													   V0=V0)
		end
		rIdx, λAll, VHi, qIter = update_eigvals(Anew, Vnew, VHi, λs, γ, ϵ,
												nHi, use_bk, 1)
		λ1, λr, λr₊ = λAll[[1, rIdx, rIdx+1]]
		@info("Updated subspace size: $(rIdx)")
		@info("λs: $(λ1), $(λr), $(λr₊)")
		# update everything
		V0 = copy(Vnew)
		append!(pracSteps, (use_bk ? nPredBK : nPredSI))
		append!(theoSteps, (use_bk ? nTheoBK : nTheoSI))
		append!(perfSteps, nIter)
		Anorm = Anew
	end
	return perfSteps, pracSteps, theoSteps, dsPrac, dsTheo
end


s = ArgParseSettings(description="""
    Monitor a series of sparse edge updates to an SBM and measure both the
    theoretically predicted as well as actual resulting subspace distances.""")
@add_arg_table s begin
	"--monitor_freq"
		help = "The monitoring frequency"
		arg_type = Int
		default = 10
    "--n"
        help = "The number of nodes"
        arg_type = Int
        default = 1000
    "--p"
        help = "The intercluster edge probability"
        arg_type = Float64
        default = 0.5
    "--q"
        help = "The intracluster edge probability"
        arg_type = Float64
        default = 0.1
    "--num_comms"
        help = "The number of SBM communities"
        arg_type = Int
        default = 10
    "--num_edits"
        help = "The total number of edits to the graph"
        arg_type = Int
        default = 100
	"--seed"
		help = "The random seed for the RNG"
		arg_type = Int
		default = 999
	"--alpha"
		help = "The sparsity value for each modification"
		arg_type = Float64
		default = 0.1
	"--gamma"
		help = "The eigenvalue decay parameter γ"
		arg_type = Float64
		default = 0.1
	"--delta"
		help = "The probability of failure for randomized iteration"
		arg_type = Float64
		default = 1e-4
	"--eps"
		help = "The desired subspace distance to retain"
		arg_type = Float64
		default = 1e-3
	"--use_bk"
		help = "Set to use a block Krylov method instead of subspace iteration"
		action = :store_true
end
parsed = parse_args(s); Random.seed!(parsed["seed"])
n, p, q = parsed["n"], parsed["p"], parsed["q"]
num_comms, num_edits = parsed["num_comms"], parsed["num_edits"]
α, γ, δ, ϵ = parsed["alpha"], parsed["gamma"], parsed["delta"], parsed["eps"]
monitor_freq, use_bk = parsed["monitor_freq"], parsed["use_bk"]

perf_steps, pracSteps, theoSteps, dsPrac, dsTheo =
	track_sbm(n, num_comms, num_edits, monitor_freq, p=p, q=q, α=α, γ=γ, δ=δ,
			  ϵ=ϵ, use_bk=use_bk)
df = DataFrame(k=(1:num_edits), steps=perf_steps, nPredPrac=pracSteps,
			   nPredTheo=theoSteps, dsPrac=dsPrac, dsTheo=dsTheo)
CSV.write("theory_sbm_$(n)_$(num_comms)_$(num_edits)_$(monitor_freq)_$(α).csv", df)
