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
	update_eigvals(Anew, Vnew, λs, γ, ϵ, qPred, r, skip_last) -> (sSize, λAll, Vold, qIter)

Perform phase 2 of the eigenvalue estimation algorithm, applying randomized
subspace or block krylov (if `use_bk == true`) on the deflated matrix
`(I - Vnew * Vnew') * Anew * (I - Vnew * Vnew')`, given the dominant eigenvalues
`λs`, the assumed decay factor `γ`, a target accuracy `ϵ`, the number of higher
order eigenvalues `r` and the number `skip_last` of eigenvalue ratios to avoid
considering.
"""
function update_eigvals(Anew, Vnew, Vold, λs, γ, ϵ, nHi, skip_last)
    n = size(Anew)[1]; ℓ = nHi + 5  # oversampling factor
	Vold[:, 1:nHi], λHi, qIter = Utils.bkvals(Anew, nHi, Vold, ϵ,
											  getVecs=true, Vp=Vnew)
	λAll = abs.(vcat(λs, λHi))
	conn_comp = sum(abs.(λAll .- maximum(λAll)) .<= ϵ)
	sSize, λAll = Utils.getMinRatio(abs.(vcat(λs, λHi)), conn_comp + skip_last)
	return sSize, λAll, Vold, qIter
end

# opnorm of a LinearMap
lmap_opnorm(Lmap) = first(svds(Lmap, ritzvec=false, tol=1e-6, nsv=1)[1].S)


"""
	resize_subspace(A, V0, VHi, sSize)

Appropriately resize the subspace `V0` when `sSize != size(V0)[2]`, using the
leading `sSize` Ritz vectors of the larger subspace `[V0, VHi]` associated to
`A`.
"""
function resize_subspace(A, V0, VHi, sSize)
    sDiff = sSize - size(V0)[2]
    if (sDiff > 0)
            # add excess vectors from
            V0 = hcat(V0, VHi[:, 1:sDiff])
    else
        if (sDiff < 0)  # keep leading `sSize` vectors
            V0, _ = Utils.rayleigh_ritz(A, hcat(V0, VHi), sSize)
        end
    end
    return V0
end


"""
	update_subspace(A, V0, use_bk, sSize, ϵ)

Run the LOBPCG or subspace iteration methods, depending on whether `use_bk` has
been set, on the matrix `A` with starting guess `V0` to find an `ϵ`-accurate
subspace of size `sSize`.
"""
function update_subspace(A, V0, use_bk, sSize, ϵ)
	n = size(A)[1]
    if use_bk
		return @timed Utils.block_krylov(A, sSize, copy(V0), ϵ, maxiter=n)
    else
		return @timed Utils.subspace_iteration(A, sSize, n, ϵ, V0=copy(V0))
    end
end


# Track the invariant subspace
# ============================
function track_subspace(gSrc, gTgt, eAdd; monitor_freq::Int, ratio::Float64,
						γ::Float64, δ::Float64, ϵ::Float64, use_bk::Bool)
	@info("Network init - [OK]. Remaining edges: $(length(eAdd)) - |E|: $(gTgt.ne)")
	gComp = connected_components(gTgt)[1]
	Anorm = EvolvingNetworks.reg_adjacency(gSrc)
	# preallocate E matrix
	n = size(Anorm)[1]
	Emat = SparseMatrixCSC{Float64, Int64}(sparse([], [], [], n, n))
    # initial dimensions and evals / evecs
    max_dim = 120; nHi = 5; V0 = Utils._qrDirect(randn(n, max_dim))
    V0, D, _ = Utils.block_krylov(Anorm, max_dim, V0, ϵ, maxiter=n)
	# get number of connected components
    conn_comp = sum(abs.(D .- maximum(D)) .<= ϵ)
	sSize, _ = Utils.getMinRatio(abs.(D), conn_comp + 1)
    λ1, λr, λr₊ = D[[1, sSize, sSize+1]]
    # isolate initial eigenspace and higher order eigenspace
    V0 = V0[:, 1:sSize]; @info("λs: $(λr), $(λr₊)")
    G = randn(n, nHi); VHi = Utils._qrDirect(G - V0 * (V0' * G))
    # stat lists
	nImp = []; Δs = []; Δother = []; steps = []; stepsHi = []
	bkPred = []; siPred = []; core = []; high = []
    # number of added edges
    eAdded = 0
    for j = 1:length(eAdd)
		# add edge to graph
		(ei, ej) = pop!(eAdd); has_edge(gSrc, ei, ej) && continue
        add_edge!(gSrc, ei, ej); eAdded += 1
		if (eAdded % monitor_freq) == 0
			Anew = EvolvingNetworks.reg_adjacency(gSrc)
			E = LinearMap(X -> Anew * X - Anorm * X, n, n, issymmetric=true)
            Enorm = lmap_opnorm(E); Anorm = Anew
            # update subspace if new size is different
            V0 = resize_subspace(Anew, V0, VHi, sSize)
			Δ = Utils.davKahProxy(V0, E, Enorm, ϵ, λ1, λr, λr₊)
			(Δ == 0) && continue
			@info("Repeat: $(eAdded) - |E|_2 = $(Enorm) - Δ: $(Δ)")
            push!(Δs, Δ)  # update stat
			# prediction using a-priori bound, if applicable
			nPredBK, nPredSI, eNorm_r, eNorm_r₊ =
				Utils.getBounds(Δ, E, λr, λr₊, ϵ, γ, n, V0, VHi, sSize)
            push!(bkPred, nPredBK); push!(siPred, nPredSI)  # update stats
            # update subspace
			(Vnew, λs, runIt), coreTime, _ =
				update_subspace(Anew, V0, use_bk, sSize, ϵ)
            push!(steps, runIt)
			# update high order eigenvalues for next iteration
			(sSize, λAll, VHi, qIter), hiTime, _ =
				@timed update_eigvals(Anew, Vnew, VHi, λs, γ, ϵ, nHi, 1)
			# update subspace size and new λ1, λr, λr₊, V0
			λ1, λr, λr₊ = λAll[[1, sSize, sSize+1]]; V0 = copy(Vnew)
			# update statistics
			append!(nImp, sSize); append!(stepsHi, qIter)
			append!(core, coreTime); append!(high, hiTime)
			# update linear map
			Anorm = Anew
		end
	end
	# Return all stats
	return steps, siPred, bkPred, stepsHi, nImp, Δs, core, high
end

s = ArgParseSettings(description="""
	Track the leading eigenspace of the temporal-reddit-reply dataset using
	the proposed adaptive method.""")
@add_arg_table s begin
	"--filepath"
		help = "Path to file containing data"
		arg_type = String
		default = "datasets/temporal-reddit-reply.txt"
	"--monitor_freq"
		help = "The monitoring frequency"
		arg_type = Int
		default = 10
	"--ratio"
		help = "The ratio of edges in the starting graph and the static version"
		arg_type = Float64
		default = 0.9
	"--seed"
		help = "The random seed for the RNG"
		arg_type = Int
		default = 999
	"--gamma"
		help = "The eigenvalue decay parameter γ"
		arg_type = Float64
		default = 0.1
	"--delta_fail"
		help = "The probability of failure for randomized iteration"
		arg_type = Float64
		default = 1e-4
	"--eps"
		help = "The desired subspace distance to retain"
		arg_type = Float64
		default = 1e-3
	"--record_true"
		help = "Set to record true subspace distance at each iter"
		action = :store_true
	"--max_nodes"
		help = "Maximum number of vertices to keep"
		arg_type = Int
		default = 100000
	"--use_bk"
		help = "Set to use a block Krylov method instead of subspace iteration"
		action = :store_true
end
parsed = parse_args(s); Random.seed!(parsed["seed"])
monitor_freq, ratio = parsed["monitor_freq"], parsed["ratio"]
γ, δ_fail, ϵ = parsed["gamma"], parsed["delta_fail"], parsed["eps"]
use_bk, record_true = parsed["use_bk"], parsed["record_true"]
max_nodes = parsed["max_nodes"]
filepath = parsed["filepath"]

# get the temporal-reddit-reply graph
gSrc, gTgt, eAdd =
	EvolvingNetworks.genRedditReplyGraphs(filepath, max_nodes, ratio)
# number of nodes
nTotal = size(adjacency_matrix(gTgt))[1]

# retrieve statistics of interest
steps, sPredSI, sPredBK, stepsHi, nImp, Δs, core, high =
track_subspace(
	gSrc, gTgt, eAdd, monitor_freq=monitor_freq, ratio=ratio, γ=γ,
	δ=δ_fail, ϵ=ϵ, use_bk=use_bk)
df = DataFrame(k=(1:length(steps)) .* monitor_freq,
			   steps=steps, stepsHi=stepsHi, bkPred=sPredBK,
			   siPred=sPredSI, nImp=nImp, Ds=Δs,
			   coreTime=core, highTime=high)
# write data to output
CSV.write("track_reddit_max_nodes-$(max_nodes)_freq-$(monitor_freq)_bk_$(use_bk).csv", df)
