using ArgParse
using Arpack
using CSV, DataFrames
using LinearAlgebra, Random, Statistics
using LinearMaps
using PyPlot

include("EvolvingNetworks.jl")
include("Utils.jl")


"""
	getRatio(Rs, skip)

Sort the eigenvalues in `Rs` in decreasing order of magnitude and return the
index of the minimizing ratio of successive eigenvalues, as well as the sorted
list of them.
"""
function getRatio(Rs, skip)
	sort!(Rs, rev=true); ratios = Rs[2:end] ./ Rs[1:(end-1)]
	sSize = first(filter(x -> x > skip, sortperm(ratios)))
	return sSize, Rs
end


function genMatrix(n, d)
	Q = Matrix(qr(randn(n, d)).Q); Σ = (2.0 .* randn(d)).^2;
	return Q * Diagonal(Σ) * Q'
	# A = randn(n, d); return (1 / sqrt(d)) * A * A'
end


#= genPert(n): create a low-rank matrix =#
function genPert(n)
	v = randn(n) / sqrt(n); ϵi = (rand() <= 0.5) ? 1.0 : -1.0
	return ϵi, v
end


#= genPertFull(n): create a matrix with i.i.d. entries N(0,1/n^2) =#
function genPertFull(n)
	return (1 / n) * randn(n, n)
end


"""
	update_eigvals(Anew, Vnew, Vold, λs, ϵ, nHi, use_bk, skip_last)

Compute `nHi` higher order eigenvalues of the deflated matrix
`(I - Vnew * Vnew') * Anew * (I - Vnew * Vnew')` starting from the previous
estimate `Vold`, up to target accuracy `ϵ`.
"""
function update_eigvals(Anew, Vnew, Vold, λs, ϵ, nHi, use_bk, skip_last)
	n = size(Anew)[1]; ℓ = nHi + 5  # oversampling factor
	if use_bk
		Vold[:, 1:nHi], λHi, qIter = Utils.bkvals(Anew, nHi, Vold, ϵ,
												  getVecs=true, Vp=Vnew)
	else
		Vold[:, 1:nHi], λHi, qIter = Utils.sivals(Anew, ℓ, ϵ, nconv=nHi, V0=Vold,
												  getVecs=true, Vp=Vnew)
	end
	sSize, λAll = getRatio(abs.(vcat(λs, λHi)), skip_last)
	return sSize, λAll, Vold, qIter
end


function track_full(n::Int, maxDim::Int, numEdits::Int;
					γ::Float64, δ::Float64, ϵ::Float64,
					use_bk::Bool, rnd_guess::Bool)
	r = trunc(Int, sqrt(n))
	A = (1 / (r * log(n))) * randn(n, r) * randn(r, n)
	Amap = LinearMap(X -> A' * (A * X), n, n, issymmetric=true)
	D, V, _ = eigs(Amap, nev=maxDim)  # true evals, evecs
	sSize, Dr = getRatio(D, 1); s1, sr, sr₊ = Dr[[1, sSize, sSize+1]];
	@info("σs: $(sr), $(sr₊) - sSize: $(sSize)")
	# rough eigval estimates
	# (add +1 since we work with Anorm + 1.0I in the sequel)
	# eigvec - keep correct dimension
	V0 = V[:, 1:sSize]; nHi = 5; VHi = Utils._qrDirect(randn(n, nHi))
	# steps elapsed, predicted (BK) and predicted (SI)
	subSteps = []; subPredBK = []; subPredSI = []
	# subspace distance (true and predicted)
	subDistT = []; subDistP = []
	# accumulated edits
	for j = 1:numEdits
		# update A
		E = genPertFull(n)
		(j % 10 == 0) && println("Running repeat $(j)...")
		Anew = A + E;
		# get new singular values
		Amap = LinearMap(X -> Anew' * (Anew * X), n, n, issymmetric=true)
		# norm of perturbation and perturbation of singular values
		Enorm = (1.1 / sqrt(n))
		σPert = (sqrt(s1) + sqrt(sr) + log(n)) / n
		@info("sr: $(sr), sr₊: $(sr₊), σPert: $(σPert)")
		if rnd_guess
			V0 = Utils._qrDirect(randn(n, sSize))
			nPredBK = nPredSI = n
			Vnew, λs, endIt = begin
				if use_bk
					Utils.block_krylov(Amap, sSize, V0, ϵ, maxiter=nPredBK)
				else
					Utils.subspace_iteration(Amap, sSize, nPredSI, ϵ, V0=copy(V0))
				end
			end
			Δ = opnorm(V0 * V0' - Vnew * Vnew')
			# set correct predictions if using random guess
			nPredBK, nPredSI = Utils.getIterBounds(Δ, ϵ, sr, sr₊, Enorm, γ, n)
		else
			# compute Δ bound
			Δdenom = sr - sr₊ - 2 * σPert
			Δ = ((2 / sqrt(n)) + (sqrt(sSize) / n)) / Δdenom
			nPredBK, nPredSI = Utils.getIterBounds(Δ, ϵ, sr, sr₊, Enorm, γ, n)
			Vnew, λs, endIt = begin
				if use_bk
					Utils.block_krylov(Amap, sSize, V0, ϵ, maxiter=nPredBK)
				else
					Utils.subspace_iteration(Amap, sSize, nPredSI, ϵ, V0=copy(V0))
				end
			end
		end
		Δex = first(svds(LinearMap(X -> V0 * (V0' * X) - Vnew * (Vnew' * X),
								   n, n, issymmetric=true), tol=1e-6)[1].S)
		@info("Δs: $(Δ) vs. $(Δex)")
		append!(subDistT, Δex)
		append!(subDistP, Δ)
		append!(subSteps, endIt)
		append!(subPredBK, nPredBK)
		append!(subPredSI, nPredSI)
		V0 = copy(Vnew)
		# update high-order eigenvalues for next iteration
		sSize, λAll, VHi, qIter =
			update_eigvals(Amap, Vnew, VHi, λs, ϵ, nHi, use_bk, 2)
		s1, sr, sr₊ = λAll[[1, sSize, sSize+1]]
		@info("Updated subspace size: $(sSize)")
		# update A
		A += E
	end
	# return all step statistics
	return subSteps, subPredBK, subPredSI, subDistT, subDistP
end



function track_lowrank(n::Int, d::Int, maxDim::Int, numEdits::Int;
                       γ::Float64, δ::Float64, monitor_freq::Int,
					   ϵ::Float64, use_bk::Bool, rnd_guess::Bool)
    Asym = genMatrix(n, d)
    D, V, _ = eigs(Symmetric(Asym), nev=maxDim)  # true evals, evecs
	sSize, Dr = getRatio(D, 1); λ1, λr, λr₊ = Dr[[1, sSize, sSize+1]]
	@info("λs: $(Dr) - sSize: $(sSize)")
	# eigvec - keep correct dimension
	V0 = V[:, 1:sSize]; nHi = 5; VHi = Utils._qrDirect(randn(n, nHi))
	# steps elapsed, predicted (BK) and predicted (SI)
	subSteps = []; subPredBK = []; subPredSI = []
	# subspace distance (true and predicted)
	subDistT = []; subDistP = []
	# accumulated edits
	for j = 1:numEdits
		# update Asym
		ϵi, v = genPert(n); BLAS.ger!(ϵi, v, v, Asym); E = ϵi * (v .* v')
		(j % 10 == 0) && println("Running repeat $(j)...")
		# norm of perturbation and perturbation of singular values
		Enorm = (1.1 / sqrt(n))
		Ptheo = sqrt(sSize / n) + sqrt(2 * log(n) / n)
		if rnd_guess
			V0 = Utils._qrDirect(randn(n, sSize))
			nPredBK = nPredSI = n
			Vnew, λs, endIt = begin
				if use_bk
					Utils.block_krylov(Asym, sSize, V0, ϵ, maxiter=nPredBK)
				else
					Utils.subspace_iteration(Asym, sSize, nPredSI, ϵ, V0=copy(V0))
				end
			end
			Δ = opnorm(V0 * V0' - Vnew * Vnew')
			# set correct predictions if using random guess
			nPredBK, nPredSI = Utils.getIterBounds(Δ, ϵ, λr, λr₊, Enorm, γ, n,
												   r=sSize, δ=δ)
			@info("Δ: $(Δ)")
		else
			# compute Δ bound
			Δdenom = λr - λr₊
            Δ = Ptheo / Δdenom
			nPredBK, nPredSI = Utils.getIterBounds(Δ, ϵ, λr, λr₊, Enorm, γ, n,
												   r=sSize, δ=δ)
			ρt = (λr - Enorm) / (λr₊ + Enorm)
			@info("Ratio: $(ρt)")
			Vnew, λs, endIt = begin
				if use_bk
					Utils.block_krylov(Asym, sSize, V0, ϵ, maxiter=nPredBK)
				else
					Utils.subspace_iteration(Asym, sSize, nPredSI, ϵ, V0=copy(V0))
				end
			end
		end
		Δex = first(svds(LinearMap(X -> V0 * (V0' * X) - Vnew * (Vnew' * X),
								   n, n, issymmetric=true), tol=1e-6)[1].S)
		@info("Δ: $(Δ) - Δex: $(Δex)")
		append!(subDistT, Δex)
		append!(subDistP, Δ)
		append!(subSteps, endIt)
		append!(subPredBK, nPredBK)
		append!(subPredSI, nPredSI)
		V0 = copy(Vnew)
		# update high-order eigenvalues for next iteration
		sSize, λAll, VHi, qIter =
			update_eigvals(Asym, Vnew, VHi, λs, ϵ, nHi, use_bk, 2)
		λ1, λr, λr₊ = λAll[[1, sSize, sSize+1]]
		@info("Updated subspace size: $(sSize)")
	end
	# return all step statistics
	return subSteps, subPredBK, subPredSI, subDistT, subDistP
end


s = ArgParseSettings(description="""
	Monitor the connected components of a real dataset by counting the
	number of eigenvalues of the normalized laplacian which are close to 0.""")
@add_arg_table s begin
	"--n"
		help = "Number of data points"
		arg_type = Int
		default = 500
	"--d"
		help = "Dimension of data"
		arg_type = Int
		default = 100
	"--max_dim"
		help = "Maximum number of principal components"
		arg_type = Int
		default = 10
	"--num_edits"
		help = "The number of edits"
		arg_type = Int
		default = 500
	"--monitor_freq"
		help = "The monitoring frequency"
		arg_type = Int
		default = 10
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
	"--random_guess"
		help = "Set to report number of iterations predicted by random guess"
		action = :store_true
	"--use_bk"
		help = "Use a block krylov method instead of randomized subspace iter."
		action = :store_true
	"--update_type"
		help = "Choose the type of random updates - one of {full, low_rank}"
		range_tester = (x -> lowercase(x) in ["full", "low_rank"])
end
parsed = parse_args(s); Random.seed!(parsed["seed"])
monitor_freq = parsed["monitor_freq"]
n, d, maxDim, nEdits = parsed["n"], parsed["d"], parsed["max_dim"], parsed["num_edits"]
γ, δ, ϵ = parsed["gamma"], parsed["delta_fail"], parsed["eps"]
use_bk, rnd_guess = parsed["use_bk"], parsed["random_guess"]
upd_type = lowercase(parsed["update_type"])

if upd_type == "low_rank"
	sSteps, sPredBK, sPredSI, subT, subP = track_lowrank(
		n, d, maxDim, nEdits, monitor_freq=monitor_freq, γ=γ,
		δ=δ, ϵ=ϵ, use_bk=use_bk, rnd_guess=rnd_guess)
	df = DataFrame(k=(1:length(sSteps)) .* monitor_freq,
				   sSteps=sSteps, sPredBK=sPredBK, sPredSI=sPredSI,
				   subT=subT, subP=subP)
	CSV.write("track_pca_$(n)x$(d)_freq-$(monitor_freq)_bk_$(use_bk)_rnd_$(rnd_guess).csv", df)
else
	sSteps, sPredBK, sPredSI, subT, subP = track_full(
		n, maxDim, nEdits, γ=γ, δ=δ, ϵ=ϵ, use_bk=use_bk,
		rnd_guess=rnd_guess)
	df = DataFrame(k=(1:length(sSteps)) .* monitor_freq,
				   sSteps=sSteps, sPredBK=sPredBK,
				   sPredSI=sPredSI, subT=subT, subP=subP)
	CSV.write("track_pca_full_$(n)_freq-$(monitor_freq)_bk_$(use_bk)_rnd_$(rnd_guess).csv", df)
end

