using ArgParse
using Arpack
using CSV, DataFrames
using LinearAlgebra, Random, Statistics
using LinearMaps
using ToeplitzMatrices

include("EvolvingNetworks.jl")
include("IOUtils.jl")
include("Utils.jl")


"""
	avg_adiag!(A)

Average `A` over its antidiagonals, modifying it in-place, returning the
modified matrix.
"""
function avg_adiag!(A)
	r, c = size(A); Aflip = reverse(A, dims=1)  # flip along columns
	for idx = (-(r-1)):(c-1)  # average across all diagonals
		Aflip[diagind(Aflip, idx)] .= mean(diag(Aflip, idx))
	end
	return reverse(Aflip, dims=1)  # flip again
end


"""
	hankel_AT(A, X)

Efficient implementation of `X -> A' * X` when A is a `Hankel` matrix. Employs
the Toeplitz structure of `A.T`.
"""
function hankel_AT(A, X)
	return reverse(Toeplitz(A.T') * X)
end


"""
	avg_adiag(A)

Average `A` over its antidiagonals, returning a copy containing the modified
matrix. The original matrix `A` is preserved.
"""
function avg_adiag(A)
	return avg_adiag!(copy(A))
end


"""
	lowRankReconstruct(A, U, Σ; avg_each=false)

Reconstruct a Hankel matrix `A` with SVD given by `A = U₁ Σ₁ V₁^T + U₂ Σ₂ V₂^T`
given a set of approximate singular vectors `U` and singular values `Σ` such
that `U ≃ U₁` and `Σ ≃ Σ₁`. Hankelization (averaging across the antidiagonals)
is performed for each factor separately if `avg_each=true` (default: `false`).
"""
function lowRankReconstruct(A, U, Σ; avg_each=false)
	r = size(U)[2]; Hmat = fill(0.0, size(A))
	if avg_each  # average each factor
		for idx = 1:r
			Hmat += avg_adiag!(U[:, idx] .* (U[:, idx]' * A))
		end
	else
		Hmat = U * (U' * A); @show size(Hmat)
		avg_adiag!(Hmat)
	end
	return Hankel(Hmat)
end



"""
	getRatio(opA, ϵ, Vnew, λs, VHi, nHi, skip_last)

Get the minimum eigenvalue ratio for the operator `opA` using the `λs` dominant
eigenvalues as well as `nHi` high order eigenvalues, given the eigenspace `Vp`,
the high-order eigenspace `VHi`, the desired accuracy `ϵ` as well as the number
of ratios to skip `skip_last`.
"""
function getRatio(opA, ϵ, Vnew, λs, VHi, nHi, skip_last)
	n = size(opA)[1]
	VHi[:, 1:nHi], λHi, qIter = Utils.bkvals(opA, nHi, VHi, ϵ,
											 getVecs=true, Vp=Vnew)
	D = sort(abs.(vcat(λs, λHi)))
	ratios = D[1:(end-1)] ./ D[2:end]
	rIdx = first(filter(x -> x < length(D) - skip_last,
						sortperm(ratios)))
	return rIdx, D, VHi, qIter
end


# extractTS(A::Hankel): extract the time series out of a Hankel matrix.
extractTS(A::Hankel) = vcat(A[:, 1], A[end, 2:end])


"""
	track_ssa(n, wLen, monitor_freq, ϵ, use_bk, random_guess, max_dim,
			  filepath; avg_each=false, sample_epochs=[])

Perform Singular Spectrum Analysis on time series data every `monitor_freq`
"updates", using a window of length `wLen` and a target dimension of `n`.
Determine the principal components with target accuracy `ϵ`, using up to
`max_dim` principal components.
`avg_each` indicates whether each component should be Hankelized separately
during low-rank reconstruction.
Optionally, `sample_epochs` provides a series of instances during which the
original and reconstructed time series are compared and saved to a .csv file.
"""
function track_ssa(n, wLen, monitor_freq, ϵ, use_bk, random_guess, max_dim,
				   filepath; avg_each=false, sample_epochs=[], skip_last=3)
	tData, _ = IOUtils.readPower(filepath, period=30, step=20)
	nHi = 5; γ = 0.1; sIdx = 1; epochCnt = 0
	eval_meth = use_bk ? :block_krylov : :subspace_iteration
	# create matrix and corresponding linear operator
	A = IOUtils.getLaggedCovMatrix(tData, n, sIdx, wLen); Anew = nothing
	dimA = size(A)
	opA = LinearMap(X -> A * hankel_AT(A, X), dimA[1], dimA[1],
					issymmetric=true)
	# get true svdvals, svdvecs
	V0, D, itersRun = Utils.block_krylov(opA, max_dim+1,
										 Matrix(qr(randn(dimA[1], max_dim+1)).Q), ϵ)
	sort!(D); VHi = Matrix(qr(randn(dimA[1], nHi)).Q)
	ratios = D[1:(end-1)] ./ D[2:end]
	rIdx = first(filter(x -> x < length(D) - skip_last, sortperm(ratios)))
	sSize = length(D) - rIdx
	# eigval estimates
	λ1, λr, λr₊ = D[[end, rIdx+1, rIdx]]; λs = D[rIdx+1:end]
	@info("Initial dimension: $(sSize) - evals: $(λr), $(λr₊)")
	# eigvec - keep correct dimension
	V0 = V0[:, 1:sSize]
	# iteration history + predictions
	runHist = [itersRun]; bkPred = [size(opA)[1]]; siPred = [size(opA)[1]]; nImp = [sSize]
	# copy Anew for now to define Eop
	Anew = Hankel(copy(A))
	Eop = LinearMap(X -> Anew * hankel_AT(Anew, X) - A * hankel_AT(A, X),
					dimA[1], dimA[1], issymmetric=true)
	while true
		sIdx += monitor_freq; epochCnt += 1
		# get new lagged covariance matrix
		try
			Anew = IOUtils.getLaggedCovMatrix(tData, n, sIdx, wLen)
			opA = LinearMap(X -> Anew * hankel_AT(Anew, X),
						    size(Anew)[1], size(Anew)[1], issymmetric=true)
		catch ErrorException
			@info("Reached end of time series. Terminating...")
			break
		end
		# old/new starting indices
		oldIdx = sIdx - monitor_freq; newIdx = oldIdx + (n - wLen + 1)
		Enorm, eTime, _ = @timed abs(first(eigs(Eop, nev=1, ritzvec=false, tol=ϵ)[1]))
		Δ = Utils.davKahProxy(V0, Eop, Enorm, ϵ, λ1, λr, λr₊)
		@info("|E|_2 = $(Enorm) - Δ: $(Δ) - comptime: $(eTime)")
		nPredBK, nPredSI = Utils.getIterBounds(Δ, ϵ, λr, λr₊, Enorm, γ, dimA[1])
		if use_bk
			V0, λs, itersRun = Utils.block_krylov(opA, sSize, V0, ϵ,
												  maxiter=nPredBK)
		else
			V0, λs, itersRun = Utils.subspace_iteration(opA, sSize, nPredSI,
														ϵ, V0=copy(V0))
		end
		rIdx, D, VHi, qIter = getRatio(opA, ϵ, V0, λs, VHi, nHi, skip_last)
		# if epoch is among sample epochs, save time series
		if epochCnt in sample_epochs
			@info("$(epochCnt) checkpoint: saving...")
			Hrec = lowRankReconstruct(Anew, V0, D[1:sSize], avg_each=avg_each)
			TSrec = extractTS(Hrec)  # extract reconstructed time series
			TSorg = extractTS(Anew)  # extract original time series
			CSV.write("power_$(epochCnt).csv",
					  DataFrame(k=1:length(TSorg), org=sqrt(n) .* TSorg,
								rec=sqrt(n) .* TSrec))
			# pop first elt
			pop!(sample_epochs)
		end
		sSize = length(D) - rIdx  # updated size of subspace
		append!(nImp, sSize)
		# update V0 according to whether random guess was set
		V0 = random_guess ? Utils._qrDirect(randn(size(V0)...)) : copy(V0)
		# update run history
		append!(runHist, itersRun)
		append!(bkPred, nPredBK)
		append!(siPred, nPredSI)
		(length(sample_epochs) == 0) && break
		A = Hankel(copy(Anew))
	end
	# return all step statistics
	return runHist, bkPred, siPred, nImp
end

s = ArgParseSettings(description="""
	Track the low-rank component of a time series using Singular Spectrum Analysis.""")
@add_arg_table s begin
	"--filepath"
		help = "The path to the file containing the data"
		arg_type = String
		default = "datasets/household_power.txt"
	"--n"
		help = "Number of data points"
		arg_type = Int
		default = 12096
	"--window_len"
		help = "Window length for SSA"
		arg_type = Int
		default = 4032
	"--max_dim"
		help = "Maximum number of principal components"
		arg_type = Int
		default = 10
	"--monitor_freq"
		help = "The monitoring frequency"
		arg_type = Int
		default = 6
	"--seed"
		help = "The random seed for the RNG"
		arg_type = Int
		default = 999
	"--gamma"
		help = "The eigenvalue decay parameter γ"
		arg_type = Float64
		default = 0.1
	"--eps"
		help = "The desired subspace distance to retain"
		arg_type = Float64
		default = 1e-3
	"--random_guess"
		help = "Set to report number of iterations predicted by random guess"
		action = :store_true
	"--use_bk"
		help = "Use a block krylov method instead of subspace iter."
		action = :store_true
	"--avg_each"
		help = "Set to average each component in time series reconstruction"
		action = :store_true
	"--sample_epochs"
		help = "A list of epochs to compare original and reconstructed time series"
		arg_type = Int
		nargs = '*'
end
parsed = parse_args(s); Random.seed!(parsed["seed"])
monitor_freq = parsed["monitor_freq"]
n, max_dim = parsed["n"], parsed["max_dim"]
φγ, ϵ = parsed["gamma"], parsed["eps"]
use_bk, rnd_guess = parsed["use_bk"], parsed["random_guess"]
wLen = parsed["window_len"]
avg_each, sample_epochs = parsed["avg_each"], parsed["sample_epochs"]
filepath = parsed["filepath"]

# run experiment and save history to .csv file
runHist, bkPred, siPred, nImp = track_ssa(
	n, wLen, monitor_freq, ϵ, use_bk, rnd_guess, max_dim, filepath,
	avg_each=avg_each, sample_epochs=sort(sample_epochs, rev=true))
df = DataFrame(k=(1:length(runHist)) .* monitor_freq,
			   runHist=runHist, bkPred=bkPred, siPred=siPred, nImp=nImp)
CSV.write("track_ssa_$(n)_$(wLen)_freq-$(monitor_freq)_bk_$(use_bk)_rnd_$(rnd_guess).csv", df)
