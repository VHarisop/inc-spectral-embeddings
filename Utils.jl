module Utils

	using Arpack
	using IterativeSolvers
	using LinearAlgebra
    using LinearMaps
	using Random
	using SparseArrays
	using Statistics


    _mkMat(V) = convert(Matrix, V)


	"""
		_qrDirect(A)

	Obtain and return the Q factor from the (reduced) QR factorization of A.
	Overwrites A in-place.
	"""
	function _qrDirect(A)
		return LAPACK.orgqr!(LAPACK.geqrf!(A)...)
	end

	function _qFactor(A)
		return _qrDirect(copy(A))
	end


	"""
		_qr(A)

	Obtain and return the Q and R factors from the (reduced) QR factorization
	of A.
	"""
	function _qr(A)
		d = size(A)[2]
		AA, τ = LAPACK.geqrf!(copy(A)); R = triu(AA)[1:d, :]
		return LAPACK.orgqr!(AA, τ), R
	end



	"""
		pick_rand_idx(probs)

	Pick a random element from `1:n` given a vector of probabilities `probs`
    of length `n`.
	"""
	function pick_rand(probs)
		p = rand(); return findfirst((cumsum(probs) .- p) .>= 0)
	end


    """
        rayleigh_ritz(A::Union{Array, LinearMap, SparseMatrixCSC}, V0, n_pairs)

    Get the `n_pairs` leading Ritz vectors and values for a given matrix `A`
    and subspace `V0`.
    """
    function rayleigh_ritz(A::Union{Array, LinearMap, SparseMatrixCSC}, V0,
                           n_pairs)
        _, Q, D = schur(Symmetric(V0' * _mkMat(A * V0))); D = real.(D)
        inds = sortperm(D, rev=true)[1:n_pairs]  # leading indices
        return (V0 * Q)[:, inds], D[inds]
    end


    # convergence test in subspace iteration
    function _si_convergence(A::Union{Array, SparseMatrixCSC, LinearMap},
							 V0, nconv, ϵ)
        Vp, D = rayleigh_ritz(A, V0, nconv)
		resNorms = sum((_mkMat(A * Vp) - D' .* Vp).^2, dims=1)
		return Vp, D, sum(resNorms .<= ϵ^2)
    end


    """
        subspace_iteration(A, ℓ, iters, ϵ=1e-5; nconv=ℓ, V0=nothing) -> (V, λs, niter)

    Run subspace iteration on the matrix  `A` for at most `iters` iterations
    to compute `nconv` largest eigenpairs, using a block of size `ℓ` and up to
    numerical tolerance `ϵ`.
    Optionally, start from initial guess `V`.
    Returns:
    - `V`: the computed Ritz vectors
    - `λs`: the computed Ritz values
    - `niter`: number of iterations elapsed
    """
	function subspace_iteration(A::Union{AbstractArray, SparseMatrixCSC}, ℓ,
								iters, ϵ=1e-3; nconv=ℓ, V0=nothing,
								Vp::Union{Nothing, Array}=nothing)
        opA = LinearMap(X -> A * X, size(A)[1], size(A)[2], issymmetric=true)
		return subspace_iteration(opA, ℓ, iters, ϵ, nconv=nconv, V0=V0, Vp=Vp)
    end


    """
        subspace_iteration(A::LinearMap, ℓ, iters, ϵ=1e-5;
                           nconv=ℓ, V0=nothing) -> (V, λs, niter)

    Run subspace iteration on the operator `A` for at most `iters` iterations
	to compute `nconv` largest eigenpairs, using a block of size `ℓ` and up to
    numerical tolerance `ϵ`.
	Optionally, start from initial guess `V` and orthogonalize against another
	subspace `Vp`.
    Returns:
    - `V`: the computed Ritz vectors
    - `λs`: the computed Ritz values
    - `niter`: number of iterations elapsed
    """
	function subspace_iteration(A::LinearMap, ℓ, iters, ϵ=1e-5; nconv=ℓ,
								V0=nothing, Vp::Union{Nothing, Array}=nothing)
		n = size(A)[1]  # dimension of im(A)
		V0 = (V0 == nothing) ? _qrDirect(randn(n, ℓ)) : V0
        p = size(V0)[2]
        # take care of size discrepancies
        if (p < ℓ)  # add missing vectors, random sample
            Vrem = randn(n, ℓ - p)
            V0 = hcat(V0, _qrDirect(Vrem - V0 * (V0' * Vrem)))
		elseif (p > ℓ)  # drop vectors corresponding to the extremal Schur vals
            _, Q, D = schur(Symmetric(V0' * _mkMat(A * V0))); V0[:] = V0 * Q
            inds = sortperm(D, rev=true)[1:ℓ]; V0 = V0[:, inds]
        end
		for i = 1:iters
			Vt, λs, numConv = _si_convergence(A, V0, nconv, ϵ)
			(numConv == nconv) && return Vt, λs, i+1
			V0[:] = _mkMat((Vp == nothing) ? A * V0 : A * (V0 - Vp * (Vp' * V0)))
			V0[:] = _qrDirect(V0)
		end
		return V0, sort(eigvals(Symmetric(V0' * _mkMat(A * V0))), rev=true), iters
	end


	"""
		block_krylov(A::Union{Array, SparseMatrixCSC, LinearMap}, r::Int,
					 V0::Array{Float64, 2}, ϵ; maxiter=size(A)[1],
					 Vp::Union{Array, Nothing}) -> (X, λ, niter)

	Run the LOBPCG method with initial guess V0 to compute `r`
	extremal eigenpairs up to tolerance `ϵ`. Optionally set the maximal
    number of iterations `maxiter`, as well as a subspace `Vp` to orthogonalize
	against. Return:
	- `X`: the Ritz vectors found
	- `λ`: the set of computed eigenvalues
	- `niter`: number of iterations required to converge to tolerance ϵ
	"""
	function block_krylov(A::Union{AbstractArray, SparseMatrixCSC, LinearMap},
						  r::Int, V0::Array{Float64, 2}, ϵ::Real;
						  maxiter=size(A)[1], Vp::Union{Array, Nothing}=nothing)
		p = size(V0)[2]; d = size(A)[2]
		if (p < r)  # take care of posdef exception
			G = randn(d, r - p)
			V0 = hcat(V0, _qrDirect(G - V0 * (V0' * G)))
		end
		# make sure to take LOBPCG's spurious PosDefException into accnt.
		try
			lRes = lobpcg(A, true, V0, r, tol=ϵ, maxiter=maxiter, C=Vp)
			return lRes.X, lRes.λ, first(lRes.iterations)
		catch PosDefException
			@warn("PosDefException encountered - falling back to `subspace_iteration`")
			return subspace_iteration(A, r, maxiter, ϵ, V0=V0, Vp=Vp)
		end
	end


	"""
		getMinRatio(λs::Array, n_skip)

	Given an array of eigenvalues `λs`, sort them in algebraically decreasing
	order and return the index of the minimizing ratio of successive eigvals
	as well as the sorted array.
	"""
	function getMinRatio(λs::Array, n_skip)
		sort!(λs, rev=true); ratios = λs[2:end] ./ λs[1:(end-1)]
		rIdx = first(filter(x -> x > n_skip, sortperm(ratios)))
		return rIdx, λs
	end


	"""
		rand_block_krylov(A, r, ϵ; V0=randn(size(A)[2], r)) -> (K, λs)

	Run the randomized Block Krylov method from Musco & Musco's 2015 paper
	to compute ``r`` extremal eigenvalues and their eigenvectors.
	Return:
	- `K`: the resulting subspace
	- `λs`: the associated eigenvalues
	"""
	function rand_block_krylov(A, r::Int, ϵ::Real; V0=randn(size(A)[2], r))
		d = size(A)[1]; p = size(V0)[2]
		# take care of smaller size - append missing vectors
		# and orthogonalize them against given subspace.
		if p < r
			G = randn(d, r - p)
			V0 = hcat(V0, _qrDirect(G - V0 * (V0' * G)))
		end
        M = A * V0; q = trunc(Int, log2(d / sqrt(ϵ))) + 1
		K = fill(0.0, d, q * r)
		for i = 1:q
			K[:, ((i-1) * r + 1) : (i * r)] = M[:]
			M[:] = A * M  # update (A)^q Π
		end
		K = _qrDirect(copy(K)); _, Q, D = schur(Symmetric(K' * (A * K)))
		# sort D and pick top `r` corresponding eigenvectors
		inds = sortperm(D, rev=true)[1:r]; K[:] = K * Q
		return K[:, inds], D[inds]
	end


	"""
		bkvals(A, r::Int, V0, ϵ::Real; σ=0.0, Vp::Union{Array, Nothing}=nothing) -> (λ, niter)

	Run the LOBPCG method with initial guess `V0` to compute `r`
	extremal eigenvalues. Optionally, apply a shift `σ` and orthogonalize
	against a given subspace `Vp`. Return:
	- `λ`: the set of computed eigenvalues
	- `niter`: number of iterations required to converge to tolerance ϵ
	"""
	function bkvals(A, r::Int, V0, ϵ::Real; σ=0.0,
					Vp::Union{Array, Nothing}=nothing, getVecs=false)
		Vnew, λ, niter = block_krylov(A + σ * I, r, V0, ϵ, Vp=Vp)
		if getVecs
			return Vnew, λ, niter
		else
			return λ, niter
		end
	end


    """
		bkvals(A, r::Int, ϵ::Real; σ=0.0, Vp::Union{Array, Nothing}=nothing) -> (λ, niter)

    Run the LOBPCG method with random Gaussian initial guess to compute
    `r` extremal eigenvalues. Optionally, apply a shift `σ` and orthogonalize
	against a given subspace `Vp`. Return:
    - `λ`: the set of computed eigenvalues
    - `niter`: number of iterations required to converge to tolerance `ϵ`.
    """
	function bkvals(A, r::Int, ϵ::Real; σ=0.0, Vp::Union{Array, Nothing}=nothing,
					getVecs=false)
		return bkvals(A, r, randn(size(A)[1], r), ϵ, σ=σ, getVecs=getVecs, Vp=Vp)
    end


    """
        sivals(A, ℓ::Int, ϵ::Real; nconv=ℓ, V0=nothing, σ=0.0, maxiter=size(A)[1]) -> (λ, niter)

    Compute the `nconv` largest eigenvalues of `A` up to tolerance `ϵ`, using
    a block size `ℓ`. Optionally, start from an estimate `V0` and apply a
    shift `σ`, as well as orthogonalize against a given subspace `Vp`. Return:
    - `λ`: the set of computed eigenvalues
    - `niter`: the number of iterations required to converge to tolerance `ϵ`.
    """
    function sivals(A, ℓ::Int, ϵ::Real; nconv=ℓ, V0=nothing, σ=0.0,
					maxiter=size(A)[1], Vp::Union{Array, Nothing}=nothing,
					getVecs=false)
        Vnew, λ, niter = subspace_iteration(A + σ * I, maxiter, ϵ, nconv=nconv,
											V0=V0, Vp=Vp)
		if getVecs
			return Vnew, λ, niter
		else
        	return λ, niter
		end
    end


	"""
		inverse_iteration(A, r, iters; V0=nothing, σ=0.1)

	Perform inverse iteration on the matrix ``A + \\sigma I`` for `iters`
	iterations starting from a subspace estimate ``V_0``.
	"""
	function inverse_iteration(A, r, iters; V0=nothing, σ=0.1,
							   maxiter=nothing, ϵ=nothing)
		n = size(A)[1]; Aug = A + σ * I
		Vnext = (V0 == nothing) ? _qrDirect(randn(n, r)) : copy(V0)
		for i = 1:iters
			@simd for j = 1:r
				Vnext[:, j] = conjGrad(Aug, Vnext[:, j], maxiter=maxiter, ϵ=ϵ)
			end
			Vnext[:] = _qrDirect(Vnext)  # use conjugate gradient
		end
		return Vnext, eigvals(Vnext' * A * Vnext)
	end


	"""
		getIterBounds(Δ, ϵ, λr, λr₊) -> (ubBK, ubSI)

	Given an estimated initial distance `Δ`, an accuracy level `ϵ` and the
	eigenvalues `λr, λr₊`, the ratio of which controls convergence,
	returns the corresponding upper bounds for warm-started block
	krylov and subspace iteration methods, `ubBK` and `ubSI`.
	"""
	function getIterBounds(Δ, ϵ, λr, λr₊, n=1)
		(Δ <= 0) || (Δ >= 1) && return n, n
		Δ0 = Δ / sqrt(1 - Δ^2); ρ = λr / λr₊
		predBK = trunc(Int, (1 / sqrt(ρ - 1)) * (log(1 / ϵ) + log(Δ0))) + 1
		predSI = trunc(Int, (1 / log(ρ)) * (log(1 / ϵ) + log(Δ0))) + 1
		return real(predBK), real(predSI)
	end


	"""
		getBounds(Δ, λr, λr₊, ϵ, γ, n, V0, VHi, r)
	Given an estimated initial distance `Δ`, an accuracy level `ϵ`, the
	previous eigenvalues `λr, λr₊` as well as the norm of the perturbation
	`Enorm` and the decay factor `γ`, returns the corresponding upper bounds
	for warm-started block krylov and subspace iteration methods,
	`ubBK` and `ubSI`. If the Davis-Kahan bound is not applicable, return
	the naive bound `n`, corresponding to the size of the involved matrices.
	"""
	function getBounds(Δ, E, λr, λr₊, ϵ, γ, n, V0, VHi, r)
		(Δ == 0) && return 0, 0, 0, 0  # nothing to do if zero distance
		Vall = hcat(V0, VHi)
		eNorm_r = norm(E * Vall[:, r]); eNorm_r₊ = norm(E * Vall[:, r+1])
		# if Δ not suitable, return naive bound
		(Δ > 0.999) && return trunc(Int, n/2), trunc(Int, n/2), eNorm_r, eNorm_r₊
		ρ = max((λr - sqrt(2) * eNorm_r) / (λr₊ + sqrt(2) * eNorm_r),
			   1 + γ)
		Δ0 = Δ / sqrt(1 - Δ^2)
		predBK = trunc(Int, ceil((1 / sqrt(ρ - 1)) * (log(1 / ϵ) + log(Δ0))))
		predSI = trunc(Int, ceil((1 / log(ρ)) * (log(1 / ϵ) + log(Δ0))))
		return real(predBK), real(predSI), eNorm_r, eNorm_r₊
	end



	"""
		getIterBounds(Δ, ϵ, λr, λr₊, Enorm, γ, n) -> (ubBK, ubSI)

	Given an estimated initial distance `Δ`, an accuracy level `ϵ`, the
	previous eigenvalues `λr, λr₊` as well as the norm of the perturbation
	`Enorm` and the decay factor `γ`, returns the corresponding upper bounds
	for warm-started block krylov and subspace iteration methods,
	`ubBK` and `ubSI`. If the Davis-Kahan bound is not applicable, return
	the naive bound `n`, corresponding to the size of the involved matrices.
	"""
	function getIterBounds(Δ, ϵ, λr, λr₊, Enorm, γ, n; r=nothing, δ=nothing)
		(Δ == 0) && return n, n
		# determine the convergence rate factor
		if λr > Enorm
			ρ = (λr - Enorm) / (λr₊ + Enorm)
			ρ = (ρ ≤ 1) ? 1 + γ : ρ
		else
			ρ = 1 + γ
		end
		# if Δ not suitable, return naive bound from Gauss init
		if (Δ > 0.999)
			if (r == nothing)
				return n, n
			else
				C = log(_compCp(n, r, r, δ) / ϵ)
				return trunc(Int, C / sqrt(ρ)), trunc(Int, C / log(ρ))
			end
		end
		Δ0 = Δ / sqrt(1 - Δ^2)
		predBK = trunc(Int, (1 / sqrt(ρ - 1)) * (log(1 / ϵ) + log(Δ0))) + 1
		predSI = trunc(Int, (1 / log(ρ)) * (log(1 / ϵ) + log(Δ0))) + 1
		return real(predBK), real(predSI)
	end


	"""
		davKahProxy(V0, E, ϵ, λ1, λr, λr₊)

	Compute the proxy for the Davis-Kahan perturbation distance given the
	former subspace estimate `V0`, the perturbation `E`, the previous estimate
	accuracy `ϵ`, and the set of (estimated) eigenvalues `λ1, λr, λr₊`.
	"""
	function davKahProxy(V0, E, ϵ, λ1, λr, λr₊; which=:LM)
		eNorm = first(svds(E, nsv=1)[1].S)
		evNorm = first(svds(E * V0, nsv=1)[1].S)
		numer = 2 * sqrt(ϵ * eNorm^2 + evNorm^2)
		denom = (λr - λr₊ - (3 * λ1 * (ϵ^2)))
		return (numer / denom)
	end


	"""
		davKahProxy(V0, E, eNorm, ϵ, λ1, λr, λr₊; which=:LM)

	Compute the proxy for the Davis-Kahan perturbation distance given the
	former subspace estimate `V0`, the perturbation `E` and its spectral
	norm `Enorm`, the previous estimate accuracy `ϵ` and the set of (estimated)
	eigenvalues `λ1, λr, λr₊`.
	"""
	function davKahProxy(V0, E, eNorm, ϵ, λ1, λr, λr₊)
		evNorm = first(svds(E * V0, nsv=1)[1].S)
		numer = 2 * sqrt(ϵ * eNorm^2 + evNorm^2)
		denom = λr - λr₊ - (3 * λ1 * (ϵ^2))
		return (numer / denom)
	end


	function _compCp(n, ℓ, r, δ)
		p = ℓ - r  # "oversampling" factor
		Cp = ((exp(1) * sqrt(ℓ)) / (p + 1)) * (2 / δ)^(1 / (p + 1)) * (
			sqrt(n - ℓ + p) + sqrt(ℓ) + sqrt(2 * log(2 / δ)))
		return Cp
	end


end
