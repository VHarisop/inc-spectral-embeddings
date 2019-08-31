#!/usr/bin/env julia

module EvolvingNetworks

	using Arpack
	using Distributions
	using LinearAlgebra
	using LinearMaps
	using LightGraphs
	using Random
	using SparseArrays

	include("IOUtils.jl")
	include("Utils.jl")


	pickRand(eCollection) = begin
		rand(collect(eCollection))
	end


	"""
		pickNnzIdx(A::SparseMatrixCSC, num::Int)

	Collect `num` random nonzero indices from a sparse matrix `A`. Returns
	an array of tuples `(i, j)` containing nonzero indices.
	"""
	pickNnzIdx(A::SparseMatrixCSC, num::Int) = begin
		ePick = []; numNnz = length(A.nzval)
        idxSet = unique(rand(1:numNnz, num))  # keep unique indices
		rowIdx, colIdx = findnz(A)  # row and column indices of nonzero elts
		return collect(zip(rowIdx[idxSet], colIdx[idxSet]))
	end


	"""
        remRandEdges(g, ratio) -> (g, eRem)

	Remove a fixed `ratio` of randomly selected edges from `g`.
    Return:
    - `g`: the modified graph
    - `eRem`: an array containing the removed edges as `(from, to)`
      tuples
	"""
	function remRandEdges(g, ratio)
		nRem = trunc(Int, ratio * g.ne)
		eRem = pickNnzIdx(triu(adjacency_matrix(g)), nRem)
		for e in eRem
			rem_edge!(g, e[1], e[2])
		end
		return g, eRem
	end


	"""
		norm_laplacian(g)

	Return the normalized laplacian matrix of a graph g.
	"""
	function norm_laplacian(g)
		D = Diagonal(degree(g)); A = adjacency_matrix(g)
		return (1.0I - D^(-1/2) * A * D^(-1/2))
	end


	"""
		norm_adjacency(g)

	Return the normalized adjacency matrix of a graph g.
	"""
	function norm_adjacency(g)
		D = Diagonal(degree(g)); L = laplacian_matrix(g)
		return 1.0I - D^(-1/2) * L * D^(-1/2)
	end


	# optimized multiplication with regularized adjacency matrix
	_regMul(A, Dinv, τ, N, X) = begin
		V = Dinv .* X
		return Dinv .* (A * V) + (τ / N) * (Dinv .* sum(V))
	end


	"""
		reg_adjacency(g, τ=nothing)

	Return the regularized adjacency matrix of the graph `g` with
	regularization parameter `τ`; if the latter is unspecified, it is set
	to the average degree of the graph.
	Returns a `LinearMap` object implementing efficient matrix-vector
	multiplication.
	"""
	function reg_adjacency(g, τ=nothing)
		τ = (τ == nothing) ? mean(degree(g)) : τ
		# regularized diagonal
		Dinv = 1 ./ sqrt.(degree(g) .+ τ)
		A = adjacency_matrix(g); n = size(A)[1]
		return LinearMap(X -> _regMul(A, Dinv, τ, n, X), n, n, issymmetric=true)
	end


	#= _remEdge(gSrc, gDst) : remove an edge from gSrc which was shared
	   between gSrc and gDst =#
	function _remEdge(gSrc, gDst)
		cmnE = edges(intersect(gSrc, gDst))
		return (length(cmnE) == 0) ? false : rem_edge!(gSrc, pickRand(cmnE))
	end


	#= _addEdge(gSrc, gDst) : add an edge from gDst to gSrc, if such an
	   edge exists =#
	function _addEdge(gSrc, gDst)
		diffE = edges(difference(gDst, gSrc))
		return (length(diffE) == 0) ? false : add_edge!(gSrc, pickRand(diffE))
	end


	#= _addNonEdge(gSrc, gDst) : add an edge to gSrc which does not exist
	   in gDst =#
	function _addNonEdge(gSrc, gDst)
		isDir = is_directed(gSrc); exclude = edges(gDst)
		n = nv(gDst); nE = ne(gDst) * (if is_directed(gDst) 1 else 2 end)
		nMiss = n * (n - 1) - nE
		probs = ((n - 1) .- (length.(gDst.fadjlist))) ./ (nMiss)
		while true
			sIdx = Utils.pick_rand_idx(probs)
			diffE = setdiff(1:n, gDst.fadjlist[sIdx], sIdx)
			if length(diffE) == 0
				return false
			end
			tIdx = rand(diffE)
			if isDir
				sIdx, tIdx = sort([sIdx, tIdx])
			end
			if !((sIdx, tIdx) in exclude)
				return add_edge!(gSrc, sIdx, tIdx)
			end
		end
	end


	#= _remNonEdge(gSrc, gDst) : remove an edge from gSrc which does not
	   exist in gDst =#
	function _remNonEdge(gSrc, gDst)
		diffE = edges(difference(gSrc, gDst))
		return (length(diffE) == 0) ? false : rem_edge!(gSrc, pickRand(diffE))
	end


	"""
		evolveGraph(gSrc, gDst; p=0.9, edit=:add)

	Evolve a graph `gSrc` towards `gDst` with probability ``p`` using the
	specified edit type (one of {:add, :remove}). With probability ``1-p``,
	perform an edit that increases the edit distance of the two graphs.
	Returns ``true`` or ``false`` depending on whether an edit was performed.
	"""
	function evolveGraph(gSrc, gDst; p=0.9, edit=:add)
		if rand() <= p
			if edit == :add
				return _addEdge(gSrc, gDst)
			else
				return _remNonEdge(gSrc, gDst)
			end
		else
			if edit == :add
				return _addNonEdge(gSrc, gDst)
			else
				return _remEdge(gSrc, gDst)
			end
		end
	end


	"""
		evolveGraph_rndEdit(gSrc, gDst; p=0.9, p_add=0.5)

	Evolve a graph `gSrc` towards `gDst` with probability ``p`` using an edit
	which is addition with probability `p_add`, and deletion otherwise. With
	probability ``1 - p``, perform an edit that increases the edit distance of
	the two graphs.
	Returns ``true`` or ``false`` depending on whether an edit was performed.
	"""
	function evolveGraph_rndEdit(gSrc, gDst; p=0.9, p_add=0.5)
		edit = (rand() <= p_add) ? (:add) : (:remove)
		return evolveGraph(gSrc, gDst, p=p, edit=edit)
	end


	"""
		genSBM(n, comms; p=0.9, q=0.1, balanced=false)

	Generate an instance of a stochastic block model with ``n`` nodes,
	`comms` communities, and inter / intra community edge probabilities
	``p, q`` respectively. If `balanced == true`, will generate equal
	number of nodes per community (default: false).
	"""
	function genSBM(n, comms; p=0.9, q=0.1, balanced=false)
		if balanced
			perCom = repeat([div(n, comms)], comms)
		else
			perCom = rand(Multinomial(n, comms))
		end
		return genSBM(perCom, p=p, q=q)
	end


	"""
		genSBM(popVec; p=0.9, q=0.1)

	Generate an instance of a stochastic block model with population per
	community given by `popVec`, with inter/intra community probabilities
	`p, q` respectively.
	"""
	function genSBM(popVec; p=0.9, q=0.1)
		pMat = q * ones(length(popVec), length(popVec))
		fill!(view(pMat, diagind(pMat)), p)  # fill diagonal with p
		return stochastic_block_model(pMat .* popVec, popVec)
	end


	"""
		genCollegeMsgGraphs(fname::String; connected=true)

	Read the College Message dataset from the specified file `fname` and return
	a pair of graphs; the former corresponds to the initial state of the graph
	without any edges, and the latter corresponds to the full graph. If
	`connected` is true, returns the largest weakly connected component
	of the graph.
	"""
	function genCollegeMsgGraphs(fname::String; connected=true)
		df = IOUtils.readCollegeMsg(fname);
		n = maximum(union(df.from, df.to))
		gSrc = Graph(n); gTgt = Graph(n)
		for (i, j) in zip(df.from, df.to)
			# add all except self-loops
			(i ≠ j) && add_edge!(gTgt, i, j)
		end
		(connected) && begin
			gComp = connected_components(gTgt)[1]
			return gSrc[gComp], gTgt[gComp]
		end
		return gSrc, gTgt
	end


	"""
		genCollegeMsgGraphs(fname::String, ratio::Float64; connected=true)

	Given a parameter `ratio`, returns a pair of graphs whose edge sets have
	sizes with ratio given by `ratio`. These edges correspond to the 'first'
	edges in the temporal version of the graph.
	"""
	function genCollegeMsgGraphs(fname::String, ratio::Float64; connected=true)
		df = IOUtils.readCollegeMsg(fname); vidAll = union(df.from, df.to)
		# number of vertices
		n = maximum(vidAll)
		gSrc, gTgt = Graph(n), Graph(n)
		for (i, j) in zip(df.from, df.to)
			# add everything except self-loops
			(i ≠ j) && add_edge!(gTgt, i, j)
		end
		(connected) && begin
			gComp = connected_components(gTgt)[1];
		end
		nE = gTgt[gComp].ne; nAdd = trunc(Int, ratio * nE); eDone = 0; eAdd = []
		for (i, j) in zip(df.from, df.to)
			(i == j) && continue  # skip if self-loop
			if (i in gComp) && (j in gComp)
				if (eDone ≤ nAdd)
					has_edge(gTgt, i, j) && (add_edge!(gSrc, i, j) && (eDone += 1))
				else
					push!(eAdd, (i, j))
				end
			end
		end
        filter!(x -> !has_edge(gSrc, x[1], x[2]), eAdd)
        return gSrc, gTgt, eAdd
	end


	"""
		genRedditReplyGraphs(fname::String, numUsers::Int; connected=true)

	Read the Temporal Reddit Reply dataset from the specified file `fname`
	and return a graph corresponding to the largest weakly connected component
	of the dataset, if `connected` is set. Only include users whose IDs fall
	in the range `{1, ..., numUsers}`.
	"""
	function genRedditReplyGraphs(fname::String, numUsers::Int;
								  connected=true)
		df = IOUtils.readRedditReply(fname, numUsers)
		n = length(unique(union(df.from, df.to)))
		gSrc = Graph(n); gTgt = Graph(n)
		for (i, j) in zip(df.from, df.to)
			# add all except self-loops
			(i ≠ j) && add_edge!(gTgt, i, j)
		end
		(connected) && begin
			gComp = connected_components(gTgt)[1]
			return gSrc[gComp], gTgt[gComp]
		end
		return gSrc, gTgt
	end


	"""
		genRedditReplyGraphs(fname::String, numUsers::Int, ratio::Float64;
							 connected=true)

	Read the Temporal Reddit Reply dataset from the specified file `fname`
	and return a graph corresponding to the largest weakly connected component
	of the dataset, if `connected` is set. Only include users whose IDs fall
	in the range `{1, ..., numUsers}`.
	"""
	function genRedditReplyGraphs(fname::String, numUsers::Int, ratio::Float64;
								  connected=true)
		df = IOUtils.readRedditReply(fname, numUsers)
		n = length(unique(union(df.from, df.to)))
		gSrc = Graph(n); gTgt = Graph(n)
		for (i, j) in zip(df.from, df.to)
			# add all except self-loops
			(i ≠ j) && add_edge!(gTgt, i, j)
		end
		(connected) && begin
			gComp = connected_components(gTgt)[1];
		end
		nE = gTgt[gComp].ne; nAdd = trunc(Int, ratio * nE); eDone = 0; eAdd = []
		for (i, j) in zip(df.from, df.to)
			(i == j) && continue  # skip if self-loop
			if (i in gComp) && (j in gComp)  # if in connected component
				if (eDone ≤ nAdd)
					if has_edge(gTgt, i, j)
						add_edge!(gSrc, i, j) && (eDone += 1)
					end
				else
					has_edge(gTgt, i, j) && push!(eAdd, (i, j))
				end
			end
		end
		return gSrc, gTgt, eAdd
	end


end
