module IOUtils

	using CSV
	using DataFrames
	using Dates
	using LightGraphs
	using LinearAlgebra
	using Random
	using SparseArrays
	using Statistics
	using ToeplitzMatrices


	"""
		readCollegeMsg(name::String)

	Read the college message dataset stored in file `name`. Return
	a DataFrame containing all the data with extra columns containing
	month and day information.
	"""
	function readCollegeMsg(name="datasets/college-msg.txt")
		# read .CSV file containing temporal links
		df = CSV.read(name, delim=" "); names!(df, [:from, :to, :time])
		# make day/month columns and sort according to them
		df.day = map.(x -> day(unix2datetime(x)), df.time)
		df.month = map.(x -> month(unix2datetime(x)), df.time)
        select!(df, Not(:time))  # delete time column
		return sort(df, (:month, :day))
	end


	"""
		readRedditReply(name="datasets/temporal-reddit-reply.txt", num_users::Int)

	Read the Temporal Reddit Reply dataset, including only users that fall in
	the range `{1, ..., num_users}`. Return a DataFrame containing the data.
	"""
	function readRedditReply(name="datasets/temporal-reddit-reply.txt",
							 num_users=1000)
		f = CSV.File(name, delim=" ", header=[:from, :to, :time])
		table = (from=Int[], to=Int[], time=[])
		for row in f
			if (row.from <= num_users) && (row.to <= num_users)
				push!(table.from, row.from)
				push!(table.to, row.to)
				push!(table.time, row.time)
			end
		end
		return DataFrame(table)
	end


	"""
		readPowerData(name="datasets/household_power.txt";
					  period=15, step=(period ÷ 2))

	Read the household power consumption dataset, returning the vector of power
	and the vector of power differences. `period` contains the desired period
	for computing the moving averages of power consumption (default: `15`),
	while `step` contains the step by which to move forward through the time
	series (default: `period ÷ 2`).
	"""
	function readPower(name="datasets/household_power.txt";
					   period=15, step=(period ÷ 2))
		df = CSV.read(name, delim=";", copycols=true)
		# isolate power component
		data = df.Global_active_power
		# compute moving average over 15 minutes
		movingAvg(x::Array, period::Int) = begin
			avgs = Float64[]; cSum = sum(x[1:period]); cIdx = 1
			while true
				push!(avgs, cSum / period)
				(cIdx + period > length(x)) && break
				cSum += x[cIdx + period] - x[cIdx]; cIdx += step
			end
			return avgs
		end
		mAvg = movingAvg(data, period); mAvgDiffs = diff(mAvg)
		return mAvg[2:end], mAvgDiffs
	end


	"""
		getLaggedCovMatrix(tS, n, sIdx, wLen)

	Generate a lagged covariance matrix of size `(n - wLen + 1) x wLen` given
	a time series `tS`, starting at index `sIdx`. Returns a `Toeplitz` matrix
	which admits fast matrix-vector multiplication. If `sIdx + n ≥ length(tS)`
	raises an `ErrorException`.
	"""
	function getLaggedCovMatrix(tS, n, sIdx, wLen)
		if (sIdx + n) ≥ length(tS)
			throw(ErrorException("Insufficient data remaining."))
		end
		vc = tS[sIdx:(sIdx+wLen-1)]; vr = tS[(sIdx+wLen-1):(sIdx+n-1)]
		return Hankel((1 / sqrt(n)) * Hankel(vc, vr))
	end


end
