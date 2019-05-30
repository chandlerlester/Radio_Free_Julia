#========================================================================

	Steady state solution for housing problem with non-convexities

	Translated julia code based on code from Ben Moll's Website:
		http://www.princeton.edu/%7Emoll/HACTproject.htm

		For Julia 1.0.0
========================================================================#
using LinearAlgebra, SparseArrays, Plots

# Model parameters
γ = 2 # CRRA utility
r = 0.035 #return on asset
α = 1/3 # parameter for cobb-douglas production function
η = 0.2 # for our benefit function
p = 1 #price
ϕ = 2 # coefficient of a in the budget constraint
hmin = 2.3 #lower threshold for houses
δ = 0.05 # captal depreciation
ρ = 0.05 # discount rate

# Parameters for productivity and the poisson process q
z_1 = .1
z_2 = .135
z = [z_1 z_2]
λ_1 = .5 # prob of transitioning from state 1 to 2
λ_2 = .5  # state 2 to 1
λ = [λ_1 λ_2]
z_ave = (z_1*λ_2 + z_2*λ_1)/(λ_1 + λ_2) # Average value of z


H =500 # Number of elements in grid space for a
a_min = 0.0
a_max = 3.0

# Grid spaces for variables
a = LinRange(a_min, a_max, H)
a = convert(Array,a)
da = (a_max-a_min)/(H-1)

aa = [a a]
zz = ones(H,1)*z #check later

# Construct the matrix to summarize the evolution of z over time
# this matrix will not change over the finite differences and is based on the poisson process

Aswitch = [-sparse(I,H,H)*λ[1] sparse(I,H,H)*λ[2] ; sparse(I,H,H)*λ[1] -sparse(I,H,H)*λ[2]]

# Paramters for the simulation
maxit = 120
crit = 10^(-10)
Δ = 1000 #step size for HJB

# Spaces for the finite difference terms
Vaf, Vab, c= [zeros(H,2) for i in 1:3]

dist =[]
V_n=[]

h = min.((α*η/(r*p))^(1/(1-α)) + hmin, ϕ*aa/p)
h = h.*(h.>=hmin)
f = η*(max.(h.-hmin,0)).^α - r*p*h

v0 = (zz + r*aa).^(1-γ)/(1-γ)/ρ

global v = v0

# The main loop, this makes sure markets clear

for n in 1:maxit
	V=v
	push!(V_n, V)
	# Forward difference
	Vaf[1:H-1,:] = (V[2:H,:]-V[1:H-1,:])/da
	Vaf[H,:] = (z' .+ f[H,:] .+ r*a_max).^(-γ)

	# Backward difference
	Vab[2:H,:] = (V[2:H,:]-V[1:H-1,:])/da
	Vab[1,:] = (z' .+ f[1,:].+ r*a_min).^(-γ)

	# caculate consumption and savings
	# First the forward difference case
	cf = (max.(Vaf,10^(-10))).^(-1/γ)
	ssf = zz + f + r.*aa -cf
	Hf = cf.^(1-γ)/(1/γ) + Vaf.*ssf

	# consumption and savings for the backward difference
	cb = (max.(Vab,10^(-10))).^(-1/γ)
	ssb = zz + f + r.*aa -cb
	Hb = cb.^(1-γ)/(1-γ) + Vab.*ssb

	#Consumption and V' at the steady state
	c0 = zz + f + r.*aa
	Va0 = c0.^(-γ)

	# Upwind scheme chooses between the forward or backward difference
	Ieither = (1 .- (ssf.>0)) .* (1 .- (ssb.<0))
	Iunique = (ssb.<0).*(1 .- (ssf.>0)) + (1 .- (ssb.<0)).*(ssf.>0)
	Iboth = (ssb.<0).*(ssf.>0)
    If= Iunique.*(ssf .> 0) + Iboth.*(Hf.>=Hb)  #positive drift → forward difference
    Ib = Iunique.*(ssb .< 0) + Iboth.*(Hb.>=Hf) #negative drift → backward difference
    I0 = Ieither  #at steady state

	global c = cf.*If + cb.*Ib + c0.*I0
    u = (c.^(1-γ))/(1-γ)

	X = -min.(ssb,0)/da
	Y = -max.(ssf,0)/da + min.(ssb,0)/da
	Z = max.(ssf,0)/da

	A1 = spdiagm(0=>Y[:,1], -1 => X[2:H,1], 1=> Z[1:H-1,1])
	A2 = spdiagm(0=>Y[:,2], -1 => X[2:H,2], 1=> Z[1:H-1,2])
	AA = [A1 spzeros(H,H); spzeros(H,H) A2]

	global A= AA + Aswitch

	B = (1/Δ + ρ)*sparse(I,2*H,2*H)-A

	u_stacked = [u[:,1]; u[:,2]]
	V_stacked = [V[:,1]; V[:,2]]

	b = u_stacked + V_stacked/Δ

	V_stacked = B\b

	global V = [V_stacked[1:H] V_stacked[H+1:2*H]]

	V_change = V-v

	global v = V

	push!(dist, findmax(V_change)[1])
		if dist[n] < crit
			println("Value Function Converge Iteration = $(n)")
			break
		end
end

# Now solve the Fokker-Planck equation
AT = A'
gg =[] #create empty cell to store g

# Create an initial distribution
gg0 = ones(2*H,1)
g_sum = gg0'*ones(2*H,1)*da
gg0 = gg0./g_sum #normalize the initial distribution

push!(gg,gg0)
N = 1000
dt = 10
g_dist = zeros(N,1)

for n in 1:N
	push!(gg,(sparse(I,2*H,2*H)-AT*dt)\gg[n])
	g_dist[n] = findmax(abs.(gg[n+1]-gg[n]))[1]
end

g = [gg[N][1:H] gg[N][H+1:2*H]]
adot = zz + f + r.*aa - c
astar = p*hmin/ϕ

obj, index = findmin(abs.(astar.-a))

c2 = c - η*max.(h.-hmin,0).^α

amax1 = a_max
amin1 = -0.1

#Plot savings, output has some rough parts unlike matlab counter part...
plot(a, adot[:,1], label="\$s_{1}(a)\$", legend=:topright,
	xlims=(amin1,amax1),color=:blue,
	ylabel="Savings", xlabel="Wealth")
plot!(a,adot[:,2], label="\$s_{2}(a)\$", color=:red)
plot!(LinRange(amin1,amax1,H),zeros(H,1), label="",line=:dash, color=:black)
plot!(a_min.*ones(H,1), label="",line=:dash, color=:black, seriestype=:vline)
plot!(astar.*ones(H,1), label="",line=:dash, color=:black, seriestype=:vline)
png("savings_plot")

# Housing consumption, less smooth as well
plot(a, c2[:,1], label="\$c_{1}(a)\$",
	xlims=(amin1,amax1),color=:blue,
	ylabel="Housing Consumption", xlabel="Wealth", legend=:bottomright)
plot!(a,c2[:,2], label="\$c_{2}(a)\$", color=:red)
plot!(a_min.*ones(H,1), label="",line=:dash, color=:black, seriestype=:vline)
plot!(astar.*ones(H,1), label="",line=:dash, color=:black, seriestype=:vline)

# Housing
plot(a, h[:,1], label="\$h_{1}(a)\$",
	xlims=(amin1,amax1),color=:blue,
	ylabel="Housing", xlabel="Wealth", legend=:false)
plot!(a,h[:,2], label="\$h_{2}(a)\$", color=:red, line=:dashdot)
plot!(a_min.*ones(H,1), label="",line=:dash, color=:black, seriestype=:vline)
plot!(astar.*ones(H,1), label="",line=:dash, color=:black, seriestype=:vline)


# Benefit
plot(a, f[:,1],
	xlims=(amin1,amax1),color=:blue,
	ylabel="Pecuniary Benefit from Housing", xlabel="Wealth", legend=:false)
plot!(a,f[:,2], color=:red, line=:dashdot)
plot!(a_min.*ones(H,1), label="",line=:dash, color=:black, seriestype=:vline)
plot!(astar.*ones(H,1), label="",line=:dash, color=:black, seriestype=:vline)

# Value Function
plot(a, v[:,1], label="\$v_{1}(a)\$",
	xlims=(amin1,amax1),color=:blue,
	ylabel="Value Function", xlabel="Wealth", legend=:bottomright)
plot!(a,v[:,2], label="\$v_{2}(a)\$", color=:red, line=:dashdot)
plot!(a_min.*ones(H,1), label="",line=:dash, color=:black, seriestype=:vline)
plot!(astar.*ones(H,1), label="",line=:dash, color=:black, seriestype=:vline)

# Value Function
plot(a, g[:,1], label="\$g_{1}(a)\$",
	xlims=(amin1,amax1), ylim=(0,3.5),color=:blue,
	ylabel="Densities", xlabel="Wealth", legend=:bottomright)
plot!(a,g[:,2], label="\$g_{2}(a)\$", color=:red, line=:dashdot)
plot!(a_min.*ones(H,1), label="",line=:dash, color=:black, seriestype=:vline)
plot!(astar.*ones(H,1), label="",line=:dash, color=:black, seriestype=:vline)
