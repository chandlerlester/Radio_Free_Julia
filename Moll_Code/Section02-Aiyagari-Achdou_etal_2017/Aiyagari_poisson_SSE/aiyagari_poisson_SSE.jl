#========================================================================

	Steady state solution for an Aiyagari model with a diffusion process

	Translated julia code based on code from Ben Moll's Website:
		http://www.princeton.edu/%7Emoll/HACTproject.htm

		For Julia 1.0.0
========================================================================#
using LinearAlgebra, SparseArrays, Plots 

# Model parameters
γ = 2 # CRRA utility
α = 1/3 # parameter for cobb-douglas production function
δ = 0.05 # captal depreciation
ρ = 0.05 # discount rate
A_prod =0.1 # term in production function

# Parameters for productivity and the poisson process q
z_1 = 1
z_2 = 2*z_1
z = [z_1 z_2]
λ_1 = 1/3 # prob of transitioning from state 1 to 2
λ_2 = 1/3  # state 2 to 1
λ = [λ_1 λ_2]
z_ave = (z_1*λ_2 + z_2*λ_1)/(λ_1 + λ_2) # Average value of z


H = 1000 # Number of elements in grid space for a
a_min = 0.0
a_max = 20.0

# Grid spaces for variables
a = LinRange(a_min, a_max, H)
a = convert(Array,a)
da = (a_max-a_min)/(H-1)

aa = [a a]
zz = ones(H,1)*z #check later

# Paramters for the simulation
maxit = 100
Ir = 40
crit = 10^(-6)
critS = 1e-5
Δ = 1000 #step size for HJB

# Spaces for the finite difference terms
Vaf, Vab, Vzf, Vzb, Vzz, c= [zeros(H,2) for i in 1:6]
V_r = zeros(H,2,H)

# Spaces for other terms
r_r =[]
r_min_r =[]
r_max_r =[]
KD=[]
KS=[]
S=[]
adot=[]
V_n=[]
g_r=[]

# Construct the matrix to summarize the evolution of z over time
# this matrix will not change over the finite differences and is based on the poisson process

Aswitch = [-sparse(I,H,H)*λ[1] sparse(I,H,H)*λ[2] ; sparse(I,H,H)*λ[1] -sparse(I,H,H)*λ[2]]


# Prices
r =0.04 #initial r
global w = 0.05

r0=0.03
global r_min=0.01
global r_max=.99*ρ

dist = zeros(1,maxit)
v0=zeros(H,2)

# The main loop, this makes sure markets clear

for ir in 1:Ir
	global r =r
	global r_min = r_min
	global r_max = r_max
	global v0 = v0

	println("Main Loop iteration $(ir)")

	push!(r_r, r)
	push!(r_min_r, r_min)
	push!(r_max_r, r_max )

	push!(KD, (α*A_prod/(r+δ))^(1/(1-α))*z_ave)
	w = (1-α)*A_prod*KD[ir].^α * z_ave^(-α)

	v0[:,1] = (w*z[1] .+ r.*a).^(1-γ)/(1-γ)/ρ
	v0[:,2] = (w*z[2] .+ r.*a).^(1-γ)/(1-γ)/ρ

	if ir >1
		v0 = V_r[:,:,ir-1]
	end

	global v=v0

# Inner loop, value function iteration

	for n in 1:maxit
		V=v
		push!(V_n, V)
		# Forward difference
		Vaf[1:H-1,:] = (V[2:H,:]-V[1:H-1,:])/da
		Vaf[H,:] = (w*z .+ r.*a_max).^(-γ)

		# Backward difference
		Vab[2:H,:] = (V[2:H,:]-V[1:H-1,:])/da
		Vab[1,:] = (w*z .+ r.*a_min).^(-γ)

		# caculate consumption and savings
		# First the forward difference case
		cf = Vaf.^(-1/γ)
		sf = w*zz + r.*aa -cf

		# consumption and savings for the backward difference
		cb = Vab.^(-1/γ)
		sb = w*zz + r.*aa -cb

		#Consumption and V' at the steady state
		c0 = w*zz + r.*aa
		Va0 = c0.^(-γ)

		# Setup to the Upwind scheme
		If = sf .> 0
		Ib = sb .< 0
		I0 = (1 .- If - Ib)

		#Now implement the upwind scheme
		Va_Upwind = Vaf.*If + Vab.*Ib + Va0.*I0

		c= Va_Upwind.^(-1/γ)
		u = c.^(1-γ)/(1-γ)

		X = -min.(sb,0)/da
		Y = -max.(sf,0)/da + min.(sb,0)/da
		Z = max.(sf,0)/da

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

		v = V

		dist[n] = findmax(V_change)[1]

		if dist[n] < crit
			println("Value Function Converge Iteration = $(n)")
			break
		end
	end
	# Now solve the Fokker-Planck equation
	AT = A'
	b = zeros(2*H,1)

	# Dirty fix to avoid matrix singularities
	i_fix=1
	b[i_fix] =.1
	row = [zeros(1,i_fix-1) 1 zeros(1,H*2-i_fix)]
	AT[i_fix,:] = row

	# Now solve the linear system
	gg = AT\b
	g_sum = gg'*ones(2*H,1)*da
	gg = gg./g_sum

	global g = [gg[1:H] gg[H+1:2*H]]

	push!(g_r, g)
	push!(adot, w*zz+r.*aa-c)
	V_r[:,:,ir] = V

	push!(KS, g[:,1]'*a*da + g[:,2]'*a*da)
	push!(S, KS[ir]-KD[ir])

	# Update aggregate capital
	if S[ir]>critS
		println("Excess Supply")
		r_max =r
		r = 0.5*(r+r_min)
	elseif S[ir] < -critS
		println("Excess Demand")
		r_min =r
		r = 0.5*(r+r_max)
	elseif abs(S[ir]) < critS
		println("Equilibrium Found!!!")
		break
	end
end

# Graphs
