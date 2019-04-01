#========================================================================

	Steady state solution for an Aiyagari model with a diffusion process

	Translated julia code based on code from Ben Moll's Website:
		http://www.princeton.edu/moll/HACTproject.htm

		For Julia 1.0.0
========================================================================#
using LinearAlgebra, SparseArrays, Plots

# Model parameters
γ = 2 # CRRA utility
α = 1/3 # parameter for cobb-douglas production function
δ = 0.05 # captal depreciation
ρ = 0.05 # discount rate
A_prod =0.3 # term in production function

# Parameters for productivity and the poisson process q
z_1 = .2
z_2 = 2*z_1
z = [z_1 z_2]
λ_1 = 1 # prob of transitioning from state 1 to 2
λ_2 = 1  # state 2 to 1
λ = [λ_1 λ_2]
z_ave = (z_1*λ_2 + z_2*λ_1)/(λ_1 + λ_2) # Average value of z


H = 500 # Number of elements in grid space for a
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
Ir = 100
crit = 10^(-6)
critS = 1e-5
Δ = 1000 #step size for HJB

# Spaces for the finite difference terms
Vaf, Vab, c= [zeros(H,2) for i in 1:3]
V_r = zeros(H,2,H)

# Create r grid space
r_min = -0.0499
r_max = 0.049
r_grid = LinRange(r_min, r_max, Ir)

# Set up an initial guess
r=r_grid[1]
KD_1 = (α*A_prod/(r+δ))^(1/(1-α))*z_ave
w =(1-α)*A_prod*(KD_1/z_ave)^α

v0 = zeros(H,2)
v0[:,1]=(w*z[1] .+ max(r,0.01).*a).^(1-γ)/(1-γ)/ρ
v0[:,2]=(w*z[2] .+ max(r,0.01).*a).^(1-γ)/(1-γ)/ρ


# Spaces for other terms
w_r =[]
KD= zeros(1,Ir)
S=[]
adot=[]
V_n=[]
g_r=[]

# Construct the matrix to summarize the evolution of z over time
# this matrix will not change over the finite differences and is based on the poisson process

Aswitch = [-sparse(I,H,H)*λ[1] sparse(I,H,H)*λ[2] ; sparse(I,H,H)*λ[1] -sparse(I,H,H)*λ[2]]


# The main loop, this makes sure markets clear

for ir in 1:Ir
	global r =r_grid[ir]
	global v0 = v0

	println("Main Loop iteration $(ir)")

	KD[ir]=(α*A_prod/(r+δ))^(1/(1-α))*z_ave
	global w = (1-α)*A_prod*KD[ir].^α * z_ave^(-α)

	push!(w_r,w)

	if ir >1
		global v0 = V_r[:,:,ir-1]
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
		I0 = (1 .- If .- Ib)

		#Now implement the upwind scheme
		Va_Upwind = Vaf.*If + Vab.*Ib + Va0.*I0

		global c= Va_Upwind.^(-1/γ)
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

		V = [V_stacked[1:H] V_stacked[H+1:2*H]]

		V_change = V-v

		global v = V

		dist = findmax(V_change)[1]

		if n>1 && dist < crit
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
	V_r[:,:,ir] = v

	push!(S, g[:,1]'*a*da + g[:,2]'*a*da)
end

# Graphs
S_max = findmax(S)[1]
amin1=a_min-0.02
aaa=LinRange(amin1,S_max,Ir)
rrr=LinRange(-0.06,0.06,Ir)
KD_new = (α*A_prod./max.(rrr.+δ,0)).^(1/(1-α)).*z_ave

plot(S,r_grid, xlims=(amin1,0.6),ylims=(-0.06,0.06), color=:blue,
	label="\$S(r)\$", legend=:bottomright, xlabel="K",ylabel="r",grid=false)
plot!(KD_new,rrr, label="\$F_{k}(K,1)-\\delta\$")
plot!(zeros(Ir,1).+a_min,rrr,label="\$a=\\underline{a}\$",color=:yellow,line=:dash)
plot!(aaa,ones(Ir,1).*ρ,label="\$r=\\rho\$",color=:purple,line=:dash)
plot!(aaa,ones(Ir,1).*(-δ),label="\$r=-\\delta\$",color=:green, line=:dash)
png("Asset_supply")
