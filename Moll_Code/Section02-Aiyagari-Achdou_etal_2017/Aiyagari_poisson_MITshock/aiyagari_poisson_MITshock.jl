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

T =200 #timing setup for shock
N = 400
dt = T/N
time = (0:N-1)*dt
max_price_it = 300
con_crit=10^(-5)
relax =0.1

# Construct the TFP sequence
corr =0.8
ν=1-corr
A_prod_t = zeros(N,1)
A_prod_t[1] = .97*A_prod

for n =1:N-1
	A_prod_t[n+1] = dt*ν*(A_prod-A_prod_t[n]) + A_prod_t[n]
end

plot(time, A_prod_t, xlims=(0,40))

# Now to set up the other grid spaces
H = 1000 # Number of elements in grid space for a
a_min = -0.8
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
critS = 10^(-5)
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
gg_s=[]

# Construct the matrix to summarize the evolution of z over time
# this matrix will not change over the finite differences and is based on the poisson process

Aswitch = [-sparse(I,H,H)*λ[1] sparse(I,H,H)*λ[2] ; sparse(I,H,H)*λ[1] -sparse(I,H,H)*λ[2]]


# Prices
r =0.04 #initial r
w = 0.05

r0=0.03
global r_min=0.01
global r_max=.99*ρ

dist = []
v0=zeros(H,2)

v0[:,1] = (w*z[1].+r.*a).^(1-γ)/(1-γ)/ρ
v0[:,2] = (w*z[2].+r.*a).^(1-γ)/(1-γ)/ρ

# The main loop, this makes sure markets clear

for ir in 1:Ir
	global r =r
	global r_min = r_min
	global r_max = r_max

	println("Main Loop iteration $(ir)")

	push!(r_r, r)
	push!(r_min_r, r_min)
	push!(r_max_r, r_max )

	push!(KD, (α*A_prod/(r+δ))^(1/(1-α))*z_ave)
	global w = (1-α)*A_prod*KD[ir].^α * z_ave^(-α)

	if ir >1
		global v0 = V_r[:,:,ir-1]
	end

	global v=v0

# Inner loop, value function iteration

	for n in 1:maxit
		global V=v
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

		global V = [V_stacked[1:H] V_stacked[H+1:2*H]]

		V_change = V-v

		global v = V

		push!(dist,findmax(abs.(V_change))[1])

		if n>1 && dist[n] < crit
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
	g_sum = gg'*ones(2*H,1).*da
	gg = gg./g_sum

	global g = [gg[1:H] gg[H+1:2*H]]

	push!(gg_s,gg)
	push!(g_r, g)
	push!(adot, w*zz+r.*aa-c)
	V_r[:,:,ir] = V

	push!(KS, g[:,1]'*a*da + g[:,2]'*a*da)
	push!(S, KS[ir]-KD[ir])

	# Update aggregate capital
	if S[ir]>critS
		println("Excess Supply")
		global r_max =r
		global r = 0.5*(r+r_min)
	elseif S[ir] < -critS
		println("Excess Demand")
		global r_min =r
		global r = 0.5*(r+r_max)
	elseif abs(S[ir]) < critS
		println("Equilibrium Found!!!")
		break
	end
end

# Save some objects
v_st = v
gg_st = gg_s[end]
K_st = KS[end]
w_st =w
r_st =r
C_st = gg_s[end]'*reshape(c,H*2,1).*da
A_prod_st = A_prod

# Transition Dynamics
gg0 = gg_st

# Set up locations for preallocting values
gg_br =[]
c_t=[]
A_t=[]
C_t=[]
K_t, K_out, r_t, w_t = [zeros(N,1) for i in 1:4]
dist_it=[]

K_t = K_st*ones(N,1)

for it in 1:max_price_it
	println("Price Iteration")
	println(it)

	global w_t = (1-α)*A_prod_t.*K_t.^α .*z_ave^(-α)
	global r_t = (α*A_prod_t.*K_t.^(α-1) .*z_ave^(1-α)) .- δ

	global V=v_st

	for n in N:-1:1
		# Forward difference
		Vaf[1:H-1,:] = (V[2:H,:]-V[1:H-1,:])/da
		Vaf[H,:] = (w_t[n]*z .+ r_t[n].*a_max).^(-γ)

		# Backward difference
		Vab[2:H,:] = (V[2:H,:]-V[1:H-1,:])/da
		Vab[1,:] = (w_t[n]*z .+ r_t[n].*a_min).^(-γ)

		# caculate consumption and savings
		# First the forward difference case
		cf = Vaf.^(-1/γ)
		sf = w_t[n]*zz + r_t[n].*aa -cf

		# consumption and savings for the backward difference
		cb = Vab.^(-1/γ)
		sb = w_t[n]*zz + r_t[n].*aa -cb

		#Consumption and V' at the steady state
		c0 = w_t[n]*zz + r_t[n].*aa
		Va0 = c0.^(-γ)

		# Setup to the Upwind scheme
		If = sf .> 0
		Ib = sb .< 0
		I0 = (1 .- If .- Ib)

		#Now implement the upwind scheme
		Va_Upwind = Vaf.*If + Vab.*Ib + Va0.*I0

		global c= Va_Upwind.^(-1/γ)
		u = c.^(1-γ)/(1-γ)
		push!(c_t,c)

		X = -min.(sb,0)/da
		Y = -max.(sf,0)/da + min.(sb,0)/da
		Z = max.(sf,0)/da

		A1 = spdiagm(0=>Y[:,1], -1 => X[2:H,1], 1=> Z[1:H-1,1])
		A2 = spdiagm(0=>Y[:,2], -1 => X[2:H,2], 1=> Z[1:H-1,2])
		AA = [A1 spzeros(H,H); spzeros(H,H) A2]

		global A= AA + Aswitch

		push!(A_t,A)

		B = (1/dt + ρ)*sparse(I,2*H,2*H)-A

		u_stacked = [u[:,1]; u[:,2]]
		V_stacked = [V[:,1]; V[:,2]]

		b = u_stacked + V_stacked/dt

		V_stacked = B\b

		global V = [V_stacked[1:H] V_stacked[H+1:2*H]]
	end
	# Now update the distribution
	push!(gg_br,gg0)
	for n = 1:N
		AT = A_t[end-(n-1)]'
		push!(gg_br,(sparse(I,2*H,2*H)-AT*dt)\gg_br[n])
		global K_out[n] = (gg_br[n][1:H])'*a*da + gg_br[n][H+1:2*H]'*a*da
		push!(C_t,gg_br[n]'*reshape(c_t[n],2*H,1).*da)
	end

	push!(dist_it, findmax(abs.(K_out-K_t))[1])

	global K_t = relax.*K_out + (1-relax).*K_t

	if dist_it[it]<con_crit
		println("Equilibrium Found")
		break
	end

	global gg_br=[]
end

# Inequality measurements

Wealth_neg_t,Wealth_Gini_t,CapInc_Gini_t,Inc_Gini_t,top_inc_t,top_wealth_t,neg_frac_t = [zeros(1,400) for i in 1:7]

for n in 1:N
	Wealth_neg_t[n] = gg_br[n][1:H]'*min.(a,0).*da + gg_br[n][H+1:2*H]'*min.(a,0).*da
	g_a_count = gg_br[n][1:H] + gg_br[n][H+1:2*H]

	# Discrete Gini for checking
	g_a = g_a_count*da
	S_a = cumsum(g_a.*a,dims=1)/sum(g_a.*a)
	trapez_a = .5 .* (S_a[1]*g_a[1]+sum((S_a[2:H]+S_a[1:H-1]).*g_a[2:H]))
	Wealth_Gini_t[n] = 1 .-2*trapez_a


	# Gini of Capital income
	yk = r_t[n].*a
	g_yk=g_a

	S_yk = cumsum(g_yk.*yk,dims=1)/sum(g_yk.*yk)
	trapez_yk = .5 .* (S_yk[1]*g_yk[1]+sum((S_yk[2:H]+S_yk[1:H-1]).*g_yk[2:H]))
	CapInc_Gini_t[n] = 1 -2*trapez_yk

	# Gini of total income
	y = w_t[n]*zz + r_t[n]*aa
	Ny = 2*H
	yy =reshape(y,Ny,1)
	index = sortperm(yy[:])
	yy=yy[index]
	g_y = gg_br[n][index].*da

	S_y = cumsum(g_y.*yy,dims=1)/sum(g_y.*yy)
	trapez_y = .5 .* (S_y[1]*g_y[1]+sum((S_y[1:H-1]+S_y[1:H-1]).*g_y[2:H]))
	Inc_Gini_t[n] = 1 -2*trapez_y

	G_y =cumsum(g_y,dims=2)
	G_a = cumsum(g_a,dims=2)

	#Top 10% income share
	p1=0.1
	obj, index = findmin(abs.((1 .-G_y) .-p1))
	top_inc_t[n]=1 .-S_y[index]

	#Top 10% Wealth share
	obj, index = findmin(abs.((1 .-G_a) .-p1))
	top_wealth_t[n]=1 .-S_a[index]

	neg_frac_t[n]=findmax(G_a[a.<=0])[1]


end

p1 = plot(time, A_prod_t,xlims=(0,100), title="Aggregate Prod.",legend=false)
plot!(time,A_prod_st*ones(N,1),line=:dash,color=:black)

p2 = plot(time, K_t,xlims=(0,100),ylims=(0.293,0.3),title="Aggregate Capital St.",legend=false )
plot!(p2,time,K_st.*ones(N,1),line=:dash,color=:black)

p3 = plot(time,w_t, xlims=(0,100),title="Wage",legend=false)
plot!(p3,time,w_st.*ones(N,1),line=:dash,color=:black)

p4=plot(time,r_t,xlims=(0,100),title="Interest Rate",legend=false)
plot!(p4,time,r_st.*ones(N,1),line=:dash,color=:black)

p5 =plot(time,Wealth_Gini_t',xlims=(0,100),title="Wealth Gini",legend=false)
plot!(p5,time,Wealth_Gini_t[N].*ones(N,1),line=:dash,color=:black)

p6 =plot(time,Inc_Gini_t',xlims=(0,100),title="Income Gini",legend=false)
plot!(p6,time,Inc_Gini_t[N].*ones(N,1),line=:dash,color=:black)

plot(p1,p2,p3,p4,p5,p6,layout=(2,3),legend=false)
png("Inequality_Measures")
