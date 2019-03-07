#========================================================================

	Steady state solution for an Aiyagari model with a diffusion process

	Translated julia code based on code from Ben Moll's Website: 
		http://www.princeton.edu/%7Emoll/HACTproject.htm


========================================================================#
using LinearAlgebra, SparseArrays, PyPlot

# Model parameters
γ = 2 # CRRA utility
α = 0.35 # parameter for cobb-douglas production function
δ = 0.1 # captal depreciation
z_mean = 1.0 # mean of levels of diffusion process
σ_sq = (.10)^2 # also for diffusion process
corr = exp(-0.3)
ρ = 0.05

K = 3.8 #inital guess of aggregate capital, need this to be close for algorithm to converge
relax = 0.99 # a relaxation parameter
J=40

z_min = 0.5
z_max = 1.5
a_min = -1.0
a_max = 30.0
H =100

# Paramters for the simulation
maxit = 100
maxitK = 100
crit = 10^(-6)
critK = 1e-5
Δ = 1000 #step size for HJB

# Diffusion process in levels
θ = -log(corr)
Var = σ_sq/(2*θ)

# Grid spaces for variables
a = LinRange(a_min, a_max, H)
a = convert(Array,a)
da = (a_max-a_min)/(H-1)

z = LinRange(z_min, z_max, J)
z = convert(Array,z)
dz = (z_max-z_min)/(J-1)
dz_sq = dz^2

aa = a*ones(1,J)
zz = ones(H,1)*z' #check later

# Other variables
μ = θ*(z_mean.-z) #drift for HJB via Ito's lemma
s_sq = σ_sq.*ones(1,J)

# Spaces for the finite difference terms
Vaf, Vab, Vzf, Vzb, Vzz, c = [zeros(H,J) for i in 1:6]

# Construct the matrix to summarize the evolution of z over time
	# this matrix will not change over the finite differences

yy = -(s_sq/dz_sq)' - μ/dz
χ = s_sq/(2*dz_sq)
ζ = μ/dz + (s_sq/(2*dz_sq))'

# Define the Diagonals of this matrix
updiag_z = zeros(H,1)
   for j = 1:J
	   global updiag_z =[updiag_z; repeat([ζ[j]], H, 1)]
   end
updiag_z =(updiag_z[:])


centerdiag_z=repeat([χ[1]+yy[1]],H,1)
   for j = 2:J-1
	   global centerdiag_z = [centerdiag_z; repeat([yy[j]], H, 1)]
   end
centerdiag_z=[centerdiag_z; repeat([yy[J]+ζ[J]], H, 1)]
centerdiag_z = centerdiag_z[:]

lowdiag_z = repeat([χ[2]], H, 1)
   for j=3:J
	   global lowdiag_z = [lowdiag_z; repeat([χ[j]],H,1)]
   end
lowdiag_z=lowdiag_z[:]

Aswitch = sparse(Diagonal(centerdiag_z))+ [zeros(H,H*J);  sparse(Diagonal(lowdiag_z)) zeros(H*(J-1), H)]+ sparse(Diagonal(updiag_z))[(H+1):end,1:(H*J)]

# Inital Guess for the prices
r = α * K^(α-1)-δ
w = (1-α) * K^(α)
v0 = (w*zz + r.* aa).^(1-γ)/(1-γ)/ρ
global v = v0
dist = zeros(1,maxit)

# The main loop, this makes sure markets clear

for iter in 1:maxitK
	println("Main Loop iteration $(iter)")

	for n in 1:maxit
		V=v

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

		updiag = 0
		   for j = 1:J
			   updiag =[updiag; Z[1:H-1,j];0]
		   end
		updiag =(updiag[:])

		centerdiag=reshape(Y,H*J,1)

		lowdiag = X[2:H,1]
		   for j=2:J
			   lowdiag = [lowdiag; 0; X[2:H,j]]
		   end
		lowdiag=lowdiag[:]

		AA = sparse(Diagonal(centerdiag[:]))+ [zeros(1, H*J); sparse(Diagonal(lowdiag)) zeros(H*J-1,1)] + sparse(Diagonal(updiag))[2:end, 1:(H*J)]

		global A= AA + Aswitch
		B = (1/Δ + ρ)*sparse(I,H*J,H*J)-A

		u_stacked = reshape(u, H*J, 1)
		V_stacked = reshape(V,H*J, 1)

		b = u_stacked + V_stacked/Δ

		V_stacked = B\b

		V = reshape(V_stacked, H,J)
		V_change = V-v
		global v = V

		dist[n] = findmax(V_change)[1]

		if dist[n] < crit
			println("Value Function Converge Iteration = $(n)")
			break
		end
	end
	# Now solve the Fokker-Planck equation
	AT = A'
	b = zeros(H*J,1)

	# Dirty fix to avoid matrix singularities
	i_fix=1
	b[i_fix] =.1
	row = [zeros(1,i_fix-1) 1 zeros(1,H*J-i_fix)]
	AT[i_fix,:] = row

	# Now solve the linear system
	gg = AT\b
	g_sum = gg'*ones(H*J,1)*da*dz
	gg = gg./g_sum

	global g = reshape(gg,H,J)

	# Update aggregate capital
	S = sum(g'*a*da*dz)
	println(S)

	if abs(K-S)<critK
		break
	end

	#update prices
	global K =relax*K + (1-relax)*S
	global r = α * K^(α-1) - δ
	global w = (1-α) * K^(α)
end

# Graphs

ss = w*zz + r.*aa - c
icut = 50
acut=a[1:icut]
sscut=ss[1:icut,:]
gcut = g[1:icut,:]


surface(acut,z,sscut', α=0.7,camera=(45,25),
        colorbar=false, zrotation=90,
        xlabel="Wealth", ylabel="Density")
png("OptimalSavings")

surface(acut,z,gcut', α=0.7,camera=(45,25))
png("KFE_pdf")
