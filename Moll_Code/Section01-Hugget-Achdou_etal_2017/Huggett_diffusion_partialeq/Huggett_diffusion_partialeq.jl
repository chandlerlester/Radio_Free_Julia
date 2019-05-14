#==============================================================================

    Partial equilibrium with a diffusion process.

               Translated from matlab code on Ben Moll's website:
                              http://www.princeton.edu/~moll/HACTproject.htm

==============================================================================#

using LinearAlgebra, SparseArrays, Plots

γ = 2 # parameter from CRRA utility
r = 0.04 # the interest rate, exogenous in this version
ρ = 0.05 # the discount rate

# Ornstein-Uhlenbeck process in levels
σ_sq = 0.05
corr= 0.9
θ = -log(corr)
Var = σ_sq/(2*θ)
zmean = 1


H = 100 # number of points in the grid space for a
amin=-0.1
amax= 30
a = LinRange(amin,amax,H)
da = (amax-amin)/(H-1)

J =40 #number of points in grid space for z
zmin =0.4
zmax = 1.5
z=LinRange(zmin,zmax,J)
dz = (zmax-zmin)/(J-1)
dz_sq = dz^2

μ = θ*(zmean.-z) # Drift
s2 = σ_sq.*ones(J,1) # Variance

aa = a*ones(1,J)
zz = ones(H,1)*z'

maxit = 20 # the maximum number of iterations we allow the finite differencing algorithm
crit = 10^(-6) # our critical value
Δ = 1000

dVf, dVb, c = [zeros(H,J) for i in 1:3]

# Create the matrix that summarizes the evolution of z
yy = min.(μ,0)/dz - max.(μ,0)/dz - s2/dz_sq
χ = -min.(μ,0)/dz + s2/(2*dz_sq)
ζ = max.(μ,0)/dz + s2/(2*dz_sq)


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

lowdiag_z = repeat([χ[2]], H, 1)
	for j=3:J
		global lowdiag_z = [lowdiag_z; repeat([χ[j]],H,1)]
	end
lowdiag_z=lowdiag_z[:]

# spdiags in Matlab allows for automatic trimming/adding of zeros
    # spdiagm does not do this

Bswitch =  spdiagm(0=>centerdiag_z[:], -H=>lowdiag_z[:], H=>updiag_z[H+1:end-H])
# Inital Guess
v0= (zz + r.*aa).^(1-γ)/(1-γ)/ρ

global v=v0

dist=[]
V_n=[]

# The finite differeing loop

for n in 1:maxit
    V=v
    push!(V_n,V)
    # Forward differencing
    dVf[1:H-1,:]=(V[2:H,:]-V[1:H-1,:])/da
    dVf[H,:] = (z .+ r.*amax).^(-γ) # impose state constraint at the max, just in case
    # Backward differencing
    dVb[2:H,:]=(V[2:H,:]-V[1:H-1,:])/da
    dVb[1,:]= (z .+r.*amin).^(-γ)

    # Find consumption and savings with the forward difference
    cf = dVf.^(-1/γ)
    sf = zz +r.*aa -cf

    # Find consumption and savings with the backward difference
    cb= dVb.^(-1/γ)
    sb = zz +r.*aa -cb

    # Find consumption and savings at steady state
    c0 = zz + r.*aa
    dV0 = c0.^(-γ)

    # Now implement the upwind scheme in order to select the best differencing method
    If = sf .> 0
    Ib = sb .< 0
    I0 = (1 .- If .- Ib)

    # State constraint at amin is automatically implemented

    dV_Upwind = dVf.*If + dVb.*Ib + dV0.*I0

    global c = dV_Upwind.^(-1/γ)
    u = c.^(1-γ)/(1-γ)
    # Construct matrix for the evolution of the system
    X = -min.(sb,0)/da
    Y= -max.(sf,0)/da + min.(sb,0)/da
    Z = max.(sf,0)/da

	# Define the Diagonals of this transition matrix
	updiag = 0
       for j = 1:J
           updiag =[updiag; Z[1:H-1,j]; 0]
       end
    updiag =(updiag[:])

    centerdiag=reshape(Y, H*J, 1)
    centerdiag = (centerdiag[:]) # for tuples

   lowdiag = X[2:H, 1]
       for j = 2:J
           lowdiag = [lowdiag; 0; X[2:H,j]]
       end
   lowdiag=(lowdiag)

   # spdiags in Matlab allows for automatic trimming/adding of zeros
       # spdiagm does not do this
  	AA = sparse(Diagonal(centerdiag))+ [zeros(1, H*J); sparse(Diagonal(lowdiag)) zeros(H*J-1,1)] + sparse(Diagonal(updiag))[2:end, 1:(H*J)] # trim first element


    global A = AA + Bswitch

    B = (ρ +1/Δ).*sparse(I,H*J,H*J) -A
    u_stacked = reshape(u,H*J,1)
    V_stacked = reshape(V,H*J,1)

    b = u_stacked + V_stacked/Δ
    V_stacked = B\b # Solves the system of equations

    V = reshape(V_stacked,H,J)

    V_change = V - v

    global v = V

    push!(dist, findmax(abs.(V_change))[1])

    if dist[n] < crit
        println("Value Function converged, Iteration=")
        println(n)
        break
    end
end

#====================================================
        Fokker-Planck Equation
====================================================#

AT = A' # we need the transpose of our A matrix
b =zeros(H*J,1)

# We need to fix one value to prevent the matrix from being singular
i_fix =1
b[i_fix] = .1
row = [zeros(1,i_fix-1) 1 zeros(1,J*H-i_fix)]
AT[i_fix,:] = row

# Using this solve the linear system
gg = AT\b
g_sum = gg'*ones(H*J,1)*da*dz
gg = gg./g_sum # we need gg to sum to one

g = reshape(gg,H,J)

# Graphs
ss = zz + r.*aa -c

icut = 90
acut=a[1:icut]
sscut=ss[1:icut,:]

surface(acut,z,sscut', alpha=0.9,camera=(45,35),
        colorbar=false,xlabel="Wealth", ylabel="Productivity",
		title="Savings Over Wealth and Productivity", size=[800,480])
png("OptimalSavings")

icut2 = 30
acut2=a[1:icut2]
gcut = g[1:icut2,:]

surface(acut2,z,gcut',alpha=.9,camera=(45,35),title="\$ \\textrm{Density } g(a,z)\$",
	xlabel="Wealth", ylabel="Productivity", colorbar=:false,
	xlims=(amin, findmax(acut2)[1]), ylims=(zmin,zmax),size=[800,480])
#plot!(acut2,z,gcut',α=.2,seriestype=:wireframe)
png("KFE_pdf")
