#===============================================================================

  Code for investment under uncertainty by Heterogenous firms

  Translated from matlab code from Ben Moll's website:
      www.princeton.edu/~moll/HACTproject.htm


================================================================================#

using LinearAlgebra, SparseArrays, Plots

# Set up model parameters
ρ = 0.05  #discount rate
α = 0.5 # Cobb-douglas production function
Θ = 2.7 # quadratic adjustment cost
δ = 0.025 # depreciation rate


# Parameters for the Ornstein-Uhlenbeck process for z
var = 0.026^2
zmean = exp(var/2)
corr = 0.859
θ = -log(corr)
σ_sq = 2*θ*var

# Set up the gridspace for z
J = 40
zmin = zmean*0.6
zmax=zmean*1.4
z = LinRange(zmin,zmax,J)
dz = (zmax-zmin)/(J-1)
dz_sq = dz^2

# Set up gridspace for capital
H =100
kmin=1
kmax = 100
k = LinRange(kmin,kmax, H)
dk = (kmax-kmin)/(H-1)

kk = k*ones(1,J)
zz = (z*ones(1,H))'

# Set up terms for drift and variance of the UO process that depend on zz
μ = (-θ*log.(z) .+ σ_sq/2).*z
Σ_sq = σ_sq*(z.^2)

# Production function
F = zz.*kk.^α

# Parameters for our loop
maxit = 20
crit=10^(-6)
Δ = 1000

# Empty spaces for saving things
Vkb, Vkf, c = [zeros(H,J) for i in 1:3]
dist =[]
x_all=[]

# Create the matrix that summarizes the evolution of our system in the z dimension
 yy = (-Σ_sq/dz_sq + min.(μ,0)/dz - max.(μ,0)/dz)
 χ = Σ_sq/(2*dz_sq)- min.(μ,0)/dz
 ζ = max.(μ,0)/dz + Σ_sq/(2*dz_sq)

 # Define the Diagonals of this matrix
 updiag_z = zeros(H,1)
 	for j = 1:J
		global updiag_z =[updiag_z; repeat([ζ[j]], H, 1)]
	end

 centerdiag_z=repeat([χ[1]+yy[1]],H,1)
	for j = 2:J-1
		global centerdiag_z = [centerdiag_z; repeat([yy[j]], H, 1)]
	end
 centerdiag_z=[centerdiag_z; repeat([yy[J]+ζ[J]], H, 1)]

lowdiag_z = repeat([χ[2]], H, 1)
	for j=3:J
		global lowdiag_z = [lowdiag_z; repeat([χ[j]],H,1)]
	end
# spdiags in Matlab allows for automatic trimming/adding of zeros
    # spdiagm does not do this

Bswitch =  spdiagm(0=>centerdiag_z[:], -H=>lowdiag_z[:], H=>updiag_z[H+1:end-H])

# Inital guess for v
v0 = (F-δ.*kk-0.5*Θ*δ^2 .*kk)/ρ
global v =v0


for n in 1:maxit
	global V=v
    #forward difference
    Vkf[1:H-1,:] = (V[2:H,:]-V[1:H-1,:])/dk
    Vkf[H,:] = (1+Θ*δ)*ones(1,J)#will never be used, but impose state constraint a<=amax just in case
    #backward difference
    Vkb[2:H,:] = (V[2:H,:]-V[1:H-1,:])/dk
    Vkb[1,:] = (1+Θ*δ)*ones(1,J) #state constraint boundary condition

    #I_concave = Vab .> Vaf #indicator whether value function is concave (problems arise if this is not the case)

    # investment and savings with forward difference
    xf = (Vkf.-1)/Θ.*kk
    ssf = xf-δ.*kk
	Hf = F -xf -0.5*Θ*(xf./kk).^2 .*kk + Vkf.*ssf
    # consumption and savings with backward difference
	xb = (Vkb.-1)/Θ.*kk
    ssb = xb-δ.*kk
	Hb = F -xb -0.5*Θ*(xb./kk).^2 .*kk + Vkb.*ssb
    # investment at steady state
    x0 = δ.*kk

    #= dV_upwind makes a choice of forward or backward differences based on
     the sign of the drift    =#
	Ieither = (1 .- (ssf.>0)) .* (1 .- (ssb.<0))
	Iunique = (ssb.<0).*(1 .- (ssf.>0)) + (1 .- (ssb.<0)).*(ssf.>0)
	Iboth = (ssb.<0).*(ssf.>0)
    If= Iunique.*(ssf .> 0) + Iboth.*(Hf.>=Hb)  #positive drift → forward difference
    Ib = Iunique.*(ssb .< 0) + Iboth.*(Hb.>=Hf) #negative drift → backward difference
    I0 = Ieither  #at steady state

	x = xf.*If + xb.*Ib + x0.*I0
    profits = F-x-0.5*Θ*(x./kk).^2. .*kk

    # CONSTRUCT MATRIX
    X = -ssb.*Ib/dk
    Y = - ssf.*If/dk  + ssb.*Ib/dk
    Z = ssf.*If/dk

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
	A = AA + Bswitch

    B = (1/Δ + ρ)*sparse(I, H*J, H*J) - A

    profits_stacked = reshape(profits, H*J,1)
    V_stacked = reshape(V, H*J,1)

    b = profits_stacked + V_stacked/Δ
    V_stacked = B\b #SOLVE SYSTEM OF EQUATIONS

    global V = reshape(V_stacked, H,J)

    Vchange = V-v
	global v = V

	push!(dist, findmax(abs.(Vchange))[1])
	push!(x_all,x)

	if dist[n]<crit
		println("Value function converged iteration = $(n)")
		break
	end
end

plot(k,v, legend=false)

plot(dist, legend=false)

kdot=x_all[end]-δ.*kk
plot(kk,kdot,label="")
plot!(kk,zeros(H,1), line=:dash, color=:black, label="")
