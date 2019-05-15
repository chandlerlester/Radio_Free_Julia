#==============================================================================

  Lifcycle Model based on code from Ben Moll's website:
      www.princeton.edu/~moll/HACTproject.htm

==============================================================================#

using LinearAlgebra, SparseArrays, Plots

# First set up model parameters
γ = 2 #CRRA utility parameter
σ_sq = (0.8)^2 # for the processes for productivity
corr = exp(-.9) # persistence for productivity
ρ = 0.05 # discount rate
r = 0.035 #interest rate
w =1 #wages

zmean = exp(σ_sq/2) # mean productivity
θ = -log(corr) # parameter for OU process for productivity

J =15 # number of points in gridspace for productivity
zmin = 0.75
zmax = 2.5

H = 300 #number of points in gridspace for assets
amin = 0
amax=100

T = 75 # maximum age of agents
N = 300 # number of age steps
dt = T/N

maxit =1000 #max number of iterations for HJB
crit = 10^(-10) # convergence criterion for HJB
convergence_criterion=10^(-5) #criterion for the KF and HJB?

# Set up some key variables
a = LinRange(amin,amax, H)
da = (amax-amin)/(H-1)
aa = a*ones(1,J)

z = LinRange(zmin,zmax,J)
dz = (zmax-zmin)/(J-1)
dz_sq = dz^2
zz=z*ones(1,H)

μ = -θ.*z.*log.(z)+σ_sq/2*z # drift for z
Σ_sq = σ_sq.*z.^2 #variance from Ito's lemma

# Create the matrix that summarizes the evolution of our system in the z dimension
 yy = (-Σ_sq/dz_sq - μ/dz)
 χ = Σ_sq/(2*dz_sq)
 ζ = μ/dz + Σ_sq/(2*dz_sq)

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

Aswitch =  spdiagm(0=>centerdiag_z[:], -H=>lowdiag_z[:], H=>updiag_z[H+1:end-H])

# Set terminal condition, i.e. value of death ≈ 0
small_number=10^(-8)
v_terminal = small_number*(small_number.+aa).^(1-γ)/(1-γ)

global V=v_terminal

# Set up spaces for variables from the finite difference loop
Vaf, Vab, c = [zeros(H,J) for i in 1:3]

v = zeros(H,J,N)
gg =[]
A_t=[]
c_t=[]
ss_t=[]

#================ Main loop =================================================#


for n in N:-1:1
	global V=V
    v[:,:,n]=V
    #forward difference
    Vaf[1:H-1,:] = (V[2:H,:]-V[1:H-1,:])/da
    Vaf[H,:] = (w*z .+ r.*amax).^(-γ) #will never be used, but impose state constraint a<=amax just in case
    #backward difference
    Vab[2:H,:] = (V[2:H,:]-V[1:H-1,:])/da
    Vab[1,:] = (w*z .+ r.*amin).^(-γ) #state constraint boundary condition

    #I_concave = Vab .> Vaf #indicator whether value function is concave (problems arise if this is not the case)

    # consumption and savings with forward difference
    cf = Vaf.^(-1/γ)
    ssf = zz' + r.*aa - cf
    # consumption and savings with backward difference
    cb = Vab.^(-1/γ)
    ssb = zz' + r.*aa - cb
    # consumption and derivative of value function at steady state
    c0 = zz' + r.*aa
    V0 = c0.^(-γ)

    #= dV_upwind makes a choice of forward or backward differences based on
     the sign of the drift    =#
    If = ssf .> 0 #positive drift → forward difference
    Ib = ssb .< 0 #negative drift → backward difference
    I0 = (1.0 .-If-Ib) #at steady state

    dV_Upwind = Vaf.*If + Vab.*Ib + V0.*I0  #important to include third term
    c = dV_Upwind.^(-1/γ)
    u = c.^(1-γ)/(1-γ)

    # CONSTRUCT MATRIX
    X = - min.(ssb,0)/da
    Y = - max.(ssf,0)/da + min.(ssb,0)/da
    Z = max.(ssf,0)/da

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
	A = AA + Aswitch

    push!(A_t,A)
    B = (1/dt + ρ)*sparse(I, H*J, H*J) - A

    u_stacked = reshape(u, H*J,1)
    V_stacked = reshape(V, H*J,1)

    b = u_stacked + V_stacked/dt
    V_stacked = B\b #SOLVE SYSTEM OF EQUATIONS

    global V = reshape(V_stacked, H,J)

    push!(ss_t,zz'+ r.*aa -c)
	push!(c_t, c)
end

# Plots

#inital consumption
plot(a, c_t[end], legend=false)
#inital savings
plot(a, ss_t[end], legend=false)
plot!(a, zeros(1,H))

# Create plots from lifecycle.pdf
dt_inv = convert(Int64, 1/dt)
dt_inv40 = convert(Int64, 40/dt)
dt_inv70 = convert(Int64, 70/dt)

# Policy function for consumption

p1=plot(a, c_t[300-dt_inv][:,1], label="Age 1, Lowest Income", ylims=(0,5), xlims=(0,80))
plot!(a, c_t[300-dt_inv][:,J], label="Age 1, Highest Income", xlabel="Wealth")
plot!(a, c_t[300-dt_inv40][:,1], label="Age 40 Lowest Income", ylabel="Consumption" )
plot!(a, c_t[300-dt_inv40][:,J], label="Age 40 Highest Income", legend=:bottomright)
plot!(a, c_t[300-dt_inv70][:,1], label="Age 70 Lowest Income")
plot!(a, c_t[300-dt_inv70][:,J], label="Age 70 Highest Income")

# Policy function for savings
p2= plot(a, ss_t[300-dt_inv][:,1], label="Age 1, Lowest Income", ylims=(-5,2), xlims=(0,100))
plot!(a, ss_t[300-dt_inv][:,J], label="Age 1, Highest Income", xlabel="Wealth")
plot!(a, ss_t[300-dt_inv40][:,1], label="Age 40 Lowest Income", ylabel="Savings" )
plot!(a, ss_t[300-dt_inv40][:,J], label="Age 40 Highest Income", legend=:bottomright)
plot!(a, ss_t[300-dt_inv70][:,1], label="Age 70 Lowest Income" )
plot!(a, ss_t[300-dt_inv70][:,J], label="Age 70 Highest Income")
plot!(a, zeros(H,1), line=:dash, color=:black, label="")

plot(p1,p2,legend=:bottomright,layout = (1,2))
