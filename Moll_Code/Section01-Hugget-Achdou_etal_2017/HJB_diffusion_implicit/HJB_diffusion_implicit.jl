#==============================================================================
    Code for solving the Hamiltonian Jacboi Bellman for
	   an Hugget model with a Diffusion process

	Translated Julia code from Matlab code by Ben Moll:
        http://www.princeton.edu/~moll/HACTproject.htm

        Updated to Julia 1.0.0
==============================================================================#

using Distributions, Plots, SparseArrays, LinearAlgebra


γ= 2.0 #gamma parameter for CRRA utility
r = 0.03 #interest rate
ρ = 0.05 #the discount rate


# Z our state variable follows this process
#= for this process:
 		dlog(z) = -θ⋅log(z)dt + σ^2⋅dw
	and
		log(z)∼N(0,var) where var = σ^2/(2⋅θ) =#
var = 0.07
μ_z = exp(var/2)
corr = 0.9
θ = -log(corr)
σ_sq = 2*θ*var


# create the grid for a
H = 100 #number of points on grid
amin = -0.02# min value
amax = 4 # max value
a = LinRange(amin, amax, H)
da = (amax-amin)/(H-1)

# create the grid for z
J = 40
z_min = μ_z*0.8
z_max = μ_z*1.2
z = LinRange(z_min, z_max, J)
dz = (z_max-z_min)/(J-1)
dz_sq = dz^2


# Check the pdf to make sure our grid isn't cutting off the tails of
	# our distribution
y = pdf.(LogNormal(0, var), z)
plot(z,y, grid=false,
		xlabel="z", ylabel="Probability",
		legend=false, color="purple", title="PDF of z")
png("PDF_of_z")

#create matrices for k and z
z= convert(Array, z)'
aa = a*ones(1,J)
zz = ones(H,1)*z

# use Ito's lemma to find the drift and variance of our optimization equation

μ = (-θ*log.(z).+σ_sq/2).*z # the drift from Ito's lemma
Σ_sq = σ_sq.*z.^2 #the variance from Ito's lemma

max_it = 100
ε = 10^(-6)
Δ = 1000

# set up all of these empty matrices
Vaf, Vab, Vzf, Vzb, Vzz = [zeros(H,J) for i in 1:6]

#==============================================================================

    Now we are going to construct a matrix summarizing the evolution of V_z

    This comes from the following discretized Bellman equation:

    ρv_ij = u(c_ij) + v_k(zF(k_{i}) -δk-c) + v_z⋅μ(z)
                                   + 1/2(v_{zz})σ^2(z)

                                   or

    ρv_ij = u(c_ij) + v_k(zF(k_{i}) -δk-c) + ((v_{i,j+1}-v_{i,j})/Δz)μ(z)
                            + 1/2((v_{i,j+1}-2v_{i,j}+v_{ij-1})/Δz^2)σ^2(z)

    Assume forward difference because of boundary conditions

==============================================================================#

 yy = (-Σ_sq/dz_sq - μ/dz)
 χ = Σ_sq/(2*dz_sq)
 ζ = μ/dz + Σ_sq/(2*dz_sq)


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

# spdiags in Matlab allows for automatic trimming/adding of zeros
    # spdiagm does not do this

B_switch = sparse(Diagonal(centerdiag_z))+ [zeros(H,H*J);  sparse(Diagonal(lowdiag_z)) zeros(H*(J-1), H)]+ sparse(Diagonal(updiag_z))[(H+1):end,1:(H*J)] # trim off rows of zeros


# Now it's time to solve the model, first put in a guess for the value function
v0 = (zz + r.*aa).^(1-γ)/(1-γ)/ρ
v=v0

maxit= 30 #set number of iterations (only need 6 to converge)
dist = [] # set up empty array for the convergence criteria

for n = 1:maxit
    V=v

    #Now set up the forward difference

    Vaf[1:H-1,:] = (V[2:H, :] - V[1:H-1,:])/da
    Vaf[H,:] = (z.+r.*amax).^(-γ) # imposes a constraint

    #backward difference
    Vab[2:H,:] = (V[2:H, :] - V[1:H-1,:])/da
    Vab[1,:] = (z.+r.*amin).^(-γ)

    #I_concave = Vab .> Vaf # indicator for whether the value function is concave

    # Consumption and savings functions
    cf = Vaf.^(-1/γ)
    sf = zz + r.*aa - cf

    # consumption and saving backwards difference

    cb = Vab.^(-1/γ)
    sb = zz + r.*aa - cb
    #println(sb)
    #consumption and derivative of the value function at the steady state

    c0 = zz + r.*aa
    Va0 = c0.^(-γ)

    # df chooses between the forward or backward difference

    If = sf.>0 # positive drift will ⇒ forward difference
    Ib = sb.<0 # negative drift ⇒ backward difference
    I0=(1.0.-If-Ib) # at steady state

    Va_upwind = Vaf.*If + Vab.*Ib + Va0.*I0 # need to include SS term

    global c = Va_upwind.^(-1/γ)
    u = (c.^(1-γ))/(1-γ)

    # Now to constuct the A matrix
    X = -min.(sb, zeros(H,1))/da
    #println(X)
    Y = -max.(sf, zeros(H,1))/da + min.(sb, zeros(H,1))/da
    Z = max.(sf, zeros(H,1))/da

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

  A = AA + B_switch
  B = (1/Δ + ρ)*sparse(I, H*J, H*J) - A

  u_stacked= reshape(u, H*J, 1)
  V_stacked = reshape(V,H*J, 1)

  b = u_stacked + (V_stacked./Δ)

  V_stacked = B\b

  V = reshape(V_stacked, H, J)

  V_change = V-v

  global v= V

  # need push function to add to an already existing array
  push!(dist, findmax(abs.(V_change))[1])
  if dist[n].< ε
      println("Value Function Converged Iteration=")
      println(n)
      break
  end

end

# calculate the savings for kk
ss = zz + r.*aa - c

# Plot the savings vs. k
plot(a, ss, grid=false,
		xlabel="a", ylabel="s(a,z)",
        xlims=(amin,amax),
		legend=false, title="Optimal Savings Policies")
plot!(a, zeros(H,1))

png("OptimalSavings")
