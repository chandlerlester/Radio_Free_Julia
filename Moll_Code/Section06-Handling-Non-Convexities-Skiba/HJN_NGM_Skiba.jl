#==============================================================================
    Code for solving the Hamiltonian Jacboi Bellman for
	   a problem with non-convexities as in Skiba 1978

	Translated Julia code from Matlab code by Ben Moll:
        http://www.princeton.edu/~moll/HACTproject.htm
==============================================================================#

using Plots, SparseArrays, LinearAlgebra


γ= 2.0 #gamma parameter for CRRA utility
ρ = 0.05 #the discount rate
α = 0.3 # the curvature of the production function (cobb-douglas)
δ = 0.05 # the depreciation rate
AH = 0.6 #high prodictivity, costs κ
AL = 0.4 #low prodictivity, free
κ = 2 #fixed cost

kssH = (α*AH/(ρ+δ))^(1/(1-α))+κ
k_st = κ./(1-(AL/AH).^(1/α))


# create the grid for k
H = 1000 #number of points on grid
k_min = 0.001*kssH # min value
k_max = 1.3*kssH # max value
k = LinRange(k_min, k_max, H)
dk = (k_max-k_min)/(H-1)

#Create the production function
yH = AH*max.(k.-κ,0).^α #high productivity
yL = AL*k.^α #low prodictivity
y = max.(yH,yL)

plot(k,y, ylabel="\$f(k)\$", xlabel="\$k\$",
		xlims=(k_min,k_max), ylims=(0,0.9),legend=false,color=:black)
plot!(k,yH, line=:dash, color=:red)
plot!(k,yL, line=:dashdot, color=:orange)

# use Ito's lemma to find the drift and variance of our optimization equation

maxit = 150 #needed to increase this for same results in julia
ε = 10^(-6)
Δ = 1000
# set up all of these empty matrices
dVf, dVb, dV0, c, If, Ib, I0, V= [zeros(H,1) for i in 1:8]

# Now it's time to solve the model, first put in a guess for the value function
v0 = (k.^α).^(1-γ)/(1-γ)/ρ
v=v0

dist = [] # set up empty array for the convergence criteria

for n = 1:maxit
    global V=v

    #Now set up the forward difference

    dVf[1:H-1] = (V[2:H] - V[1:H-1])/dk
    dVf[H] = (y[H]-δ.*k_max).^(-γ) # imposes a constraint

    #backward difference
    dVb[2:H] = (V[2:H] - V[1:H-1])/dk
    dVb[1] = (y[1]-δ.*k_min).^(-γ)

    #I_concave = Vab .> Vaf # indicator for whether the value function is concave

    # Consumption and savings functions
    cf = max.(dVf,10^(-10)).^(-1/γ)
	μf = y -δ.*k -cf
    Hf = cf.^(1-γ)/(1-γ) + dVf.*μf

    # consumption and saving backwards difference

	cb = max.(dVb,10^(-10)).^(-1/γ)
	μb = y -δ.*k -cb
    Hb = cb.^(1-γ)/(1-γ) + dVb.*μb

    #consumption and derivative of the value function at the steady state

    c0 = y- δ.*k
    dV0 = max.(c0, 10^(-10)).^(-γ)
	H0 = c0.^(1-γ)/(1-γ)

    # Upwind scheme chooses between the forward or backward difference
	Ieither = (1 .- (μf.>0)) .* (1 .- (μb.<0))
	Iunique = (μb.<0).*(1 .- (μf.>0)) + (1 .- (μb.<0)).*(μf.>0)
	Iboth = (μb.<0).*(μf.>0)
    If= Iunique.*(μf .> 0) + Iboth.*(Hf.>=Hb)  #positive drift → forward difference
    Ib = Iunique.*(μb .< 0) + Iboth.*(Hb.>=Hf) #negative drift → backward difference
    I0 = Ieither  #at steady state

    global c = cf.*If + cb.*Ib + c0.*I0
    u = (c.^(1-γ))/(1-γ)

	# CONSTRUCT MATRIX
    global X = -μb.*Ib/dk
    global Y = - μf.*If/dk  + μb.*Ib/dk
    global Z = μf.*If/dk

	global A = spdiagm(0=>Y[:], -1=>X[2:end], 1=>Z[1:end-1])


  	B = (1/Δ + ρ)*sparse(I, H, H) - A

  	u_stacked= reshape(u, H, 1)
  	V_stacked = reshape(V,H, 1)

  	b = u_stacked + (V_stacked/Δ)

  	V_stacked = B\b

  	global V = reshape(V_stacked, H, 1)

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
kdot = y -δ.*k -c
dV_upwind = dVf.*If + dVb.*Ib+dV0.*I0
Verr = c.^(1-γ)/(1-γ) + dV_upwind.*kdot - ρ.*V

#plot for consumption
plot(k, c, ylabel="\$c(k)\$", xlabel="\$k\$", label="Consumption")
plot!(k,y-δ.*k, label="Production net of depreciation", legend=:bottomright)

#plot for savings
plot(k, kdot, ylabel="\$s(k)\$", xlabel="\$k\$", legend=false) 
plot!(k,zeros(H,1), line=:dash)

#Plot the value function
plot(k, V,ylabel="\$V(k)\$", xlabel="\$k\$", legend=false)

#Plot the error in HJB equation
plot(k,Verr,ylabel="Value Function error", xlabel="\$k\$", legend=false)
