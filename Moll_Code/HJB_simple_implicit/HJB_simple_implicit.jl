#==============================================================================
    Code for solving the Hamiltonian Jacboi Bellman for
	   an basic model: ρv(a) = max_{c} u(c) + v'(a)[s(a)]
	   Where s(a) = w + ra - c(a)

	Translated Julia code from Matlab code by Ben Moll:
        http://www.princeton.edu/~moll/HACTproject.htm

        Updated to Julia 1.0.0
==============================================================================#

using Parameters, Distributions, Plots, SparseArrays, LinearAlgebra

σ= 2.0 #
ρ = 0.05 #the discount rate
r = 0.045 # the depreciation rate
w = 0.1

H = 500
a_min = -0.02
a_max = 1.0

a = LinRange(a_min, a_max, H)
a = convert(Array, a) # create grid for a values
da = (a_max-a_min)/(H-1)

maxit = 10000
ε = 10e-6

Δ = 1000

dVf, dVb= [zeros(H,1) for i =1:2]

#initial guess for V
v0 = (w.+r.*a).^(1-σ)/(1-σ)/ρ
v= v0

dist=[]

for n=1:maxit
	V=v
    #forward difference
    dVf[1:H-1] = (V[2:H]-V[1:H-1])/da;
    dVf[H]= 0;

	# backward difference
	dVb[2:H] = (V[2:H]-V[1:H-1])/da
	dVb[1] = (w.+r.*a_min).^(-σ) # the boundary condition

	I_concave = dVb .> dVf

    # consumption and savings with forward difference
    cf = dVf.^(-1/σ)
    ssf = w.+r.*a-cf

    # consumption and savings with backward difference
	cb = dVb.^(-1/σ)
	ssb = w.+r.*a-cb

    # consumption and savings at steady state
    c0=w.+r.*a
    dV0 =c0.^(-σ)

    #look at the sign of the drift
        #to choose forward or backward difference
    If = ssf .> 0 # positive drift ⇒ forward difference
    Ib = ssb .< 0  # negative drift ⇒ backward difference
    I0 = (1.0.-If-Ib)

    dV_Upwind = dVf.*If + dVb.*Ib + dV0.*I0

     global c = dV_Upwind.^(-1/σ)
     u = c.^(1-σ)/(1-σ)

     # create the transition matrix
     X = -min.(ssb,0)/da
     Y = -max.(ssf,0)/da + min.(ssb,0)/da
     Z = max.(ssf,0)/da


     A = sparse(Diagonal(Y[:])) + [zeros(1,H); sparse(Diagonal(X[2:H])) zeros(H-1,1)] + [zeros(H-1,1) sparse(Diagonal(Z[1:H-1])); zeros(1,H)]
     B = (ρ + 1/Δ)*sparse(I, H, H) - A

     b = u + V/Δ
     V = B\b
     V_change = V-v
     global v= V

	push!(dist,findmax(abs.(V_change))[1])
	if dist[n] .< ε
		println("Value Function Converged Iteration=")
		println(n)
		break
	end
end

plot(dist, grid=false,
		xlabel="Iteration", ylabel="||V^{n+1} - V^n||",
		ylims=(-0.001,0.030),
		legend=false, title="")
png("Convergence")


v_err = c.^(1-σ)/(1-σ) + dVb.*(w .+ r.*a -c) - ρ.*v

plot(a, v_err, grid=false,
		xlabel="k", ylabel="Error in the HJB equation",
		xlims=(a_min,a_max),
		legend=false, title="")
png("HJB_error")


plot(a, v, grid=false,
		xlabel="a", ylabel="V(a)",
		xlims=(a_min,a_max),
		legend=false, title="")
png("Value_function_vs_a")

plot(a, c, grid=false,
		xlabel="a", ylabel="c(a)",
		xlims=(a_min,a_max),
		legend=false, title="")
png("c(a)_vs_a")

# approximation at the borrowing constraint
a_dot = w.+ r.*a -c

plot(a, a_dot, grid=false,
		xlabel="a", ylabel="s(a)",
		xlims=(a_min,a_max), title="", label="s(a)", legend=:bottomleft)
plot!(a, zeros(H,1), label="")
png("stateconstraint")
