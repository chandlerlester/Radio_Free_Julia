#==============================================================================
    Code for solving the Hamiltonian Jacboi Bellman for
	   an basic model: ρv(a) = max_{c} u(c) + v'(a)[s(a)]
	   Where s(a) = w + ra - c(a)

	Translated Julia code from Matlab code by Ben Moll:
        http://www.princeton.edu/~moll/HACTproject.htm
==============================================================================#

using Parameters, Distributions, Plots

@with_kw type Model_parameters
    σ= 2.0 #
    ρ = 0.05 #the discount rate
    r = 0.045 # the depreciation rate
	w = 0.1
end

param = Model_parameters()
@unpack_Model_parameters(param)


I = 500
a_min = -0.02
a_max = 1.0

a = linspace(a_min, a_max, I)
a = convert(Array, a) # create grid for a values
da = (a_max-a_min)/(I-1)

maxit = 10000
ε = 10e-6

Δ = 1000

dVf, dVb, c = [zeros(I,1) for i =1:3]

#initial guess for V
v0 = (w+r.*a).^(1-σ)/(1-σ)/ρ
v= v0

dist=[]

for n=1:maxit
	V=v
    #forward difference
    dVf[1:I-1] = (V[2:I]-V[1:I-1])/da;
    dVf[I]= 0;

	# backward difference
	dVb[2:I] = (V[2:I]-V[1:I-1])/da
	dVb[1] = (w+r.*a_min).^(-σ) # the boundary condition

	I_concave = dVb .> dVf

    # consumption and savings with forward difference
    cf = dVf.^(-1/σ)
    ssf = w+r.*a-cf

    # consumption and savings with backward difference
	cb = dVb.^(-1/σ)
	ssb = w+r.*a-cb

    # consumption and savings at steady state
    c0=w+r.*a
    dV0 =c0.^(-σ)

    #look at the sign of the drift
        #to choose forward or backward difference
    If = ssf .> 0 # positive drift ⇒ forward difference
    Ib = ssb .< 0  # negative drift ⇒ backward difference
    I0 = (1-If-Ib)

    dV_Upwind = dVf.*If + dVb.*Ib + dV0.*I0

     c = dV_Upwind.^(-1/σ)
     u = c.^(1-σ)/(1-σ)

     # create the transition matrix
     X = -min(ssb,0)/da
     Y = -max(ssf,0)/da + min(ssb,0)/da
     Z = max(ssf,0)/da


     A = spdiagm((Y[:])) + [zeros(1,I); spdiagm((X[2:I])) zeros(I-1,1)] + [zeros(I-1,1) spdiagm((Z[1:I-1])); zeros(1,I)]
     B = (ρ + 1/Δ)*speye(I) - A

     b = u + V/Δ
     V = B\b
     V_change = V-v
     v= V

	push!(dist,findmax(abs(V_change))[1])
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


v_err = c.^(1-σ)/(1-σ) + dVb.*(w + r.*a -c) - ρ.*v

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
a_dot = w+ r.*a -c

plot(a, a_dot, grid=false,
		xlabel="a", ylabel="s(a)",
		xlims=(a_min,a_max), title="", label="s(a)", legend=:bottomleft)
plot!(a, zeros(I,1), label="")
png("stateconstraint")
