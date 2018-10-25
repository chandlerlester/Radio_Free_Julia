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

maxit = 20000
ε = 10e-6

dVf = zeros(I,1)
dVb = zeros(I,1)
c = zeros(I,1)

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
    μ_f = w+r.*a-cf

    # consumption and savings with backward difference
	cb = dVb.^(-1/σ)
	μ_b = w+r.*a-cb

    # consumption and savings at steady state
    c0=w+r.*a
    dV0 =c0.^(-σ)

    #look at the sign of the drift
        #to choose forward or backward difference
    If = μ_f .> 0 # positive drift ⇒ forward difference
    Ib = μ_b .< 0  # negative drift ⇒ backward difference
    I0 = (1-If-Ib)

    dV_Upwind = dVf.*If + dVb.*Ib + dV0.*I0

     c = dV_Upwind.^(-1/σ)
     V_change = c.^(1-σ)/(1-σ) + dV_Upwind.*(w + r.*a - c) - ρ.*V


	# update
	Δ = .9*da/(findmax(w + r.*a)[1])
	v = v + Δ*V_change

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
u_1 = (w+r*a_min)^(-σ)
u_2 = -σ*(w+r*a_min)^(-σ-1)
ν = sqrt(-2*(ρ-r)*u_1/u_2)
s_approx = -ν*(a-a_min).^(1/2)

plot(a, a_dot, grid=false,
		xlabel="a", ylabel="s(a)",
		xlims=(a_min,a_max), title="", label="s(a)", legend=:bottomleft)
plot!(a, s_approx, label="Approximation of s(a)")
plot!(a, zeros(I,1), label="")
png("stateconstraint")
