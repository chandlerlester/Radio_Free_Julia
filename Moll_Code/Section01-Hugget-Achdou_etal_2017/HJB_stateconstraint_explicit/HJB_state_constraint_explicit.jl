#==============================================================================

          Solving the HJB with a state constraint using
                the explicit method



               Translated from matlab code on Ben Moll's website:
                              http://www.princeton.edu/~moll/HACTproject.htm

==============================================================================#

using LinearAlgebra, SparseArrays, Plots

γ = 2 # parameter from CRRA utility
r = 0.03 # the interest rate
ρ = 0.05 # the discount rate

z1=.1
z2 =.2
z= [z1 z2]

λ1 = 0.02
λ2 = 0.03

λ = [λ1 λ2]

H = 500 # number of points in the grid space
amin=-0.02
amax=2

a = LinRange(amin,amax,H)
da = (amax-amin)/(H-1)

aa = [a a]
zz = ones(H,1)*z

maxit = 20000 # the maximum number of iterations we allow the finite differencing algorithm
crit = 10^(-6) # our critical value

dVf=zeros(H,2)
dVb = zeros(H,2)
c= zeros(H,2)

# Inital Guess
v0=zeros(H,2)
v0[:,1] = (z[1] .+ r.*a).^(1-γ)/(1-γ)/ρ
v0[:,2] = (z[2] .+ r.*a).^(1-γ)/(1-γ)/ρ
global v=v0

dist=[]

# The finite differeing loop

for n in 1:maxit
    global V=v
    # Forward differencing
    dVf[1:H-1,:]=(V[2:H,:]-V[1:H-1,:])/da
    dVf[H,:] = zeros(1,2) # will never be used
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
    I0 = (1 .- If - Ib)

    # Make sure the backward difference is using at the grid max
    Ib[H,:] = ones(1,2)
    If[H,:] = zeros(1,2)

    # State constraint at amin is automatically implemented

    global dV_Upwind = dVf.*If + dVb.*Ib + dV0.*I0

    global c = dV_Upwind.^(-1/γ)
    global V_switch = [V[:,2] V[:,1]]

    V_change = c.^(1-γ)/(1-γ) + dV_Upwind.*(zz + r.*aa -c) + ones(H,1)*λ.*(V_switch-V)-ρ.*V

    Δ = .9*da/(findmax(z2 .+r.*a)[1])
    global v = v  + Δ*V_change

    push!(dist, findmax(abs.(V_change))[1])

    if dist[n] < crit
        println("Value Function converged, Iteration=")
        println(n)
        break
    end
end

# Graphs
V_err = c.^(1-γ)/(1-γ) + dV_Upwind.*(zz +r.*aa -c) + ones(H,1)*λ.*(V_switch-V)-ρ.*V
adot = zz + r.*aa -c

plot(dist[:], xlabel="Iteration", ylabel="\$\\lvert V^{n+1}-V^{n}\\rvert\$")

plot(a,V_err, xlabel="k",ylabel="Error in HJB Equation", xlims=(amin,amax), legend=false)

plot(a,V, xlabel="a", ylabel="\$V_{i}(a)\$", xlims=(amin,amax),legend=false)

plot(a,c,xlabel="a", ylabel="\$c_{i}(a)\$", xlims=(amin,amax), legend=false)

plot(a, adot, xlabel="a", ylabel="\$s_{i}(a)\$", xlims=(amin,amax))

# Approximation at the borrowing constraint
u1=(z1+r*amin)^(-γ)
u2 = c[1,2]^(-γ)
u11 = -γ*(z1+r*amin)^(-γ-1)
ν = sqrt(-2*((ρ-r)*u1 + λ1*(u1-u2))/u11)
s_approx = -ν*(a.-amin).^(1/2)

plot(a, adot[:,1], xlabel="a", ylabel="\$s_{i}(a)\$", xlims=(amin,amax), label="\$s_{1}(a)\$")
plot!(a, adot[:,2], label="\$s_{2}(a)\$")
plot!(a, zeros(H,1), line=:dash, label="", color=:black)
plot!(a, s_approx, label="Approximatation")
