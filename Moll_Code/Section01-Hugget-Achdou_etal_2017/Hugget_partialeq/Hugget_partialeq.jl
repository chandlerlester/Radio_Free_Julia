#==============================================================================

          Solving the HJB with a state constraint using
                the explicit method



               Translated from matlab code on Ben Moll's website:
                              http://www.princeton.edu/~moll/HACTproject.htm

==============================================================================#

using LinearAlgebra, SparseArrays, Plots

γ = 2 # parameter from CRRA utility
r = 0.035 # the interest rate, exogenous in this version
ρ = 0.05 # the discount rate

z1=.1
z2 =.2
z= [z1 z2]

λ1 = 1.5
λ2 = 1.0

λ = [λ1 λ2]

H = 500 # number of points in the grid space
amin=-0.02
amax= 3

a = LinRange(amin,amax,H)
da = (amax-amin)/(H-1)

aa = [a a]
zz = ones(H,1)*z

maxit = 100 # the maximum number of iterations we allow the finite differencing algorithm
crit = 10^(-6) # our critical value
Δ = 1000

dVf=zeros(H,2)
dVb = zeros(H,2)
c= zeros(H,2)

# Create the matrix that captures the systems dynamic from λ process
Aswitch = [-sparse(I,H,H)*λ[1] sparse(I,H,H)*λ[1]; sparse(I,H,H)*λ[2] -sparse(I,H,H)*λ[2]]

# Inital Guess
v0=zeros(H,2)
v0[:,1] = (z[1] .+ r.*a).^(1-γ)/(1-γ)/ρ
v0[:,2] = (z[2] .+ r.*a).^(1-γ)/(1-γ)/ρ
global v=v0

dist=[]
V_n=[]

# The finite differeing loop

for n in 1:maxit
    global V=v
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
    I0 = (1 .- If - Ib)

    # State constraint at amin is automatically implemented

    dV_Upwind = dVf.*If + dVb.*Ib + dV0.*I0

    global c = dV_Upwind.^(-1/γ)
    u = c.^(1-γ)/(1-γ)
    # Construct matrix for the evolution of the system
    X = -min.(sb,0)/da
    Y= -max.(sf,0)/da + min.(sb,0)/da
    Z = max.(sf,0)/da

    A1 = spdiagm(0=>Y[:,1], -1 => X[2:H,1], 1=> Z[1:H-1,1])
    A2 = spdiagm(0=>Y[:,2], -1 => X[2:H,2], 1=> Z[1:H-1,2])
    AA = [A1 spzeros(H,H); spzeros(H,H) A2]

    global A = AA + Aswitch

    B = (ρ +1/Δ)*sparse(I,2*H,2*H) -A
    u_stacked = [u[:,1]; u[:,2]]
    V_stacked = [V[:,1]; V[:,2]]

    b = u_stacked + V_stacked/Δ
    V_stacked = B\b # Solves the system of equations

    V = [V_stacked[1:H] V_stacked[H+1:2*H]]

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
b =zeros(2*H,1)

# We need to fix one value to prevent the matrix from being singular
i_fix =1
b[i_fix] = .1
row = [zeros(1,i_fix-1) 1 zeros(1,2*H-i_fix)]
AT[i_fix,:] = row

# Using this solve the linear system
gg = AT\b
g_sum = gg'*ones(2*H,1)*da
gg = gg./g_sum # we need gg to sum to one

g = [gg[1:H] gg[H+1:2*H]]

# Graphs
adot = zz + r.*aa -c
plot(a, adot, xlabel="a", ylabel="\$s_{i}(a)\$", xlims=(amin,amax))
plot!(a, zeros(H,1), label="")

S = g[:,1]'*a*da + g[:,2]'*a*da

# Approximation at the borrowing constraint
u1=(z1+r*amin)^(-γ)
u2 = c[1,2]^(-γ)
u11 = -γ*(z1+r*amin)^(-γ-1)
ν = sqrt(-2*((ρ-r)*u1 + λ1*(u1-u2))/u11)
s_approx = -ν*(a.-amin).^(1/2)

amax1=1

plot(a, adot[:,1], xlabel="a", ylabel="\$s_{i}(a)\$", xlims=(amin,amax1), label="\$s_{1}(a)\$")
plot!(a, adot[:,2], label="\$s_{2}(a)\$")
plot!(a, zeros(H,1), line=:dash, label="", color=:black)
plot!(a, s_approx, label="Approximatation")


plot(a,g[:,1],xlabel="a", ylabel="\$g_{i}(a)\$", label="\$g_{1}(a)\$", xlims=(amin,amax))
plot!(a,g[:,2],label="\$g_{2}(a)\$")
