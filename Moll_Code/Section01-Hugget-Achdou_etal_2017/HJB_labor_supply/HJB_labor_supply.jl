#==============================================================================

          Solving the HJB with labor supply using the implicit method

               Translated from matlab code on Ben Moll's website:
                             https://benjaminmoll.com/codes/

==============================================================================#

using LinearAlgebra, SparseArrays, Plots, Roots, LaTeXStrings

γ = 2 # parameter from CRRA utility
φ = .5 # frisch elasticity
ρ = 0.05 # the discount rate
r = 0.03
w = 1.0

z1=.1
z2 =.2
z= [z1 z2]

λ1 = 1.5
λ2 = 1.0

λ = [λ1 λ2]

H = 500 # number of points in the grid space
amin= -0.15
amax= 3.0

a = LinRange(amin,amax,H)
da = (amax-amin)/(H-1)

aa = [a a]
zz = ones(H,1)*z

maxit = 20 # the maximum number of iterations we allow the finite differencing algorithm
crit = 10^(-6) # our critical value
Δ = 1000

dVf=zeros(H,2)
dVb = zeros(H,2)
c= zeros(H,2)

# Create the matrix that captures the systems dynamic from λ process
Aswitch = [-1*sparse(I,H,H)*λ[1] sparse(I,H,H)*λ[1]; sparse(I,H,H)*λ[2] -1*sparse(I,H,H)*λ[2]]

x0 = (w*z1)^(φ*(1-γ)/(1+γ*φ)) #inital value for labor

# create function for labor supply
labor_solve(l, a, z, w, r, γ, φ) = (l .- ((w*z*l .+ r*a).^(-γ*φ))*(w*z)^φ)

# set empty matrix and find all possible values for l
l0 = zeros(500,2)

for i in 1:H
    l_solve1(l) = labor_solve(l, a[i], z1, w, r, γ, φ)
    l01 = find_zero(l_solve1,x0)

    l_solve2(l) = labor_solve(l, a[i], z2, w, r, γ, φ)
    l02 = find_zero(l_solve2,x0)

    l0[i,:] = [l01 l02]
end


v0=zeros(H,2)
v0[:,1] = (w*z[1].*l0[1,1] .+ r.*a).^(1-γ)/(1-γ)/ρ
v0[:,2] = (w*z[2].*l0[1,2] .+ r.*a).^(1-γ)/(1-γ)/ρ
plot(a,v0[:,2])

lmin = l0[1,:]
lmax = l0[H,:]

global v=v0

# set up spaces to save variables of interest
dist=[]; V_n=[]; g_r=[]; dV_r=[]; V_r=[]; c_r=[]; adot=[]



# Loop that finds the equilibrium interest rate
for n in 1:maxit
    global V=v
    push!(V_n,V)
    # Forward differencing
    dVf[1:H-1,:]=(V[2:H,:]-V[1:H-1,:])/da
    dVf[H,:] = (w*z'.*lmax .+ r.*amax).^(-γ) # impose state constraint at the max, just in case
    # Backward differencing
    dVb[2:H,:]=(V[2:H,:]-V[1:H-1,:])/da
    dVb[1,:]= (w*z'.*lmin .+ r.*amin).^(-γ)

    # Find consumption and savings with the forward difference
    cf = dVf.^(-1/γ)
    lf = (dVf.*w.*zz).^φ
    sf = w*zz.*lf +r.*aa -cf

    # Find consumption and savings with the backward difference
    cb= dVb.^(-1/γ)
    lb = (dVb.*w.*zz).^φ
    sb = w*zz.*lb +r.*aa -cb

    # Find consumption and savings at steady state
    c0 = w*zz.*l0 + r.*aa
    dV0 = c0.^(-γ)

    # Now implement the upwind scheme in order to select the best differencing method
    Ib = (sb .< 0)
    If = (sf .> 0)#.*(1 .-Ib) this term was included in Matlab code, messes up julia code for some reason??? Probably rounding differences?
    I0 = (1 .- If - Ib)

    # State constraint at amin is automatically implemented
    global c = cf.*If + cb.*Ib + c0.*I0
    global l = lf.*If + lb.*Ib + l0.*I0
    u = c.^(1-γ)/(1-γ) - l.^(1+1/φ)/(1+1/φ)

    # Construct matrix for the evolution of the system
    global X = -Ib.*sb/da
    global Y = -If.*sf/da + Ib.*sb/da
    global Z = If.*sf/da

    A1 = spdiagm(0=>Y[:,1], -1 => X[2:H,1], 1=> Z[1:H-1,1])
    A2 = spdiagm(0=>Y[:,2], -1 => X[2:H,2], 1=> Z[1:H-1,2])
    AA = [A1 spzeros(H,H); spzeros(H,H) A2]

    global A = AA + Aswitch

    B = (ρ +1/Δ)*sparse(I,2*H,2*H) - A

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

adot = w.*zz.*l + r.*aa - c

#plot savings
plot(a, adot[:,1], label = L"s_1(a)", xlabel="Wealth, a", ylabel=L"\mathrm{Savings,\hspace{1ex} } s_i(a)")
plot!(a, adot[:,2], label = L"s_2(a)")
plot!(a, zeros(H,1), line=:dash, color=:black, label="")

#plot value function for j=1,2
plot(a, v[:,1], label = L"v_1(a)", xlabel="Wealth, a", ylabel=L"\mathrm{Value Function,\hspace{1ex} } v_i(a)")
plot!(a, v[:,2], label = L"v_2(a)", legend=:bottomright)


#plot labor for j=1,2
plot(a, l[:,1], label = L"l_1(a)", xlabel="Wealth, a", ylabel=L"\mathrm{Labor Supply,\hspace{1ex} } l_i(a)")
plot!(a, l[:,2], label = L"l_2(a)")

#plot consumption for j=1,2
plot(a, c[:,1], label = L"c_1(a)", xlabel="Wealth, a", ylabel=L"\mathrm{Consumption,\hspace{1ex} } c_i(a)")
plot!(a, c[:,2], label = L"c_2(a)", legend=:bottomright)
