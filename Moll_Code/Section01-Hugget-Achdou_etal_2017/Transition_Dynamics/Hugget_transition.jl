#===============================================================================

    Updating rule

    Based on code on Ben Moll's webite:
            www.princton.edu/~moll/HACTproject.htm

===============================================================================#

#Need to figure out best way to transfer inputs in julia
include("Hugget_inital.jl")
include("Hugget_terminal.jl")

# Get values for g, r, A from initial
g0, gg0, r, A = initial_val()

r00 = r
A0=A
AT0 = A'

# Get values for g_st, v_st, r_st from terminal
g_st, v_st, r_st = terminal()

plot(a,v_st)

plot(a, g)
plot!(a,g0, xlims=(-0.15,1))

# Create time and time increment variables
T = 20
N = 100
H =1000
dt = T/N

# Create and initial guess of the interest rate sequence
r0 = r_st*ones(N,1)

# Set up space to save variables
S = zeros(N,1)
global SS = zeros(N,1)
global dS = zeros(N,1)

v = zeros(H,2,N)

v[:,:,N] = v_st

r_new = r0
r_t =r0
A_t = []

maxit=1000
r_it =zeros(N,maxit)
Sdist = zeros(maxit,1)
dVf, dVb, c = [zeros(H,2) for i in 1:3]
dS_it=[]
SS_it=[]
gg=zeros(N+1,2*H,1)

convergence_criterion = 10^(-5)

# Speed of updating the interest rate
ξ = 20*(exp.(-0.05*(1:N))) .- exp(-0.05*N)

# main loop

for it in 1:maxit
    global r_t = r_new
    r_it[:,it] = r_t
    V = v_st

    for n in N:-1:1
        v[:,:,n]=V
        #forward difference
        dVf[1:H-1,:] = (V[2:H,:]-V[1:H-1,:])/da
        dVf[H,:] = (z .+ r_t[n].*amax).^(-s) #will never be used, but impose state constraint a<=amax just in case
        #backward difference
        dVb[2:H,:] = (V[2:H,:]-V[1:H-1,:])/da
        dVb[1,:] = (z .+ r_t[n].*amin).^(-s) #state constraint boundary condition

        I_concave = dVb .> dVf #indicator whether value function is concave (problems arise if this is not the case)

        # consumption and savings with forward difference
        cf = dVf.^(-1/s)
        ssf = zz + r_t[n].*aa - cf
        # consumption and savings with backward difference
        cb = dVb.^(-1/s)
        ssb = zz + r_t[n].*aa - cb
        # consumption and derivative of value function at steady state
        c0 = zz + r_t[n].*aa
        dV0 = c0.^(-s)

        #= dV_upwind makes a choice of forward or backward differences based on
         the sign of the drift    =#
        If = ssf .> 0 #positive drift → forward difference
        Ib = ssb .< 0 #negative drift → backward difference
        I0 = (1.0 .-If-Ib) #at steady state

        dV_Upwind = dVf.*If + dVb.*Ib + dV0.*I0  #important to include third term
        c = dV_Upwind.^(-1/s)
        u = c.^(1-s)/(1-s)

        # CONSTRUCT MATRIX
        X = - min.(ssb,0)/da
        Y = - max.(ssf,0)/da + min.(ssb,0)/da
        Z = max.(ssf,0)/da

        A1=spdiagm(0 => Y[:,1], -1 => X[2:H,1], 1 =>Z[1:H-1,1])
        A2=spdiagm(0=> Y[:,2], -1 =>X[2:H,2], 1 => Z[1:H-1,2])

        global A = [A1 spzeros(H,H);spzeros(H,H) A2] + Aswitch

        push!(A_t,A)
        B = (1/dt + ρ)*sparse(I, 2*H, 2*H) - A

        u_stacked = [u[:,1];u[:,2]]
        V_stacked = [V[:,1];V[:,2]]

        b = u_stacked + V_stacked/dt
        V_stacked = B\b #SOLVE SYSTEM OF EQUATIONS

        V = [V_stacked[1:H] V_stacked[H+1:2*H]]

        global ss = zz+ r_t[n].*aa -c
    end

    gg[1,:,:] = gg0

    #loop for KFE, solve forward
    for q in 1:N
        AT=A_t[end-(q-1)]' #the indexing is backward from original code.
        gg[q+1,:,:] = (sparse(I,2*H,2*H)-AT*dt)\gg[q,:,:]
        SS[q] = (gg[q,:,:]'*aa[:].*da)[1]
        dS[q] = (gg[q+1,:,:]'*aa[:]*da - gg[q,:,:]'*aa[:]*da)[1]
    end

    push!(dS_it,dS)
    push!(SS_it, SS)

    global r_new = r_t - ξ.*dS

    Sdist[it] = findmax(abs.(dS))[1]
    println("Iteration = $(it)")
    println("Convergence Criterion= $(Sdist[it])")

    if Sdist[it] < convergence_criterion
        break
    end
end

plot(Sdist[:])

# This graph is not right, but somehow the other ones are? Scoping issue?
#plot(1:N, SS_it[1], ylabel="Excess Supply", xlabel="Excess Demand", label="First iteration" )
#plot!(1:N, SS_it[end], label="Last Iternation")

plot(r_t[:], legend=false, xlabel="Period", ylabel="r")
plot!(1:N, r_st.*ones(N,1), line=:dash)

# More plots

N1 =4
T1=-N1*dt
time = (1:N)'*dt
time1 = T1 .+ (1:N1)'*dt
time2 = [time1'; time']
r_t2 = [r00*ones(N1,1); r_t]

plot(time2, r_t2, xlims=(T1, 10), ylims=(-0.05, 0.035), legend=:false)
plot!(time2, r_st*ones(N1+N,1), line=:dash,
        xlabel="Year", title="Equilibrium Interest Rate")

amax1 = 0.5
gmax=3
n=2


p1 = plot(a, gg[n,1:H,:],xlims=(amin,amax1),
        ylims=(0, gmax), xlabel="Wealth", ylabel="Densities",legend=false)
plot!(a, gg[n,H+1:end,:], title="\$t=0.1\$")

# set up new index
t=2
n=convert(Int64, t/dt)

p2 = plot(a, gg[n,1:H,:],xlims=(amin,amax1),
        ylims=(0, gmax), xlabel="Wealth", ylabel="Densities",legend=false)
plot!(a, gg[n,H+1:end,:], title="\$t=2\$")

t=5
n=convert(Int64, t/dt)

p3 = plot(a, gg[n,1:H,:],xlims=(amin,amax1),
        ylims=(0, gmax), xlabel="Wealth", ylabel="Densities",legend=false)
plot!(a, gg[n,H+1:end,:], title="\$t=5\$")

t=T
n=convert(Int64, t/dt)

p4 = plot(a, gg[n,1:H,:],xlims=(amin,amax1),
        ylims=(0, gmax), xlabel="Wealth", ylabel="Densities",legend=false)
plot!(a, gg[n,H+1:end,:], title="\$t=\\infty\$")


plot(p1,p2,p3,p4,legend=false,layout = (2,2))
