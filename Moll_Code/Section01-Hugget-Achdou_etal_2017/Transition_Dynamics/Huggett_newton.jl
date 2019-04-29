#===============================================================================

    Newtonian Method

    Based on code on Ben Moll's webite:
            www.princton.edu/~moll/HACTproject.htm

===============================================================================#

#Need to figure out best way to transfer inputs in julia
include("Huggett_initial.jl")
include("Huggett_terminal.jl")
include("Huggett_subroutine.jl")

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
global T = 20
global N = 100
global H =1000
global dt = T/N

f= zeros(N,1)
f1=zeros(N,1)



# Create and initial guess of the interest rate sequence
r0 = r_st*ones(N,1)

# Set up space to save variables
S = zeros(N,1)
global SS = zeros(N,1)
global dS = zeros(N,1)

v = zeros(H,2,N)

data=[]

push!(data, v_st)
push!(data, gg0)
push!(data, Aswitch)

maxit=100
convergence_criterion = 10^(-6)

# Set an intial guess of the interest rate squence
x = r_st.*ones(N,1)

#evaluate market clearing at inital guess
global A_t=[]
global gg=zeros(N+1,2*H,1)
f = huggett_subroutine(x, data)
println("Convergence Criterion at inital guess is: $(findmax(abs.(f))[1])")


# Compute the Jacobian at the inital guess
using SparseArrays
h = diagm(0=>sqrt(eps()).*max.(abs.(x),1)[:])

# Implement the Newtonian scheme
J = zeros(N,N)

# For loop for Newtonian method
for k in 1:N
    println("Coloumn of Jacobian = $(k)")

    x1 = x + h[:,k]
    global A_t=[]
    global gg=zeros(N+1,2*H,1)
    f1=huggett_subroutine(x1,data)
    J[:,k] = (f1-f)/h[k,k]
end

# Double check that the Jacobian has full rank
println("Rank of Jacobian = $(rank(J))")
J0 = J # save this inital Jacobian

# Dampening factor
ξ = 0.1

# main loop
Sdist=[]

for it in 1:maxit
    global J, f, x = J, f, x
    dx = -J\f

    x = x + ξ.*dx
    f = huggett_subroutine(x, data)
    push!(Sdist, findmax(abs.(f))[1])

    J = J + f *dx'./(dx'*dx)

    println("Iteration=$(it), Convergence Criterion=$(Sdist[it])")

    if Sdist[it]<convergence_criterion
        break
    end
end

plot(Sdist[:])

# Plot of r_t over time, note the oscillation at the end.
    # this is a common issue with Newtonian methods
r_t = x
plot(1:N, r_t[:], legend=false, xlabel="Period", ylabel="r")
plot!(1:N, r_st.*ones(N,1), line=:dash)

# More plots from tranistion dynamics code

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
