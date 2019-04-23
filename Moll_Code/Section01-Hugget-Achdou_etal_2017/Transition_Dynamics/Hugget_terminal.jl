#===============================================================================

	Code for finding the terminal state of the HJB for a Hugget Model


===============================================================================#

using LinearAlgebra, SparseArrays, Plots

function terminal()
	# Define model parameters

	global s = 2
	global ρ = 0.05
	z1 = .1
	z2 = .2
	global z = [z1 z2]
	la1 = 0.6
	la2 = 0.8
	λ = [la1 la2]

	r0 = 0.03
	rmin = 0.01
	rmax = 0.04

	H= 1000
	global amin = -0.15
	global amax = 4
	a = LinRange(amin,amax,H)'
	global a = convert(Array, a)'
	global da = (amax-amin)/(H-1);

	global aa = [a a]
	global zz = ones(H,1)*z

	maxit= 100
	crit = 1e-6
	Delta = 100

	dVf, dVb, c = [zeros(H,2) for i in 1:3]

	global Aswitch = [-sparse(I, H, H)*λ[1] sparse(I, H, H)*λ[1];sparse(I, H, H)*λ[2] -sparse(I, H, H)*λ[2]]

	Ir = 40
	crit_S = 1e-8

	# INITIAL GUESS
	r = r0

	v0=zeros(H,2)
	v0[:,1] = (z[1] .+ r.*a).^(1-s)/(1-s)/ρ
	v0[:,2] = (z[2] .+ r.*a).^(1-s)/(1-s)/ρ
	dist =[]
	r_r=[]
	rmin_r=[]
	rmax_r=[]
	V_n =[]
	g_r=[]
	adot=[]
	V_r=[]
	S=[]

	for ir=1:Ir
		push!(r_r,r)
		push!(rmin_r, rmin)
		push!(rmax_r, rmax)

		global v = v0

		for n=1:maxit
		    V = v
		    push!(V_n,V)
		    #forward difference
		    dVf[1:H-1,:] = (V[2:H,:]-V[1:H-1,:])/da
		    dVf[H,:] = (z .+ r.*amax).^(-s) #will never be used, but impose state constraint a<=amax just in case
		    #backward difference
		    dVb[2:H,:] = (V[2:H,:]-V[1:H-1,:])/da
		    dVb[1,:] = (z .+ r.*amin).^(-s) #state constraint boundary condition

		    I_concave = dVb .> dVf #indicator whether value function is concave (problems arise if this is not the case)

		    # consumption and savings with forward difference
		    cf = dVf.^(-1/s)
		    ssf = zz + r.*aa - cf
		    # consumption and savings with backward difference
		    cb = dVb.^(-1/s)
		    ssb = zz + r.*aa - cb
		    # consumption and derivative of value function at steady state
		    c0 = zz + r.*aa
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

		    B = (1/Delta + ρ)*sparse(I, 2*H, 2*H) - A

		    u_stacked = [u[:,1];u[:,2]]
		    V_stacked = [V[:,1];V[:,2]]

		    b = u_stacked + V_stacked/Delta
		    V_stacked = B\b #SOLVE SYSTEM OF EQUATIONS

		    V = [V_stacked[1:H] V_stacked[H+1:2*H]]

		    V_change = V - v
		    v = V

		    push!(dist,findmax(abs.(V_change))[1])
		    if dist[n]<crit
		        println("Value Function Converged, Iteration = ")
		        println(n)
		        break
		    end
		end

	#===============================================================================

		Code for finding the terminal state of the KFE


	===============================================================================#
		AT = A'
		b = zeros(2*H,1)

		# need to fix one value, otherwise matrix is singular
		i_fix = 1
		b[i_fix]=.1
		row = [zeros(1,i_fix-1) 1 zeros(1,2*H-i_fix)]
		AT[i_fix,:] = row

		# Solve linear system
		gg = AT\b
		g_sum = gg'*ones(2*H,1)*da
		gg = gg./g_sum;

		global g = [gg[1:H] gg[H+1:2*H]]

		check1 = g[:,1]'*ones(H,1)*da
		check2 = g[:,2]'*ones(H,1)*da

		push!(g_r, g)
		push!(adot,zz + r.*aa - c)
		push!(V_r,V)

		push!(S,g[:,1]'*a*da + g[:,2]'*a*da)

			# UPDATE INTEREST RATE
			if S[ir][1] > crit_S
			    println("Excess Supply")
			    rmax = r
			    r = 0.5*(r+rmin)
			elseif S[ir][1]< -crit_S
			    println("Excess Demand")
			    rmin = r
			    r = 0.5*(r+rmax)
			elseif abs(S[ir][1])<crit_S
			    println("Equilibrium Found, Interest rate =")
			    println(r)
			    break
			end
	end

return g, v, r

end
