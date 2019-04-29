#===============================================================================

    Subroutine for the Newtonian Method

    Based on code on Ben Moll's webite:
            www.princton.edu/~moll/HACTproject.htm

===============================================================================#


function huggett_subroutine(r_t, data)

    v_st = data[1]
    gg_0 = data[2]
    Aswitch = data[3]

    SS, dS = [zeros(N,1) for i in 1:2]

    dVf, dVb = [zeros(H,2) for j in 1:2]

    V=v_st

    v_n = zeros(H,2,N)

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

return dS
end
