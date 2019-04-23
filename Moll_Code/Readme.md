
# This is all code replicated from Ben Moll's Website: 
## http://www.princeton.edu/~moll/HACTproject.htm 

## I have currently converted the follow matlab files to julia code  

### Section 1 Hugget Model  
- HJB_stateconstraint_explicit.m = HJB_stateconstraint_explicit.jl 
- HJB_stateconstraint_implicit.m = HJB_stateconstraint_implicit.jl
- huggett_partialeq.m = huggett_partialeq.jl
- huggett_asset_supply.m = huggett_asset_supply.jl
- huggett_equilibrium_iterate.m = huggett_equilibrium_iterate.jl 
- Transition Dynamics 
  + huggett_terminal.m = Huggett_terminal.jl
  + huggett_initial.m = Huggett_initial.jl
  + huggett_transition.m = Huggett_transition.jl (simple updating method) 
  

### Section 2 Aiyagari Model 
- aiyagari_poisson_steadystate.m = aiyagari_poisson_SSE.jl
- aiyagari_poisson_asset_supply.m = Aiyagari_poisson_Asset_Supply/aiyagari_poisson_SSE.jl
- aiygari_diffusion_equilibrium.m = aiyagari_diffusion_SSE.jl

### Section 15 Additional Codes 
- HJB_simple.m = HJB_simple.jl 
- HJB_no_uncertainty_explicit.m = HJB_simple_explicit.jl 
- HJB_no_uncertainty_implicit.m = HJB_simple_implicit.jl 
- HJB_ngm.m = HJB_NGM.jl 
- HJB_ngm_implicit.m = HJB_NGM_implicit.jl 
- HJB_diffusion_implicit_RBC.m = RBC_diffusion.jl 
