
struct Choi{MPOT <: Union{MPO, LPDO}}
  M::MPOT
end


"""
    choimatrix(N::Int, gates::Vector{<:Tuple}; noise=nothing, apply_dag=false,
               cutoff=1e-15, maxdim=10000, kwargs...)

Compute the Choi matrix `Λ` from a set of gates that make up a quantum process.

If `noise = nothing` (the default), for an N-qubit process, by default the square 
root of the Choi matrix `|U⟩` is returned, such that the Choi matrix is the rank-1 matrix 
`Λ = |U⟩⟨U|`. `|U⟩` is an MPS with `2N` sites for a process running on `N` physical qubits. 
It is the "state" version of the unitary approximation for the full gate evolution `U`.

If `noise != nothing`, an approximation for the Choi matrix is returned as an MPO 
with `2N` sites, for a process with `N` physical qubits.

If `noise = nothing` and `apply_dag = true`, the Choi matrix `Λ` is returned as an MPO with 
`2N` sites. In this case, the MPO `Λ` is equal to `|U⟩⟨U|`.
"""
function choimatrix(N::Int,
                    gates::Vector{<:Tuple};
                    noise = nothing, apply_dag = false,
                    cutoff = 1e-15, maxdim = 10000, kwargs...)
  if isnothing(noise)
    # Get circuit MPO
    U = runcircuit(N, gates;
                   process = true, cutoff = 1e-15,
                   maxdim = 10000, kwargs...)
    
    # Choi indices 
    addtags!(U, "Input", plev = 0, tags = "Qubit")
    addtags!(U, "Output", plev = 1, tags = "Qubit")
    noprime!(U)
    # SVD to bring into 2N-sites MPS
    Λ0 = splitchoi(U,noise=nothing,cutoff=cutoff,maxdim=maxdim)
    # if apply_dag = true:  Λ = |U⟩⟩ ⟨⟨U†|
    # if apply_dag = false: Λ = |U⟩⟩
    Λ = (apply_dag ? MPO(Λ0) : Λ0)
  else
    # Initialize circuit MPO
    U = circuit(N)
    addtags!(U,"Input",plev=0,tags="Qubit")
    addtags!(U,"Output",plev=1,tags="Qubit")
    prime!(U,tags="Input")
    prime!(U,tags="Link")
    
    s = [siteinds(U,tags="Output")[j][1] for j in 1:length(U)]
    compiler = circuit(s)
    prime!(compiler,-1,tags="Qubit") 
    gate_tensors = compilecircuit(compiler, gates; noise=noise, kwargs...)

    M = ITensor[]
    push!(M,U[1] * noprime(U[1]))
    Cdn = combiner(inds(M[1],tags="Link")[1],inds(M[1],tags="Link")[2],
                  tags="Link,n=1")
    M[1] = M[1] * Cdn
    for j in 2:N-1
      push!(M,U[j] * noprime(U[j]))
      Cup = Cdn
      Cdn = combiner(inds(M[j],tags="Link,n=$j")[1],inds(M[j],tags="Link,n=$j")[2],tags="Link,n=$j")
      M[j] = M[j] * Cup * Cdn
    end
    push!(M, U[N] * noprime(U[N]))
    M[N] = M[N] * Cdn
    ρ = MPO(M)
    Λ0 = runcircuit(ρ,gate_tensors;apply_dag=true,cutoff=cutoff, maxdim=maxdim)
    Λ = splitchoi(Λ0,noise=noise,cutoff=cutoff,maxdim=maxdim)
  end
  return Λ
end


"""
  splitchoi(Λ::MPO;noise=nothing,cutoff=1e-15,maxdim=1000)

Map a Choi matrix from `N` sites to `2N` sites, arranged as
(input1,output1,input2,output2,…)
"""
function splitchoi(Λ::MPO;noise=nothing,cutoff=1e-15,maxdim=1000)
  T = ITensor[]
  if isnothing(noise)
    u,S,v = svd(Λ[1],firstind(Λ[1],tags="Input"), 
                cutoff=cutoff, maxdim=maxdim)
  else
    u,S,v = svd(Λ[1],inds(Λ[1],tags="Input"), 
                cutoff=cutoff, maxdim=maxdim)
  end
  push!(T,u*S)
  push!(T,v)
  
  for j in 2:length(Λ)
    if isnothing(noise)
      u,S,v = svd(Λ[j],firstind(Λ[j],tags="Input"),commonind(Λ[j-1],Λ[j]),
                  cutoff=cutoff,maxdim=maxdim)
    else
      u,S,v = svd(Λ[j],inds(Λ[j],tags="Input")[1],inds(Λ[j],tags="Input")[2],
                  commonind(Λ[j-1],Λ[j]),cutoff=cutoff,maxdim=maxdim) 
    end
    push!(T,u*S)
    push!(T,v)
  end
  Λ_split = (isnothing(noise) ? MPS(T) : MPO(T))
  return Λ_split
end

