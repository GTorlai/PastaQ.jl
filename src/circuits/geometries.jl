function randompairing(N::Int, R::Int)
  @assert iseven(N)
  qubit_list = 1:N |> collect
  bonds = Tuple[]
  
  for b in 1:N÷2
    q = qubit_list[1]
    dist = rand(1:R-1)
    while (isnothing(findfirst(x -> x == q+dist,qubit_list)))
      dist = rand(1:R-1)
    end
    push!(bonds,(q,q+dist));
    deleteat!(qubit_list,findfirst(x -> x == q, qubit_list))
    deleteat!(qubit_list,findfirst(x -> x == q+dist, qubit_list))  
  end
  return bonds
end



"""
Circuit geometries
"""

# Get site number from coordinate
coord_to_site(Lx::Int,Ly::Int,x::Int,y::Int) = Lx*(y-1) + x

"""
    lineararray(N::Int)

Return a vector of bonds for a open 1d lattice with 
`N` sites.
"""
# Returns two-qubit bonds for a linear array
function lineararray(N::Int)
  twoqubit_bonds = Vector{Vector{Vector{Int}}}(undef,0)
  # Cycle 1
  cycle = Vector{Vector{Int}}(undef,0)
  for j in 1:2:N-1
    push!(cycle,[j, j+1])
  end
  push!(twoqubit_bonds,cycle)
  if N>2
    # Cycle 2
    cycle = Vector{Vector{Int}}(undef,0)
    for j in 2:2:N-1
      push!(cycle,[j,j+1])
    end
    push!(twoqubit_bonds,cycle)
  end
  return twoqubit_bonds
end


