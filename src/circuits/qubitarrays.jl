"""
    lineararray(N::Int)

Return a vector of bonds for a open 1d lattice with
`N` sites.
"""
function lineararray(N::Int64)
  couplings = Vector{Vector{<:Tuple}}(undef, 0)
  # Cycle 1
  cycle = Vector{Tuple}(undef, 0)
  for j in 1:2:(N - 1)
    push!(cycle, (j, j + 1))
  end
  push!(couplings, cycle)
  if N > 2
    # Cycle 2
    cycle = Vector{Tuple}(undef, 0)
    for j in 2:2:(N - 1)
      push!(cycle, (j, j + 1))
    end
    push!(couplings, cycle)
  end
  return couplings
end

"""
    squarearray(Lx::Int,Ly::Int)

Return a vector containing 4 different "cycles" of bonds,
corresponding to the different tiling of a square lattice
with dimensions `Lx` and `Ly`.
Return a vector of bonds for a open 1d lattice with
`N` sites.
"""
function squarearray(Lx::Int, Ly::Int; rotated::Bool=false)
  site_index(x::Int, y::Int) = Lx * (y - 1) + x

  couplings = Vector{Vector{<:Tuple}}(undef, 0)

  if !rotated
    #
    #   Cycle A                Cycle B
    #   o - o   o - o          o   o   o   o
    #
    #   o - o   o - o          o   o   o   o
    #                          |   |   |   |
    #   o - o   o - o          o   o   o   o
    #
    #   o - o   o - o          o   o   o   o
    #
    #   Cycle 3                Cycle 4
    #   o   o   o   o          o   o - o   o
    #   |   |   |   |
    #   o   o   o   o          o   o - o   o
    #
    #   o   o   o   o          o   o - o   o
    #   |   |   |   |
    #   o   o   o   o          o   o - o   o
    #

    # A
    cycle = Vector{Tuple}(undef, 0)
    for y in 1:Ly
      for x in 1:2:(Lx - 1)
        push!(cycle, (site_index(x, y), site_index(x + 1, y)))
      end
    end
    push!(couplings, cycle)
    # B
    cycle = Vector{Tuple}(undef, 0)
    for y in 2:2:(Ly - 1)
      for x in 1:Lx
        push!(cycle, (site_index(x, y), site_index(x, y + 1)))
      end
    end
    push!(couplings, cycle)
    #C
    cycle = Vector{Tuple}(undef, 0)
    for y in 1:2:(Ly - 1)
      for x in 1:Lx
        push!(cycle, (site_index(x, y), site_index(x, y + 1)))
      end
    end
    push!(couplings, cycle)
    # D
    cycle = Vector{Tuple}(undef, 0)
    for y in 1:Ly
      for x in 2:2:(Lx - 1)
        push!(cycle, (site_index(x, y), site_index(x + 1, y)))
      end
    end
    push!(couplings, cycle)

  else

    #   Cycle A                Cycle B
    #   o   o   o   o          o   o   o   o
    #    ╲   ╲   ╲   ╲            ╱   ╱   ╱
    #     o   o   o   o          o   o   o   o
    #
    #   o   o   o   o          o   o   o   o
    #    ╲   ╲   ╲   ╲            ╱   ╱   ╱
    #     o   o   o   o          o   o   o   o
    #
    #   Cycle C                Cycle D
    #   o   o   o   o          o   o   o   o
    #
    #     o   o   o   o          o   o   o   o
    #      ╲   ╲   ╲                ╱   ╱   ╱
    #   o   o   o   o          o   o   o   o
    #
    #     o   o   o   o          o   o   o   o

    # A
    cycle = Vector{Tuple}(undef, 0)
    for y in 1:2:(Ly - 1)
      for x in 1:Lx
        push!(cycle, (site_index(x, y), site_index(x, y + 1)))
      end
    end
    push!(couplings, cycle)
    # B
    cycle = Vector{Tuple}(undef, 0)
    for y in 1:2:(Ly - 1)
      for x in 2:Lx
        push!(cycle, (site_index(x, y), site_index(x - 1, y + 1)))
      end
    end
    push!(couplings, cycle)
    # C
    cycle = Vector{Tuple}(undef, 0)
    for y in 2:2:(Ly - 1)
      for x in 1:(Lx - 1)
        push!(cycle, (site_index(x, y), site_index(x + 1, y + 1)))
      end
    end
    push!(couplings, cycle)
    # D
    cycle = Vector{Tuple}(undef, 0)
    for y in 2:2:(Ly - 1)
      for x in 1:Lx
        push!(cycle, (site_index(x, y), site_index(x, y + 1)))
      end
    end
    push!(couplings, cycle)
  end
  return couplings
end

function randomcouplings(N::Int, R::Int)
  @assert iseven(N)
  qubit_list = collect(1:N)
  couplings = Vector{Vector{Int}}(undef, 0)

  for b in 1:(N ÷ 2)
    q = qubit_list[1]
    dist = rand(1:(R - 1))
    while (isnothing(findfirst(x -> x == q + dist, qubit_list)))
      dist = rand(1:(R - 1))
    end
    push!(couplings, [q, q + dist])
    deleteat!(qubit_list, findfirst(x -> x == q, qubit_list))
    deleteat!(qubit_list, findfirst(x -> x == q + dist, qubit_list))
  end
  return couplings
end

function randomcoupling(Lx::Int, Ly::Int, R::Int)
  return error("Random couplings not implemented in 2D")
end
