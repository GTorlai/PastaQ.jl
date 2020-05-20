function hadamardlayer!(N::Int,gates::Array)
  for j in 1:N
    push!(gates,(gate = "H",site = j))
  end
end

function rand1Qrotationlayer!(N::Int,gates::Array;
                              rng=nothing)
  for j in 1:N
    if isnothing(rng)
      θ,ϕ,λ = rand!(zeros(3))
    else
      θ,ϕ,λ = rand!(rng,zeros(3))
    end
    push!(gates,(gate = "Rn",site = j, params = (θ = θ, ϕ = ϕ, λ = λ)))
  end
end

function Cxlayer!(N::Int,gates::Array,sequence::String)
  if (N ≤ 2)
    throw(ArgumentError("Cxlayer is defined for N ≥ 3"))
  end
  
  if sequence == "odd"
    for j in 1:2:(N-N%2)
      push!(gates,(gate = "Cx", site = (j,j+1)))
    end
  elseif sequence == "even"
    for j in 2:2:(N+N%2-1)
      push!(gates,(gate = "Cx", site = (j,j+1)))
    end
  else
    throw(ArgumentError("Sequence not recognized"))
  end
end
