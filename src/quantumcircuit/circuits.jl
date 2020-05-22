" INNER CIRCUITS "

function hadamardlayer!(gates::Array,N::Int)
  for j in 1:N
    push!(gates,(gate = "H",site = j))
  end
end

function rand1Qrotationlayer!(gates::Array,N::Int;
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

function Cxlayer!(gates::Array,N::Int,sequence::String)
  if (N ≤ 2)
    throw(ArgumentError("Cxlayer is defined for N ≥ 3"))
  end
  
  if sequence == "odd"
    for j in 1:2:(N-N%2)
      push!(gates,(gate = "Cx", site = [j,j+1]))
    end
  elseif sequence == "even"
    for j in 2:2:(N+N%2-1)
      push!(gates,(gate = "Cx", site = [j,j+1]))
    end
  else
    throw(ArgumentError("Sequence not recognized"))
  end
end


" MEASUREMENT CIRCUITS"

function measurementcircuit(N::Int,bases::Array)
  circuit = []
  randomsamples = rand(1:length(bases),N)
  for j in 1:N
    localbasis = bases[randomsamples[j]]
    if localbasis == "X"
      push!(circuit,(gate = "H", site = j))
    elseif localbasis == "Y"
      push!(circuit,(gate = "Km", site = j))
    else
      throw(argumenterror("Basis not recognized"))
    end
  end
  return circuit
end

## Changed it so it makes namedtupled
#function statepreparationcircuit(mps::MPS,prep::Array)
#  circuit = []
#  for j in 1:N
#    if prep[j] == "Xp"
#      push!(circuit,makegate(mps,"H",j))
#    elseif prep[j] == "Xm"
#      push!(circuit,makegate(mps,"X",j))
#      push!(circuit,makegate(mps,"H",j))
#    elseif prep[j] == "Yp"
#      push!(circuit,makegate(mps,"Kp",j))
#    elseif prep[j] == "Ym"
#      push!(circuit,makegate(mps,"X",j))
#      push!(circuit,makegate(mps,"Kp",j))
#    elseif prep[j] == "Zp"
#      nothing
#    elseif prep[j] == "Zm"
#      push!(circuit,makegate(mps,"X",j))
#    end
#  end
#  return circuit
#end

