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
