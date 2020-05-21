struct QST
  N::Int
  d::Int
  χ::Int
  seed::Int
  rng::MersenneTwister
  σ::Float64
  parstype::String
  psi::MPS
  infos::Dict
end

function QPT(;N::Int,
             d::Int=2,
             χ::Int=2,
             seed::Int=1234,
             σ::Float64=0.1,
             parstype="real")

  infos = Dict()
  rng = MersenneTwister(seed)
  
  sites = [Index(d; tags="Site, n=$s") for s in 1:N]
  links = [Index(χ; tags="Link, l=$l") for l in 1:N-1]

  M = ITensor[]
  # Site 1 
  rand_mat = σ * (ones(d,χ) - 2*rand!(rng,zeros(d,χ)))
  if parstype == "complex"
    rand_mat += im * σ * (ones(d,χ) - 2*rand!(rng,zeros(d,χ)))
  end
  push!(M,ITensor(rand_mat,sites[1],links[1]))
  # Site 2..N-1
  for j in 2:N-1
    rand_mat = σ * (ones(χ,d,χ) - 2*rand!(rng,zeros(χ,d,χ)))
    if parstype == "complex"
      rand_mat += im * σ * (ones(χ,d,χ) - 2*rand!(rng,zeros(χ,d,χ)))
    end
    push!(M,ITensor(rand_mat,links[j-1],sites[j],links[j]))
  end
  # Site N
  rand_mat = σ * (ones(χ,d) - 2*rand!(rng,zeros(χ,d)))
  if parstype == "complex"
    rand_mat += im * σ * (ones(χ,d) - 2*rand!(rng,zeros(χ,d)))
  end
  push!(M,ITensor(rand_mat,links[N-1],sites[N]))
  
  psi = MPS(M)

  infos["N"]     = N
  infos["d"]     = d
  infos["chi"]   = χ
  infos["seed"]  = seed
  infos["sigma"] = σ
    
  return QST(N,d,χ,seed,rng,σ,parstype,psi,infos)
end


