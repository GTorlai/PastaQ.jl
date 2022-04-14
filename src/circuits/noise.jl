function gate(
  ::GateName"pauli_channel",
  ::SiteType"Qubit",
  s::Index...;
  pauli_ops=["Id", "X", "Y", "Z"],
  error_probabilities=prepend!(zeros(length(pauli_ops)^length(dim.(s)) - 1), 1),
)
  @ignore_derivatives begin
    dims = dim.(s)
    @assert sum(error_probabilities) ≈ 1
    @assert all(dims .== 2)
    N = length(dims)
    length(error_probabilities) > (1 << 10) && error("Hilbert space too large")
    error_probabilities ./= sum(error_probabilities)
    kraus = zeros(Complex{Float64}, 1 << N, 1 << N, length(pauli_ops)^N)
    krausind = Index(size(kraus, 3); tags="kraus")

    basis = vec(reverse.(collect(Iterators.product(fill(pauli_ops, N)...))))
    for (k, ops) in enumerate(basis)
      kraus[:, :, k] =
        √error_probabilities[k] * reduce(kron, [gate(op, SiteType("Qubit")) for op in ops])
    end
    return ITensors.itensor(kraus, prime.(s)..., ITensors.dag.(s)..., krausind)
  end
end

is_single_qubit_noise(::GateName"pauli_channel") = false

function gate(::GateName"bit_flip", st::SiteType"Qubit", s::Index...; p::Number=0.0)
  @ignore_derivatives begin
    error_probabilities = prepend!(
      p / (2^length(dim.(s)) - 1) * ones(2^length(dim.(s)) - 1), 1 - p
    )
    return gate(
      GateName("pauli_channel"),
      st,
      s...;
      error_probabilities=error_probabilities,
      pauli_ops=["Id", "X"],
    )
  end
end
is_single_qubit_noise(::GateName"bit_flip") = false

function gate(::GateName"phase_flip", st::SiteType"Qubit", s::Index...; p::Number=0.0)
  @ignore_derivatives begin
    error_probabilities = prepend!(
      p / (2^length(dim.(s)) - 1) * ones(2^length(dim.(s)) - 1), 1 - p
    )
    return gate(
      GateName("pauli_channel"),
      st,
      s...;
      error_probabilities=error_probabilities,
      pauli_ops=["Id", "Z"],
    )
  end
end
is_single_qubit_noise(::GateName"phase_flip") = false

# make general n-qubit
function gate(::GateName"DEP", st::SiteType"Qubit", s::Index...; p::Number)
  @ignore_derivatives begin
    error_probabilities = prepend!(
      p / (4^length(dim.(s)) - 1) * ones(4^length(dim.(s)) - 1), 1 - p
    )
    gate(
      GateName("pauli_channel"),
      st,
      s...;
      error_probabilities=error_probabilities,
      pauli_ops=["Id", "X", "Y", "Z"],
    )
  end
end
function gate(::GateName"depolarizing", st::SiteType"Qubit", s::Index...; kwargs...)
  return gate(GateName("DEP"), st, s...; kwargs...)
end
is_single_qubit_noise(::GateName"DEP") = false
is_single_qubit_noise(::GateName"depolarizing") = false

function gate(::GateName"AD", ::SiteType"Qubit", s::Index...; γ::Real=0.0)
  @ignore_derivatives begin
    dims = dim.(s)
    N = length(dims)
    @assert all(dims .== 2)

    kraus = zeros(2, 2, 2)
    kraus[:, :, 1] = [
      1 0
      0 sqrt(1 - γ)
    ]
    kraus[:, :, 2] = [
      0 sqrt(γ)
      0 0
    ]
    krausind = Index(size(kraus, 3); tags="kraus")
    return ITensors.itensor(kraus, prime.(s)..., ITensors.dag.(s)..., krausind)
  end
end

function gate(::GateName"amplitude_damping", st::SiteType"Qubit", s::Index...; kwargs...)
  return gate(GateName("AD"), st, s...; kwargs...)
end

is_single_qubit_noise(::GateName"AD") = true
is_single_qubit_noise(::GateName"amplitude_damping") = true

function gate(::GateName"PD", ::SiteType"Qubit", s::Index...; γ::Real)
  @ignore_derivatives begin
    dims = dim.(s)
    N = length(dims)
    @assert all(dims .== 2)
    kraus = zeros(2, 2, 2)
    kraus[:, :, 1] = [
      1 0
      0 sqrt(1 - γ)
    ]
    kraus[:, :, 2] = [
      0 0
      0 sqrt(γ)
    ]

    krausind = Index(size(kraus, 3); tags="kraus")
    return ITensors.itensor(kraus, prime.(s)..., ITensors.dag.(s)..., krausind)
  end
end

function gate(::GateName"phase_damping", st::SiteType"Qubit", s::Index...; kwargs...)
  return gate(GateName("PD"), st, s...; kwargs...)
end
function gate(::GateName"dephasing", st::SiteType"Qubit", s::Index...; kwargs...)
  return gate(GateName("PD"), st, s...; kwargs...)
end

is_single_qubit_noise(::GateName"PD") = true
is_single_qubit_noise(::GateName"phase_damping") = true
is_single_qubit_noise(::GateName"dephasing") = true

function insertnoise(circuit::Vector{<:Vector{<:Any}}, noisemodel::Tuple; gate=nothing)
  max_g_size = maxgatesize(circuit)
  numqubits = nqubits(circuit)

  # single noise model for all
  if noisemodel[1] isa String
    tmp = []
    for k in 1:max_g_size
      tmp = vcat(tmp, k => noisemodel)
    end
    noisemodel = Tuple(tmp)
  end
  noisycircuit = []
  for layer in circuit
    noisylayer = []
    for g in layer
      noisylayer = vcat(noisylayer, [g])
      applynoise = (
        if isnothing(gate)
          true
        elseif gate isa String
          g[1] == gate
        else
          g[1] in gate
        end
      )
      if applynoise
        nq = g[2]
        # n-qubit gate
        if nq isa Tuple
          gatenoiseindex = findfirst(x -> x == length(nq), first.(noisemodel))
          isnothing(gatenoiseindex) &&
            error("Noise model not defined for $(length(nq))-qubit gates!")
          gatenoise = last(noisemodel[gatenoiseindex])
          noisecheck = is_single_qubit_noise(GateName(gatenoise[1]))
          if length(nq) > 1 && is_single_qubit_noise(GateName(gatenoise[1]))
            @ignore_derivatives @warn "Noise model not defined for $(length(nq))-qubit gates! Applying tensor-product noise instead."
            for j in nq
              noisylayer = vcat(noisylayer, [(gatenoise[1], j, gatenoise[2])])
            end
          else
            noisylayer = vcat(noisylayer, [(gatenoise[1], nq, gatenoise[2])])
          end
          # 1-qubit gate
        else
          gatenoiseindex = findfirst(x -> x == 1, first.(noisemodel))
          isnothing(gatenoiseindex) && error("Noise model not defined for 1-qubit gates!")
          gatenoise = last(noisemodel[gatenoiseindex])
          noisylayer = vcat(noisylayer, [(gatenoise[1], nq, gatenoise[2])])
        end
      end
    end
    noisycircuit = vcat(noisycircuit, [noisylayer])
  end
  return noisycircuit
end

function insertnoise(circuit::Vector{<:Any}, noisemodel::Tuple; kwargs...)
  return insertnoise([circuit], noisemodel; kwargs...)[1]
end

function maxgatesize(circuit::Vector{<:Vector{<:Any}})
  maxsize = 0
  for layer in circuit
    for g in layer
      maxsize = length(g[2]) > maxsize ? length(g[2]) : maxsize
    end
  end
  return maxsize
end

maxgatesize(circuit::Vector{<:Any}) = maxgatesize([circuit])
