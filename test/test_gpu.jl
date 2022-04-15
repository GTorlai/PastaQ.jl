using PastaQ
using ITensorGPU
using Test

const eltypes = (nothing, Float32, Float64, ComplexF32, ComplexF64)

const devices = (
  identity,
  cpu,
  #cu, # Can't test right now
)

const full_representations = (false, true)

const noises = (nothing, ("amplitude_damping", (γ=0.1,)))

const processes = (false, true)

@testset "runcircuit with eltype $eltype, device $device, full_representation $full_representation, noise $noise" for eltype in
                                                                                                                      eltypes,
  device in devices,
  full_representation in full_representations,
  noise in noises,
  process in processes

  ψ = runcircuit([("X", 1)]; eltype, device, full_representation, noise, process)
  if isnothing(eltype)
    @test Base.eltype(ψ[1]) === Float64
  else
    @test Base.eltype(ψ[1]) === eltype
  end
end
