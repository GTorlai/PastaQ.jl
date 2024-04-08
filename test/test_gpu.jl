using PastaQ
using CUDA
using Test

const eltypes = (nothing, Float32, Float64, ComplexF32, ComplexF64)

const devices = (
  identity,
  #cu, # Can't test right now
)

const full_representations = (false, true)

const noises = (nothing, 
                ("amplitude_damping", (γ=0.1,)),
                ("phase_damping", (γ=0.1,))
              )

const processes = (false, true)

@testset "runcircuit with eltype $eltype, device $device, full_representation $full_representation, noise $noise" for eltype in eltypes,
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

# Added a new test set for depolarizing noise. Since the

@testset "runcircuit with eltype $eltype, device $device, full_representation $full_representation, noise depolarizing" for eltype in (nothing, ComplexF32, ComplexF64),
  device in devices,
  full_representation in full_representations,
  process in processes

  ψ = runcircuit([("X", 1)]; eltype, device, full_representation, noise=("depolarizing",(p=0.1,)), process)
  if isnothing(eltype)
    @test Base.eltype(ψ[1]) === ComplexF64
  else
    @test Base.eltype(ψ[1]) === eltype
  end
end