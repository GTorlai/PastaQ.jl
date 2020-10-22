
#
# Functions defined for ITensors.jl objects
# that may be moved to ITensors.jl
#

function sqrt(ρ::ITensor)
  D, U = eigen(ρ)
  sqrtD = D
  sqrtD .= sqrt.(D)
  return U' * sqrtD * dag(U)
end

