using ITensors
using PastaQ
using .ITensorsGateEvolution

function main()

N = 10

osRand = ProductOps()
for n in 1:N
  osRand *= ("rand", n)
end

osX = ProductOps()
for n in 1:N
  osX *= ("X", n)
end

osZ = ProductOps()
for n in 1:N
  osZ *= ("Z", n)
end

osSw = ProductOps()
for n in 1:N-1
  osSw *= ("Sw", n, n+1)
end

osCx = ProductOps()
for n in 1:N-1
  osCx *= ("Cx", n, n+1)
end

osRand = ProductOps()
for n in 1:N-1
  osRand *= ("rand", n, n+1)
end

osT = ProductOps()
for n in 1:N-4
  osT *= ("T", n, n+2, n+4)
end

os = osRand * osX * osSw * osZ * osCx * osT

@show os

s = siteinds("qubit", N)
gates = ops(s, os)

ψ0 = productMPS(s, "0")

# Apply the gates
ψ = apply(gates, ψ0; cutoff = 1e-15, maxdim = 100)
@show dim(s[1])^(N ÷ 2)
@show maxlinkdim(ψ)

prodψ = apply(gates, prod(ψ0))
@show prod(ψ) ≈ prodψ
@show norm(prod(ψ) - prodψ) / norm(prodψ)

return ψ

end

ψ = main();

