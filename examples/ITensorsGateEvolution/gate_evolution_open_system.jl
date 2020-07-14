using ITensors
using PastaQ
using .ITensorsGateEvolution

function main()

N = 8

osX = ProductOps()
for n in 1:N
  osX *= "X", n
end

osZ = ProductOps()
for n in 1:N
  osZ *= "Z", n
end

osSw = ProductOps()
for n in 1:N-2
  osSw *= "Sw", n, n+2
end

osCx = ProductOps()
for n in 1:N-3
  osCx *= "Cx", n, n+3
end

osT = ProductOps()
for n in 1:N-3
  osT *= "T", n, n+1, n+3
end

osRx = ProductOps()
for n in 1:N
  osRx *= "Rx", n, (θ = π,)
end

os_noise = ProductOps()
for n in 1:N-4
  os_noise *= "noise", n, n+2, n+4
end

os = osX * osSw * osRx * osZ * osCx * osT * os_noise

#os = ProductOps()
#os *= "T", 1, 3, 4

@show os

s = siteinds("qubit", N; conserve_qns = false)
gates = ops(s, os)

M0 = MPO(s, "Id")

# Apply the gates

s0 = siteinds(M0)

M = apply(gates, M0; apply_dag = true, cutoff = 1e-15, maxdim = 500)
@show dim(s[1])^(N ÷ 2)
@show maxlinkdim(M)

s = siteinds(M)
for n in 1:N
  @assert hassameinds(s[n], s0[n])
end

println("Evolution complete, test the result")

set_warn_itensor_order!(18)

prodM = apply(gates, prod(M0); apply_dag = true)

@show prod(M) ≈ prodM
@show norm(prod(M) - prodM) / norm(prodM)

reset_warn_itensor_order!()

return M

end

M = main();

