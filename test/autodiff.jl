using PastaQ
using ITensors
using Test
using LinearAlgebra
using Random

@testset "one-qubit gate gradients" begin 
  Random.seed!(1234)
  N = 1
  ϵ = 1E-6
  
  q = qubits(N)
  i = q[1]
  ψ = productstate(q)
  
  # Rx
  θ = π/3
  for σ′ in [0,1]
    for σ in [0,1]
      bra = state(i,σ′+1) 
      ket = state(i,σ+1)
      function loss(θ)
        u = gate(ψ,"Rx",1, (θ = θ,))
        return abs((bra' * u * ket)[])
      end
      ∇num = (loss(θ+ϵ) - loss(θ))/(ϵ)
      @test ∇num ≈ loss'(θ) atol = 1e-5
    end
  end
  
  # Ry
  θ = π/3
  for σ′ in [0,1]
    for σ in [0,1]
      bra = state(i,σ′+1) 
      ket = state(i,σ+1)
      function loss(θ)
        u = gate(ψ,"Ry",1, (θ = θ,))
        return abs((bra' * u * ket)[])
      end
      ∇num = (loss(θ+ϵ) - loss(θ))/(ϵ)
      @test ∇num ≈ loss'(θ) atol = 1e-5
    end
  end
  
  # Rz
  ϕ = π/3
  for σ′ in [0,1]
    for σ in [0,1]
      bra = state(i,σ′+1) 
      ket = state(i,σ+1)
      function loss(ϕ)
        u = gate(ψ,"Rz",1, (ϕ = ϕ,))
        return abs((bra' * u * ket)[])
      end
      ∇num = (loss(ϕ+ϵ) - loss(ϕ))/(ϵ)
      @test ∇num ≈ loss'(ϕ) atol = 1e-5
    end
  end
  
  
  # Rn
  # XXX: wrong gradients, factor of two in the grad against θ
  θ = π/5
  ϕ = 0.13
  λ = 0.34
  for σ′ in [0,1]
    for σ in [0,1]
      bra = state(i,σ′+1) 
      ket = state(i,σ+1)
      function loss(pars)
        θ, ϕ, λ = pars
        u = gate(ψ,"Rn", 1, (θ = θ, ϕ = ϕ, λ = λ))
        return imag((bra' * u * ket)[])
      end
      p  = [θ,ϕ,λ]
      p₊ = [θ+ϵ, ϕ, λ]
      ∇θ = (loss(p₊) - loss(p))/ϵ
      p₊ = [θ, ϕ+ϵ, λ]
      ∇ϕ = (loss(p₊) - loss(p))/ϵ
      p₊ = [θ, ϕ, λ+ϵ]
      ∇λ = (loss(p₊) - loss(p))/ϵ
      
      ∇ = loss'([θ,ϕ,λ])
      @test ∇θ ≈ ∇[1] atol = 1e-5  
      @test ∇ϕ ≈ ∇[2] atol = 1e-5
      @test ∇λ ≈ ∇[3] atol = 1e-5
    end
  end
  
end



@testset "two-qubit gate gradients" begin 
  Random.seed!(1234)
  N = 2
  ϵ = 1E-8
  basis = [[0,0],[0,1],[1,0],[1,1]]
  
  q = qubits(N)
  ψ = productstate(q)
  
  
  # CRy
  θ = π/2
  # sweep the matrix row by row
  for σ′ in basis
    for σ in basis
      bra = state(q[1],σ′[1]+1) * state(q[2], σ′[2]+1)
      ket = state(q[1],σ[1]+1)  * state(q[2], σ[2]+1)
      function loss(θ)
        U = gate(ψ,"CRy",(1,2), (θ = θ, f = x -> x))
        return (bra' * U * ket)[]
      end
      ∇num = (loss(θ+ϵ) - loss(θ-ϵ))/(2ϵ)
      @test ∇num ≈ loss'(θ)
    end
  end
  
  # CRz
  ϕ = π/3
  # sweep the matrix row by row
  for σ′ in basis
    for σ in basis
      bra = state(q[1],σ′[1]+1) * state(q[2], σ′[2]+1)
      ket = state(q[1],σ[1]+1)  * state(q[2], σ[2]+1)
      function loss(ϕ)
        U = gate(ψ,"CRz",(2,1), (ϕ = ϕ,))
        return abs((bra' * U * ket)[])
      end
      ∇num = (loss(ϕ+ϵ) - loss(ϕ-ϵ))/(2ϵ)
      @test ∇num ≈ loss'(ϕ)
    end
  end
  # CRy
  θ = π/2
  # sweep the matrix row by row
  for σ′ in basis
    for σ in basis
      bra = state(q[1],σ′[1]+1) * state(q[2], σ′[2]+1)
      ket = state(q[1],σ[1]+1)  * state(q[2], σ[2]+1)
      function loss(θ)
        U = gate(ψ,"CRy",(2,1), (θ = θ, f = x -> x))
        return (bra' * U * ket)[]
      end
      ∇num = (loss(θ+ϵ) - loss(θ-ϵ))/(2ϵ)
      @test ∇num ≈ loss'(θ)
    end
  end
  
  # CRz
  ϕ = π/3
  # sweep the matrix row by row
  for σ′ in basis
    for σ in basis
      bra = state(q[1],σ′[1]+1) * state(q[2], σ′[2]+1)
      ket = state(q[1],σ[1]+1)  * state(q[2], σ[2]+1)
      function loss(ϕ)
        U = gate(ψ,"CRz",(1,2), (ϕ = ϕ,))
        return abs((bra' * U * ket)[])
      end
      ∇num = (loss(ϕ+ϵ) - loss(ϕ-ϵ))/(2ϵ)
      @test ∇num ≈ loss'(ϕ)
    end
  end
  
  # Rn
  θ = π/5
  ϕ = π/3
  λ = π/7
  for σ′ in basis
    for σ in basis
      bra = state(q[1],σ′[1]+1) * state(q[2], σ′[2]+1)
      ket = state(q[1],σ[1]+1)  * state(q[2], σ[2]+1)
      function loss(pars)
        θ, ϕ, λ=pars
        u = gate(ψ,"CRn", (1,2), (θ = θ, ϕ = ϕ, λ = λ))
        return abs((bra' * u * ket)[])
      end
      p  = [θ,ϕ,λ]
      p₊ = [θ+ϵ, ϕ, λ]
      ∇θ = (loss(p₊) - loss(p))/ϵ
      p₊ = [θ, ϕ+ϵ, λ]
      ∇ϕ = (loss(p₊) - loss(p))/ϵ
      p₊ = [θ, ϕ, λ+ϵ]
      ∇λ = (loss(p₊) - loss(p))/ϵ
      
      ∇ = loss'([θ,ϕ,λ])
      @test ∇θ ≈ ∇[1] 
      @test ∇ϕ ≈ ∇[2] 
      @test ∇λ ≈ ∇[3] 
    end
  end
  
  # Rn
  θ = π/5
  ϕ = π/3
  λ = π/7
  for σ′ in basis
    for σ in basis
      bra = state(q[1],σ′[1]+1) * state(q[2], σ′[2]+1)
      ket = state(q[1],σ[1]+1)  * state(q[2], σ[2]+1)
      function loss(pars)
        θ, ϕ, λ=pars
        u = gate(ψ,"CRn", (2,1), (θ = θ, ϕ = ϕ, λ = λ))
        return abs((bra' * u * ket)[])
      end
      p  = [θ,ϕ,λ]
      p₊ = [θ+ϵ, ϕ, λ]
      ∇θ = (loss(p₊) - loss(p))/ϵ
      p₊ = [θ, ϕ+ϵ, λ]
      ∇ϕ = (loss(p₊) - loss(p))/ϵ
      p₊ = [θ, ϕ, λ+ϵ]
      ∇λ = (loss(p₊) - loss(p))/ϵ
      
      ∇ = loss'([θ,ϕ,λ])
      @test ∇θ ≈ ∇[1] 
      @test ∇ϕ ≈ ∇[2] 
      @test ∇λ ≈ ∇[3] 
    end
  end
  
  # Rxx
  ϕ = π/3
  # sweep the matrix row by row
  for σ′ in basis
    for σ in basis
      bra = state(q[1],σ′[1]+1) * state(q[2], σ′[2]+1)
      ket = state(q[1],σ[1]+1)  * state(q[2], σ[2]+1)
      function loss(ϕ)
        U = gate(ψ,"Rxx",(1,2), (ϕ = ϕ,))
        return abs((bra' * U * ket)[])
      end
      ∇num = (loss(ϕ+ϵ) - loss(ϕ-ϵ))/(2ϵ)
      @test ∇num ≈ loss'(ϕ)
    end
  end

end









@testset "functions applied to qudit gates" begin 
  Random.seed!(1234)
  N = 2
  ϵ = 1E-8
  q = qudits(N; dim = 3)
  ψ = productstate(q)
  basis = [0,1,2]
  
  g(θ) = θ^2
  
  θ = π/3
  i = q[1]
  for σ′ in basis
    for σ in basis
      bra = state(i,σ′+1)
      ket = state(i,σ+1)
      function loss(θ)
        u = gate(ψ, "a", 1, (f = x ->  exp(im * g(θ) * x),))
        return abs((bra' * u * ket)[])
      end
      ∇num = (loss(θ+ϵ) - loss(θ))/ϵ
      @test ∇num ≈ loss'(θ) atol = 1e-5
    end
  end

  θ = π/3
  i = q[1]
  for σ′ in basis
    for σ in basis
      bra = state(i,σ′+1)
      ket = state(i,σ+1)
      function loss(θ)
        u = gate(ψ, "a† * a", 1, (f = x ->  exp(im * g(θ) * x),))
        return abs((bra' * u * ket)[])
      end
      ∇num = (loss(θ+ϵ) - loss(θ))/ϵ
      @test ∇num ≈ loss'(θ) atol = 1e-5
    end
  end
  
  θ = π/3
  i = q[1]
  for σ′ in basis
    for σ in basis
      bra = state(i,σ′+1)
      ket = state(i,σ+1)
      function loss(θ)
        u = gate(ψ, "a† * a + a * a†", 1, (f = x ->  exp(im * g(θ) * x),))
        return abs((bra' * u * ket)[])
      end
      ∇num = (loss(θ+ϵ) - loss(θ))/ϵ
      @test ∇num ≈ loss'(θ) atol = 1e-5
    end
  end

  basis = vec(Iterators.product(fill([0,1,2],2)...)|>collect)
  θ = π/3
  i = q[1]
  for σ′ in basis
    for σ in basis
      bra = state(q[1],σ′[1]+1) * state(q[2], σ′[2]+1)
      ket = state(q[1],σ[1]+1)  * state(q[2], σ[2]+1)
      function loss(θ)
        u = gate(ψ, "aa†", (1,2), (f = x ->  exp(im * g(θ) * x),))
        return abs((bra' * u * ket)[])
      end
      ∇num = (loss(θ+ϵ) - loss(θ))/ϵ
      @test ∇num ≈ loss'(θ) atol = 1e-5
    end
  end
end


@testset "trotter - 1-qubit gate" begin
  N = 4
  
  function hamiltonian(θ)
    H = Tuple[]
    for j in 1:N
      H = vcat(H, [(θ[j], "X", j)])
    end
    for j in 1:2:N-1
      H = vcat(H, [(θ[j], "CX", (j,j+1))])
    end
    return H
  end

  function variational_circuit(H)
    circ = Tuple[]
    for j in 1:N
      ξ, localop, support = H[j]
      circ = vcat(circ, [(localop, support,(f = x -> exp(im * ξ * x),))])
    end
    return circ
  end
  
  Random.seed!(1234)
  θ = rand(N)
  
  q = qubits(N)
  ψ = productstate(q)
  ϕ = (N == 1) ? productstate(q, [1]) : runcircuit(q, randomcircuit(N; depth = 2))
  
  function loss(θ)
    H = hamiltonian(θ)
    circuit = variational_circuit(H)
    U = buildcircuit(ψ, circuit)
    return -abs2(inner(ϕ, U, ψ))
  end
  
  ∇ad = loss'(θ)
  ϵ = 1e-5
  for k in 1:length(θ)
    θ[k] += ϵ
    f₊ = loss(θ)
    θ[k] -= 2*ϵ
    f₋ = loss(θ)
    ∇num = (f₊ - f₋)/(2ϵ)
    θ[k] += ϵ
    @test ∇ad[k] ≈ ∇num atol = 1e-6 
  end

end



@testset "vqe style optimization: ⟨ψ|U† O U|ψ⟩" begin 
  function Rylayer(N, θ⃗)
    return [("Ry", (n,), (θ=θ⃗[n],)) for n in 1:N]
  end
  function Rxlayer(N, θ⃗)
    return [("Rx", (n,), (θ=θ⃗[n],)) for n in 1:N]
  end
  function CXlayer(N)
    return [("CX", (n, n + 1)) for n in 1:2:(N - 1)]
  end
  
  # The variational circuit we want to optimize
  function variational_circuit(θ⃗)
    N = length(θ⃗)
    return vcat(Rylayer(N, θ⃗),CXlayer(N), Rxlayer(N, θ⃗), Rylayer(N, θ⃗), CXlayer(N), Rxlayer(N, θ⃗))
  end
  
  Random.seed!(1234)
  N = 4
  
  q = qubits(N)
  ψ = productstate(q)
  
  os = OpSum()
  for k in 1:N-1
    os += 1.0, "Z",k,"Z",k+1
    os += 1.0,"X",k
  end
  O = MPO(os,q)
  
  function loss(θ⃗)
    circuit = variational_circuit(θ⃗)
    U = buildcircuit(ψ, circuit)
    return inner(O, U, ψ)
  end
  θ⃗ = 2π .* rand(N)
  
  ∇ad = loss'(θ⃗)
  
  ϵ = 1e-5
  for k in 1:length(θ⃗)
    θ⃗[k] += ϵ
    f₊ = loss(θ⃗) 
    θ⃗[k] -= 2*ϵ
    f₋ = loss(θ⃗) 
    ∇num = (f₊ - f₋)/(2ϵ)
    θ⃗[k] += ϵ
    @test ∇ad[k] ≈ ∇num atol = 1e-8 
  end 
end

@testset "fidelity optimization: 1-qubit & 2-qubit gates" begin 
  function Rylayer(N, θ⃗)
    return [("Ry", (n,), (θ=θ⃗[n],)) for n in 1:N]
  end
  
  function RXXlayer(N, ϕ⃗)
    return [("RXX", (n, n + 1), (ϕ = ϕ⃗[n],)) for n in 1:(N÷2)]
  end
  
  # The variational circuit we want to optimize
  function variational_circuit(θ⃗,ϕ⃗)
    N = length(θ⃗)
    return vcat(Rylayer(N, θ⃗),RXXlayer(N,ϕ⃗), Rylayer(N, θ⃗), RXXlayer(N,ϕ⃗))
  end
  
  Random.seed!(1234)
  N = 8
  θ⃗ = 2π .* rand(N)
  ϕ⃗ = 2π .* rand(N÷2)
  circuit = variational_circuit(θ⃗,ϕ⃗)
  
  q = qubits(N)
  ψ = productstate(q)
  ϕ = runcircuit(q, randomcircuit(N; depth = 2))
  
  function loss(pars)
    θ⃗,ϕ⃗ = pars
    circuit = variational_circuit(θ⃗,ϕ⃗)
    U = buildcircuit(ψ, circuit)
    return -abs2(inner(ϕ, U, ψ))
  end
  
  θ⃗ = randn!(θ⃗)
  ϕ⃗ = randn!(ϕ⃗)
  ∇ad = loss'([θ⃗,ϕ⃗])
  
  ϵ = 1e-5
  for k in 1:length(θ⃗)
    θ⃗[k] += ϵ
    f₊ = loss([θ⃗,ϕ⃗]) 
    θ⃗[k] -= 2*ϵ
    f₋ = loss([θ⃗,ϕ⃗]) 
    ∇num = (f₊ - f₋)/(2ϵ)
    θ⃗[k] += ϵ
    @test ∇ad[1][k] ≈ ∇num atol = 1e-8 
  end 
  for k in 1:length(ϕ⃗)
    ϕ⃗[k] += ϵ
    f₊ = loss([θ⃗,ϕ⃗]) 
    ϕ⃗[k] -= 2*ϵ
    f₋ = loss([θ⃗,ϕ⃗]) 
    ∇num = (f₊ - f₋)/(2ϵ)
    ϕ⃗[k] += ϵ
    @test ∇ad[2][k] ≈ ∇num atol = 1e-8 
  end 
end


