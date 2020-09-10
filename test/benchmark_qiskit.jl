using PastaQ
using PyCall
pickle = pyimport("pickle");
using LinearAlgebra
using ITensors
using Test

function convert_gates(g)
  gates = Tuple[]
  for gate in g
    if (gate[1] == "U3")
      newsite = gate[2]+1
      push!(gates,("Rn",newsite,(θ = gate[3][1],ϕ = gate[3][2],λ = gate[3][3])))
    else
      newsite = (gate[2][1]+1,gate[2][2]+1)
      push!(gates,("CX",newsite))
    end
  end
  return gates
end

@testset " Unitary circuit matrix " begin
  N = 5
  path = string("test_data_unitary.pickle")
  f_in = open(path)
  testdata = pickle.load(f_in);
  exact_U = testdata["U"]
  g = testdata["gates"]
  gates = convert_gates(g)
  U0 = circuit(N)
  U = fullmatrix(runcircuit(N,gates,process=true))
  @test U ≈ exact_U
end

@testset " Noiseless Choi matrix " begin
  N = 5
  path = string("test_data_unitary.pickle")
  f_in = open(path)
  testdata = pickle.load(f_in);
  exact_choi = testdata["choi"]
  g = testdata["gates"]
  gates = convert_gates(g)

  Λ0 = choimatrix(N,gates)
  disable_warn_order!()
  Λ = fullmatrix(MPO(Λ0))
  reset_warn_order!()

  @test Λ ≈ exact_choi
end

@testset " Noisy Choi matrix " begin
  N = 5
  path = string("test_data_AD_0.1.pickle")
  f_in = open(path)
  testdata = pickle.load(f_in);
  exact_choi = testdata["choi"]
  g = testdata["gates"]
  gates = convert_gates(g)
  #Λ0 = runcircuit(N,gates,process=true,noise="AD",γ=0.1)
  Λ0 = choimatrix(N,gates;noise="AD",γ=0.1)
  disable_warn_order!()
  Λ = fullmatrix(Λ0)
  reset_warn_order!()
  @test Λ ≈ exact_choi

  path = string("test_data_PD_0.1.pickle")
  f_in = open(path)
  testdata = pickle.load(f_in);
  exact_choi = testdata["choi"]
  g = testdata["gates"]
  gates = convert_gates(g)
  #Λ0 = runcircuit(N,gates,process=true,noise="PD",γ=0.1)
  Λ0 = choimatrix(N,gates;noise="PD",γ=0.1)
  disable_warn_order!()
  Λ = fullmatrix(Λ0)
  reset_warn_order!()
  @test Λ ≈ exact_choi
  
  #path = string("test_data_DEP_0.1.pickle")
  #f_in = open(path)
  #testdata = pickle.load(f_in);
  #exact_choi = testdata["choi"]
  #g = testdata["gates"]
  #gates = convert_gates(g)
  #Λ0 = runcircuit(N,gates,process=true,noise="DEP",p=0.1)
  #disable_warn_order!()
  #Λ = fullmatrix(Λ0)
  #reset_warn_order!()
  #@test Λ ≈ exact_choi

end

