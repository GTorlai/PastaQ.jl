# Quantum gates structure
struct QuantumGates
  Id::ITensor
  X::ITensor
  Y::ITensor
  Z::ITensor
  H::ITensor
  Kp::ITensor
  Km::ITensor
end

function QuantumGates()
  # Index 
  i = Index(2)
  
  # Identity
  Id = ITensor([1 0;0 1],i,i')

  # Pauli matrices
  X = ITensor([0 1;1 0],i,i') 
  Y = ITensor([0 -im;im 0],i,i') 
  Z = ITensor([1 0;0 -1],i,i') 
  
  # Rotation in the X basis (Hadamard)
  H = (1/sqrt(2)) * ITensor([1 1; 1 -1],i,i')
  
  # Rotation in the Y basis
  Kp = (1/sqrt(2)) * ITensor([1 1; im -im],i,i')
  Km = (1/sqrt(2)) * ITensor([1 -im; 1 im],i,i')
  return QuantumGates(Id,X,Y,Z,H,Kp,Km)

end

function U3(θ,ϕ,λ)
  ind = Index(2)
  gate = [cos(θ/2.)                -exp(im*λ) * sin(θ/2.);
          exp(im*ϕ) * sin(θ/2.)    exp(im*(ϕ+λ)) * cos(θ/2.)]
  return ITensor(gate,ind,ind')
end

function cX(sites::Array{Int})#,i::IndexSet)
  i1 = Index(2)
  i2 = Index(2)
  if sites[1] < sites[2]
    gate = reshape([1 0 0 0;
                    0 0 0 1;
                    0 0 1 0;
                    0 1 0 0],(2,2,2,2))
    cx = ITensor(gate,i1'',i2'',i1,i2)
  else
    gate = reshape([1 0 0 0;
                    0 1 0 0;
                    0 0 0 1;
                    0 0 1 0],(2,2,2,2))
    cx = ITensor(gate,i1'',i2'',i1,i2)
  end
  return cx
end

