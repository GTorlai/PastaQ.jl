function FullVector(mps::MPS)
  if length(mps) == 1
    return mps
  else
    vector = mps[1] * mps[2]
    C = combiner(inds(vector,tags="n=1")[1],inds(vector,tags="n=2")[1],tags="comb")
    vector = vector * C
    for j in 3:length(mps)
      vector = vector * mps[j]
      C = combiner(inds(vector,tags="comb")[1],inds(vector,tags="n=$j")[1],tags="comb")
      vector = vector * C
    end
    return vector
  end
end


#function PrintState(mps::MPS)
#  psi_vec = FullVector(mps)
#  psi_full = psi_vec.store
#  for k in 1:length(psi_vec)
#    println(psi_full[k])
#  end
#end

#function FullMatrix(mpo::MPO)
#  matrix = mpo[1] * mpo[2]
#  Cb = combiner(inds(matrix,tags="n=1",plev=0)[1],inds(matrix,tags="n=2",plev=0)[1],tags="bra")
#  Ck = combiner(inds(matrix,tags="n=1",plev=1)[1],inds(matrix,tags="n=2",plev=1)[1],tags="ket")
#  matrix = matrix * Cb * Ck
#  for j in 3:length(mpo)
#    matrix = matrix * mpo[j]
#    Cb = combiner(inds(matrix,tags="bra")[1],inds(matrix,tags="n=$j",plev=0)[1],tags="bra")
#    Ck = combiner(inds(matrix,tags="ket")[1],inds(matrix,tags="n=$j",plev=1)[1],tags="ket")
#    matrix = matrix * Cb * Ck
#  end
#  return matrix
#end
#

