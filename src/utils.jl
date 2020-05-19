function fullvector(mps::MPS;order="native")
  if length(mps) == 1
    return mps[1]
  else
    N = length(mps)
    if (order == "native")
      vector = mps[1] * mps[2]
      C = combiner(firstind(vector,tags="n=1"),firstind(vector,tags="n=2"),tags="comb")
      vector = vector * C
      for j in 3:length(N)
        vector = vector * mps[j]
        C = combiner(firstind(vector,tags="comb"),firstind(vector,tags="n=$j"),tags="comb")
        vector = vector * C
      end
    else
      vector = mps[N] * mps[N-1]
      C = combiner(firstind(vector,tags="n=$N"),firstind(vector,tags="n=$(N-1)"),tags="comb")
      vector = vector * C
      for j in reverse(1:N-2)
        vector = vector * mps[j]
        C = combiner(firstind(vector,tags="comb"),firstind(vector,tags="n=$j"),tags="comb")
        vector = vector * C
      end
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

