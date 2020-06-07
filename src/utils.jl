function fullvector(mps::MPS;order="reverse")
  if length(mps) == 1
    return array(mps[1])
  else
    N = length(mps)
    if order == "reverse"
      vector = mps[N] * mps[N-1]
      C = combiner(firstind(vector,tags="n=$N"),firstind(vector,tags="n=$(N-1)"),tags="comb")
      vector = vector * C
      for j in reverse(1:N-2)
        vector = vector * mps[j]
        C = combiner(firstind(vector,tags="comb"),firstind(vector,tags="n=$j"),tags="comb")
        vector = vector * C
      end
    elseif order == "native"
      vector = mps[1] * mps[2]
      C = combiner(firstind(vector,tags="n=1"),firstind(vector,tags="n=2"),tags="comb")
      vector = vector * C
      for j in 3:N
        vector = vector * mps[j]
        C = combiner(firstind(vector,tags="comb"),firstind(vector,tags="n=$j"),tags="comb")
        vector = vector * C
      end
    end
    return array(vector)
  end
end

function fullmatrix(tensor::ITensor;order="reverse")
  N = Int(length(inds(tensor))/2)
  if ( N == 1)
    return array(tensor)
  else
    indices = inds(tensor,plev=0)
    if order == "reverse"
      Cb = combiner(prime(indices[N]),prime(indices[N-1]),tags="bra")
      Ck = combiner(indices[N],indices[N-1],tags="ket")
      matrix = tensor * Cb * Ck
      for j in reverse(1:N-2)
        Cb = combiner(firstind(matrix,tags="bra"),prime(indices[j]))
        Ck = combiner(firstind(matrix,tags="ket"),indices[j])
        matrix = tensor * Cb * Ck
      end
    elseif order == "native"
      Cb = combiner(prime(indices[1]),prime(indices[2]),tags="bra")
      Ck = combiner(indices[1],indices[2],tags="ket")
      matrix = tensor * Cb * Ck
      for j in 3:N
        Cb = combiner(firstind(matrix,tags="bra"),prime(indices[j]))
        Ck = combiner(firstind(matrix,tags="ket"),indices[j])
        matrix = tensor * Cb * Ck
      end
    end
    return array(matrix)
  end
end


function fullmatrix(mpo::MPO;order="reverse")
  if length(mpo) == 1
    return array(mpo[1])
  else
    N = length(mpo)
    if order == "reverse"
      matrix = mpo[N] * mpo[N-1]
      Cb = combiner(inds(matrix,tags="n=$N",plev=0)[1],inds(matrix,tags="n=$(N-1)",plev=0)[1],tags="bra")      
      Ck = combiner(inds(matrix,tags="n=$N",plev=1)[1],inds(matrix,tags="n=$(N-1)",plev=1)[1],tags="ket")      
      matrix = matrix * Cb * Ck
      for j in reverse(1:N-2)
        matrix = matrix * mpo[j]
        Cb = combiner(firstind(matrix,tags="bra"),inds(matrix,tags="n=$j",plev=0)[1],tags="bra")
        Ck = combiner(firstind(matrix,tags="ket"),inds(matrix,tags="n=$j",plev=1)[1],tags="ket")
        matrix = matrix * Cb * Ck
      end
    elseif order == "native"
      #TODO
    #  vector = mps[1] * mps[2]
    #  C = combiner(firstind(vector,tags="n=1"),firstind(vector,tags="n=2"),tags="comb")
    #  vector = vector * C
    #  for j in 3:N
    #    vector = vector * mps[j]
    #    C = combiner(firstind(vector,tags="comb"),firstind(vector,tags="n=$j"),tags="comb")
    #    vector = vector * C
    #  end
    end
    return array(matrix)
  end
end
