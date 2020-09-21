function loadtrainingdataQST(input_path::String;ismpo::Bool=false)
  data_file = h5open(input_path,"r")
  data = read(data_file,"data")
  target = (ismpo ? read(data_file,"psi",MPO) : read(data_file,"psi",MPS))
  return data,target
end

function loadtrainingdataQPT(input_path::String;ismpo::Bool=false)
  data_file = h5open(input_path,"r")
  data_in = read(data_file,"data_in")
  data_out = read(data_file,"data_out")
  target = (ismpo ? read(data_file,"choi",MPO) : read(data_file,"choi",MPS))
  return data_in,data_out,target
end

function fullvector(M::MPS; reverse::Bool = true)
  s = siteinds(M)
  if reverse
    s = Base.reverse(s)
  end
  C = combiner(s...)
  Mvec = prod(M) * dag(C)
  return array(Mvec)
end

function fullmatrix(M::MPO; reverse::Bool = true)
  s = firstsiteinds(M; plev = 0)
  if reverse
    s = Base.reverse(s)
  end
  C = combiner(s...)
  Mmat = prod(M) * dag(C) * C'
  c = combinedind(C)
  return array(permute(Mmat, c', c))
end

