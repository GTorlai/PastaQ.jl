#
# qubit
#

function ITensors.space(::SiteType"qubit";
                        conserve_qns::Bool = false)
  if conserve_qns
    return [QN() => 2]
  end
  return 2
end

ITensors.state(::SiteType"qubit",
               ::ITensors.StateName"0") = 1

ITensors.state(::SiteType"qubit",
               ::ITensors.StateName"1") = 2

ITensors.op(::OpName"Id",
            ::SiteType"qubit",
            s::Index) =
  itensor([1 0; 0 1], s', dag(s))

ITensors.op(::OpName"X",
            ::SiteType"qubit",
            s::Index) = 
  itensor([0.0 1.0; 1.0 0.0], s', dag(s))

ITensors.op(::OpName"iY",
            ::SiteType"qubit",
            s::Index) = 
  itensor([0 1; -1 0], s', dag(s))

ITensors.op(::OpName"Y",
            ::SiteType"qubit",
            s::Index) = 
  itensor([0 -im; im 0], s', dag(s))

ITensors.op(::OpName"Z",
            ::SiteType"qubit",
            s::Index) = 
  itensor([1 0; 0 -1], s', dag(s))

ITensors.op(::OpName"H",
            ::SiteType"qubit",
            s::Index) = 
  itensor([1/sqrt(2) 1/sqrt(2);
           1/sqrt(2) -1/sqrt(2)], s', dag(s))

ITensors.op(::OpName"Rx",
            ::SiteType"qubit",
            s::Index; θ::Number) =
  itensor([cos(θ/2)      -im*sin(θ/2.);
           -im*sin(θ/2.) cos(θ/2.)], s', dag(s))

ITensors.op(::OpName"Sw",
            ::SiteType"qubit",
            s1::Index,
            s2::Index) =
  itensor([1 0 0 0;
           0 0 1 0;
           0 1 0 0;
           0 0 0 1],
          s1', s2',
          dag(s1), dag(s2))

ITensors.op(::OpName"rand",
            ::SiteType"qubit",
            s::Index...) =
  randomITensor(prime.(s)...,
                dag.(s)...)

ITensors.op(::OpName"Cx",
            ::SiteType"qubit",
            s1::Index,
            s2::Index) =
  itensor([1 0 0 0;
           0 1 0 0;
           0 0 0 1;
           0 0 1 0],
          s1', s2',
          dag(s1), dag(s2))

ITensors.op(::OpName"T",
            ::SiteType"qubit",
            s1::Index,
            s2::Index,
            s3::Index) =
  itensor([1 0 0 0 0 0 0 0;
           0 1 0 0 0 0 0 0;
           0 0 1 0 0 0 0 0;
           0 0 0 1 0 0 0 0;
           0 0 0 0 1 0 0 0;
           0 0 0 0 0 1 0 0;
           0 0 0 0 0 0 0 1;
           0 0 0 0 0 0 1 0],
          s1', s2', s3',
          dag(s1), dag(s2), dag(s3))

ITensors.op(::OpName"noise",
            ::SiteType"qubit",
            s::Index...;
            krausind = hasqns(s[1]) ? Index([QN() => 2], "kraus") : Index(2, "kraus")) =
  randomITensor(prime.(s)..., dag.(s)..., krausind)

