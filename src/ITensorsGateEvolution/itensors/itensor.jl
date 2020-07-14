
#TODO: Make a constructor
#  itensor(::Vector{Pair{Block, Array}}, ::IndexSet)
#for example:
#  itensor([Block(1,1) => randn(2,2),
#           Block(2,2) => randn(3,3)], i', dag(i)))
#to set nonzero blocks.

"""
    itensor(::Array, ::QNIndexSet)

Create a block sparse ITensor from the input Array, where zeros are
dropped and nonzero blocks are determined about the zero values of
the array.
"""
function ITensors.itensor(A::Array{ElT},
                          inds::ITensors.QNIndexSet) where {ElT <: Number}
  length(A) ≠ dim(inds) && throw(DimensionMismatch("In ITensor(::Array, ::IndexSet), length of Array ($(length(A))) must match total dimension of IndexSet ($(dim(inds)))"))
  T = emptyITensor(ElT, inds)
  A = reshape(A, dims(inds))
  for vs in eachindex(T)
    Avs = A[vs]
    if !iszero(Avs)
      T[vs] = A[vs]
    end
  end
  return T
end

ITensors.itensor(A::Array{<:Number},
                 inds::ITensors.QNIndex...) =
  itensor(A, IndexSet(inds...))

