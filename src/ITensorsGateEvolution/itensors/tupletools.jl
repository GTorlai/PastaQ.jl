
"""
    diff(t::Tuple)

For a tuple of length N, return a tuple of length N-1
where element i is t[i+1] - t[i].
"""
function Base.diff(t::NTuple{N}) where N
  return ntuple(i -> t[i+1] - t[i], Val(N-1))
end

