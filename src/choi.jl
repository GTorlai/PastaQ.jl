
struct Choi{MPOT <: Union{MPO, LPDO}}
  M::MPOT
end

Base.length(C::Choi) = length(C.M)

Base.copy(C::Choi) = Choi(copy(C.M))

function Base.getindex(C::Choi, args...)
  error("getindex(C::Choi, args...) is purposefully not implemented yet.")
end

function Base.setindex!(C::Choi, args...)
  error("setindex!(C::Choi, args...) is purposefully not implemented yet.")
end


