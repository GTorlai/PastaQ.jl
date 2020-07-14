
# TODO: Make the constructor
# IndexSet{N, IndexT, DataT}(::Index...)

# TODO: export this. Maybe make this more
# specific than an alias?
const filterinds = inds

#
# Order singleton type
#

@eval struct Order{x}
  (OrderT::Type{ <: Order})() = $(Expr(:new, :OrderT))
end

Order(x) = Order{x}()

"""
    IndexSet(::Function, ::Order{N})

Construct an IndexSet of length N from a function that accepts
an integer between 1:N and returns an Index.

# Examples
```julia
IndexSet(n -> Index(1, "i\$n"), Order(4))
```
"""
ITensors.IndexSet(f::Function, N::Int) =
  IndexSet(ntuple(f, N))

ITensors.IndexSet(f::Function, ::Order{N}) where {N} =
  IndexSet(ntuple(f, Val(N)))

# TODO: make IndexSet{N} type stable for element type
# and storage type.
Base.filter(O::Order{N},
            f::Function,
            is::IndexSet) where {N} = IndexSet{N}(filter(f,
                                                         Tuple(is)))

Base.filter(O::Order,
            is::IndexSet,
            args...; kwargs...) = filter(O,
                                         ITensors.fmatch(args...;
                                                         kwargs...),
                                         is)

# intersect
ITensors.commoninds(::Order{N},
           A...;
           kwargs...) where {N} =
  IndexSet{N}(intersect(ITensors.itensor2inds.(A)...;
                        kwargs...)...)

# symdiff
ITensors.noncommoninds(::Order{N},
              A...;
              kwargs...) where {N} =
  IndexSet{N}(symdiff(ITensors.itensor2inds.(A)...;
                      kwargs...)...)

# setdiff
ITensors.uniqueinds(::Order{N},
           A...;
           kwargs...) where {N} =
  IndexSet{N}(setdiff(ITensors.itensor2inds.(A)...;
                      kwargs...)...)

# union
ITensors.unioninds(::Order{N},
          A...;
          kwargs...) where {N} =
  IndexSet{N}(union(ITensors.itensor2inds.(A)...;
                    kwargs...)...)

