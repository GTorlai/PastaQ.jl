
# Version that takes `op("S=1/2", "X", Index(2, "i"))`.
# Shortcuts searching for the correct tag, so is a bit faster.
ITensors.op(on::AbstractString,
            st::AbstractString,
            s::Index...; kwargs...) = op(OpName(on), SiteType(st), s...; kwargs...)

