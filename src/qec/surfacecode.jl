using PastaQ
using ITensors
using Random
using LinearAlgebra
import StatsBase
using Plots

struct SurfaceCode
  d::Int64
  Qcoord::Vector
  Xcoord::Vector
  Zcoord::Vector
  QonS::NamedTuple
  SonQ::Vector
end

"""
Get the qubit index from coordinates
"""
function Q_at(x::Int64, y::Int64, d::Int64)
  (x < 1 || x > 2*d-1) && error("x out of bounds")
  (y < 1 || y > 2*(d)-1) && error("y out of bounds")
  (isodd(x) && isodd(y)) && return (2*d-1)*(y-1)÷2 +(x+1)÷2
  (iseven(x) && iseven(y)) && return (2*d-1)*(y-2)÷2 + x÷2 + d 
  error("No data qubit at localtion (",x,",",y,")")
end

Q_at(coords::Vector) = 
  [Q_at(coord...) for coord in coords] 
 
"""
Get the X stabilizer index from plaquette coordinate
"""
function X_at(x::Int64, y::Int64, d::Int64)
  (x < 2 || x > 2*(d-1)) && error("x out of bounds")
  (y < 1 || y > 2*d-1) && error("y out of bounds")
  (iseven(x) && isodd(y)) && return (d-1)*(y-1)÷2+x÷2 
  error("No X stabilizer at  localtion (",x,",",y,")")
end

"""
Get the Z stabilizer index from vertex coordinate
"""
function Z_at(x::Int64, y::Int64, d::Int64)
  (x < 1 || x > 2*d-1) && error("x out of bounds")
  (y < 1 || y > 2*(d-1)) && error("y out of bounds")
  (isodd(x) && iseven(y)) && return d*(y-2)÷2+x÷2+1 
  error("No X stabilizer at  localtion (",x,",",y,")")
end

"""
Constructor
"""
function SurfaceCode(d::Int64)
  @assert isodd(d)
  Qcoord = []
  Xcoord = []
  Zcoord = []

  # build coordinates of qubits
  for y in 1:2*d-1
    if isodd(y)
      for x in 1:2:2*d 
        push!(Qcoord,[x,y])
      end
    else
      for x in 2:2:2*d-1
        push!(Qcoord,[x,y])
      end
    end
  end
  
  # build coordinates of X stabilizers
  Zcoord = vec(Iterators.product(1:2:2*d-1, 2:2:2*d-1)|>collect) 
  Xcoord = vec(Iterators.product(2:2:2*d-1, 1:2:2*d-1)|>collect)

  stabX = []
  stabZ = []

  # build the X stabilizers
  for y in 1:2:2*d-1
    for x in 2:2:2*(d-1)
      if y == 1
        # lower smooth boundary
        push!(stabX,[Q_at(x-1,y,d), Q_at(x,y+1,d), Q_at(x+1,y,d)]) 
      elseif y == 2*d-1
        # upper smooth boundary
        push!(stabX,[Q_at(x-1,y,d), Q_at(x,y-1,d), Q_at(x+1,y,d)])
      else
        # bulk stabilizers
        push!(stabX,[Q_at(x-1,y,d), Q_at(x,y+1,d), Q_at(x+1,y,d),Q_at(x,y-1,d)])
      end
    end
  end

  # build the Z stabilizers
  for y in 2:2:2*d-1
    for x in 1:2:2*d-1
      if x == 1
        # left rough boundary
        push!(stabZ,[Q_at(x,y-1,d), Q_at(x+1,y,d), Q_at(x,y+1,d)])
      elseif x == 2*d-1
        # right rough boundary
        push!(stabZ,[Q_at(x,y-1,d), Q_at(x-1,y,d), Q_at(x,y+1,d)])
      else
        # bulk stabilizers
        push!(stabZ,[Q_at(x,y-1,d), Q_at(x-1,y,d), Q_at(x,y+1,d), Q_at(x+1,y,d)])
      end
    end
  end

  QonS = (X = stabX, Z = stabZ)
  SonQ = []
  
  # build the stabilizers around each qubit
  # left-bottom corner
  push!(SonQ, (X = [1], Z = [1]))
  # bottom row
  for x in 3:2:2*(d-1)-1
    push!(SonQ, (X = [X_at(x-1,1,d),X_at(x+1,1,d)], Z = [Z_at(x,2,d)]))
  end
  # right-bottom corner
  push!(SonQ, (X = [X_at(2*d-2,1,d)], Z = [Z_at(2*d-1,2,d)]))
  
  # loop over
  for i in 1:d-2
    y = 2*i
    for x in 2:2:2*d-1
      push!(SonQ, (X = [X_at(x,y-1,d), X_at(x,y+1,d)] , Z = [Z_at(x-1,y,d),Z_at(x+1,y,d)]))
    end
    y = 2*i +1
    push!(SonQ, (X = [X_at(2,y,d)], Z = [Z_at(1,y+1,d),Z_at(1,y-1,d)]))
    for x in 3:2:2*(d-1)-1
      push!(SonQ, (X = [X_at(x-1,y,d),X_at(x+1,y,d)], Z = [Z_at(x,y+1,d),Z_at(x,y-1,d)]))
    end
    push!(SonQ, (X = [X_at(2*d-2,y,d)], Z = [Z_at(2*d-1,y+1,d),Z_at(2*d-1,y-1,d)]))
  end
  y = 2*(d-1)
  for x in 2:2:2*d-1
    push!(SonQ, (X = [X_at(x,y-1,d), X_at(x,y+1,d)] , Z = [Z_at(x-1,y,d),Z_at(x+1,y,d)]))
  end
  
  y = 2*d-1
  push!(SonQ, (X = [X_at(2,y,d)], Z = [Z_at(1,y-1,d)]))
  for x in 3:2:2*(d-1)-1
    push!(SonQ, (X = [X_at(x-1,y,d),X_at(x+1,y,d)], Z = [Z_at(x,y-1,d)]))
  end
  push!(SonQ, (X = [X_at(2*d-2,y,d)], Z = [Z_at(2*d-1,y-1,d)]))
  
  return SurfaceCode(d,Qcoord,Xcoord,Zcoord,QonS,SonQ)
end


distance(code::SurfaceCode) = code.d
nqubits(code::SurfaceCode) = code.d^2+(code.d-1)^2

"""
--------------------------------------------------------------------------------
--------------------------------------------------------------------------------
-                               PRINTING FUNCTIONS                             -
--------------------------------------------------------------------------------
--------------------------------------------------------------------------------
"""

function printlattice!(p::Plots.Plot, code::SurfaceCode, kwargs...)
  markersize = 12/sqrt(code.d)
  p = scatter!(first.(code.Qcoord),last.(code.Qcoord),markersize=markersize,color=:black)
  for y in 1:2:2*code.d-1
    p = plot!([1,2*code.d-1],[y,y],color=:black, kwargs...)
  end
  for x in 1:2:2*code.d-2
    p = plot!([x+1,x+1],[2*code.d-1,1],color=:black, kwargs...)
  end
  return p
end

printlattice(code::SurfaceCode; title::String = "") =
  printlattice!(plot([0],[0],color=:black, legend=false, aspect_ratio=:equal,
    background_color=:transparent, ticks=false, axis=false,foreground_color=:black,title=title), code)


function printTN(code::SurfaceCode; title::String = "") 
  markersize = 12/sqrt(code.d)
  p = scatter(first.(code.Qcoord),last.(code.Qcoord),markersize=markersize,color=:black,
  background_color=:transparent, ticks=false, axis=false,foreground_color=:black, title=title,legend=false, aspect_ratio=:equal)
  for y in 1:2*code.d-1
    p = plot!([1,2*code.d-1],[y,y],color=:black)
  end
  for x in 0:2*code.d-2
    p = plot!([x+1,x+1],[2*code.d-1,1],color=:black)
  end
  p = scatter!(first.(code.Xcoord),last.(code.Xcoord),markersize=5,color=:red,markershape=:square)
  p = scatter!(first.(code.Zcoord),last.(code.Zcoord),markersize=5,color=:blue,markershape=:square)
  return p
end
function printXstabilizers!(p::Plots.Plot, code::SurfaceCode; a::Float64 = 1.0, markersize=4, ϵ::Float64=0.02)
  #p = scatter!(first.(code.Xcoord),last.(code.Xcoord),markersize=10,color=:red,markershape=:star5)
  for (x,y) in code.Xcoord
    if y == 1
      p = plot!(Shape([x-1+ϵ,x,x+1-ϵ],[1,2-ϵ,1]), color=:red, opacity=.5)
    elseif y == 2*code.d-1
      p = plot!(Shape([x-1+ϵ,x,x+1-ϵ],[y,y-1+ϵ,y]), color=:red, opacity=.5)
    else
      p = plot!(Shape([x-1+ϵ,x,(x+1)-ϵ,x],[y,y+1-ϵ,y,y-1+ϵ]), color=:red, opacity=.5)
    end
  end
 return p
end

function printXstabilizers(code::SurfaceCode; kwargs...)
  p = printlattice(code)
  p = printXstabilizers!(p, code; kwargs...)
  return p
end

function printZstabilizers!(p::Plots.Plot, code::SurfaceCode; a::Float64 = 1.0, markersize=4, ϵ::Float64=0.02)
  #p = scatter!(first.(code.Zcoord),last.(code.Zcoord),markersize=10,color=:blue,markershape=:star5)
  for (x,y) in code.Zcoord
    if x == 1
      p = plot!(Shape([x,x+1-ϵ,x],[y-1+ϵ,y,y+1-ϵ]), color=:blue, opacity=.5)
    elseif x == 2*code.d-1
      p = plot!(Shape([x,x-1-ϵ,x],[y-1+ϵ,y,y+1-ϵ]), color=:blue, opacity=.5)
    else
      p = plot!(Shape([x-1+ϵ,x,(x+1)-ϵ,x],[y,y+1-ϵ,y,y-1+ϵ]), color=:blue, opacity=.5)
    end
  end
 return p
end

function printZstabilizers(code::SurfaceCode; kwargs...)
  p = printlattice(code)
  p = printZstabilizers!(p, code; kwargs...)
  return p
end

function printstabilizers(code::SurfaceCode; kwargs...)
  p = printlattice(code)
  p = printXstabilizers!(p, code; kwargs...)
  p = printZstabilizers!(p, code; kwargs...)
  return p
end


function printsyndrome!(p::Plots.Plot, s::NamedTuple, code::SurfaceCode; kwargs...)
  markersize = 8/sqrt(code.d)
  sX = s[:X]
  sZ = s[:Z]
  charges = findall(x -> x == 1, sX)
  fluxes = findall(x -> x == 1, sZ)
  for e in charges
    p = plot!([code.Xcoord[e][1]],[code.Xcoord[e][2]],markersize=3*markersize,color=:red,marker=:star5, opacity=.75, kwargs...)
  end
  for f in fluxes
    p = plot!([code.Zcoord[f][1]],[code.Zcoord[f][2]],markersize=3*markersize,color=:blue,marker=:star5, opacity=.75)
  end
  return p
end

function printsyndrome(s::NamedTuple, code::SurfaceCode; kwargs...)
  p = printlattice(code; kwargs...)
  p = printsyndrome!(p, s, code)
  return p
end

function printpauli!(p::Plots.Plot, e::Vector{<:Array}, code::SurfaceCode; kwargs...)
  markersize = 6/sqrt(code.d)
  Xerrors = findall(x -> x == 1, first.(e))
  Zerrors = findall(x -> x == 1, last.(e))
  for i in 1:nqubits(code)
    if i in Zerrors && !(i in Xerrors)
      if iseven(code.Qcoord[i][1])
        p = plot!([code.Qcoord[i][1],code.Qcoord[i][1]],[code.Qcoord[i][2]-1,code.Qcoord[i][2]+1], color=:blue, linewidth=2, kwargs...)
      else
        if code.Qcoord[i][1] == 1
          p = plot!([code.Qcoord[i][1],code.Qcoord[i][1]+1],[code.Qcoord[i][2],code.Qcoord[i][2]], color=:blue, linewidth=2)
        elseif code.Qcoord[i][1] == 2*code.d-1
          p = plot!([code.Qcoord[i][1]-1,code.Qcoord[i][1]],[code.Qcoord[i][2],code.Qcoord[i][2]], color=:blue, linewidth=2)
        else
          p = plot!([code.Qcoord[i][1]-1,code.Qcoord[i][1]+1],[code.Qcoord[i][2],code.Qcoord[i][2]], color=:blue, linewidth=2)
        end
      end
      p = plot!([code.Qcoord[i][1]],[code.Qcoord[i][2]], color=:blue, marker=:circle,markersize=2*markersize)
    end
    if i in Xerrors && !(i in Zerrors)
      if iseven(code.Qcoord[i][1])
        p = plot!([code.Qcoord[i][1]-1,code.Qcoord[i][1]+1],[code.Qcoord[i][2],code.Qcoord[i][2]], color=:red, linewidth=2)
      else
        if code.Qcoord[i][2] == 1
          p = plot!([code.Qcoord[i][1],code.Qcoord[i][1]],[code.Qcoord[i][2],code.Qcoord[i][2]+1], color=:red, linewidth=2)
        elseif code.Qcoord[i][2] == 2*code.d-1
          p = plot!([code.Qcoord[i][1],code.Qcoord[i][1]],[code.Qcoord[i][2],code.Qcoord[i][2]-1], color=:red, linewidth=2)
        else
          p = plot!([code.Qcoord[i][1],code.Qcoord[i][1]],[code.Qcoord[i][2]-1,code.Qcoord[i][2]+1], color=:red, linewidth=2)
        end
      end
      p = plot!([code.Qcoord[i][1]],[code.Qcoord[i][2]], color=:red, marker=:circle,markersize=2*markersize)
    end
    if (i in Xerrors) && (i in Zerrors)
      if iseven(code.Qcoord[i][1])
        p = plot!([code.Qcoord[i][1],code.Qcoord[i][1]],[code.Qcoord[i][2]-1,code.Qcoord[i][2]+1], color=:blue, linewidth=2)
      else
        if code.Qcoord[i][1] == 1
          p = plot!([code.Qcoord[i][1],code.Qcoord[i][1]+1],[code.Qcoord[i][2],code.Qcoord[i][2]], color=:blue, linewidth=2)
        elseif code.Qcoord[i][1] == 2*code.d-1
          p = plot!([code.Qcoord[i][1]-1,code.Qcoord[i][1]],[code.Qcoord[i][2],code.Qcoord[i][2]], color=:blue, linewidth=2)
        else
          p = plot!([code.Qcoord[i][1]-1,code.Qcoord[i][1]+1],[code.Qcoord[i][2],code.Qcoord[i][2]], color=:blue, linewidth=2)
        end
      end
      if iseven(code.Qcoord[i][1])
        p = plot!([code.Qcoord[i][1]-1,code.Qcoord[i][1]+1],[code.Qcoord[i][2],code.Qcoord[i][2]], color=:red, linewidth=2)
      else
        if code.Qcoord[i][2] == 1
          p = plot!([code.Qcoord[i][1],code.Qcoord[i][1]],[code.Qcoord[i][2],code.Qcoord[i][2]+1], color=:red, linewidth=2)
        elseif code.Qcoord[i][2] == 2*code.d-1
          p = plot!([code.Qcoord[i][1],code.Qcoord[i][1]],[code.Qcoord[i][2],code.Qcoord[i][2]-1], color=:red, linewidth=2)
        else
          p = plot!([code.Qcoord[i][1],code.Qcoord[i][1]],[code.Qcoord[i][2]-1,code.Qcoord[i][2]+1], color=:red, linewidth=2)
        end
      end
      p = plot!([code.Qcoord[i][1]],[code.Qcoord[i][2]], color=:red, marker=:circle,markersize=2*markersize)
      p = plot!([code.Qcoord[i][1]+0.05],[code.Qcoord[i][2]+0.05], color=:red, marker=:circle,markersize=2*markersize)
      p = plot!([code.Qcoord[i][1]-0.05],[code.Qcoord[i][2]-0.05], color=:blue, marker=:circle,markersize=2*markersize)
    end
  end
  return p
end
function printpauli(e::Vector{<:Array}, code::SurfaceCode; kwargs...)
  p = printlattice(code; kwargs...)
  p = printpauli!(p, e, code)
  return p
end

function printcode(e::Vector{<:Array}, s::NamedTuple, code::SurfaceCode; kwargs...)
  p = printlattice(code; kwargs...)
  p = printpauli!(p, e, code)
  p = printsyndrome!(p, s, code)
  return p
end
