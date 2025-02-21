module NamedIndices

import Base: parent, getproperty

"""
    `NamedIndex`
Name numerical indices.
NamedIndex(names...; axis=1)
`names` can be either `Symbol` or `Tuple{Symbol,NamedIndex}`.
`axis` is the axis along which the indices will slice a wrapped array.

An array is then wrapped to refer to its contents using the new names.

```julia
julia> ni = NamedIndex(:a, :b)
julia> x = ni([1,2])
julia> x.a == 1
true
julia> x.b == 2
true
```

The original array can be retrieved by using Base.parent
```
julia> ni = NamedIndex(:a, :b)
julia> x = rand(2)
julia> w_x = ni(x)
julia> parent(w_x) === x
true
```

Named Indices may be composed together to construct "namespaces".
```julia
julia> _ni = NamedIndex(:a, :b, :c)
julia> ni = NamedIndex(:a, (:b, _ni))
julia> x = ni([1,2,3,4])
julia> x.a == 1
true
julia> x.b.a == 2
true
julia> x.b.b == 3
true
julia> x.b.c == 4
```

Matrices or higher dimensional arrays will be sliced along the `axis` dimension
```
julia> ni = NamedIndex(:a, :b)
julia> x = reshape(collect(1:10), 2, 5)
2×5 Matrix{Int64}:
 1  3  5  7   9
 2  4  6  8  10

julia> ni(x).a
5-element view(::Matrix{Int64}, 1, :) with eltype Int64:
 1
 3
 5
 7
 9
```
Notice this will always return a view.
"""
struct NamedIndex{N,I,A}
    ax::Val{A}
    names::N
    indices::I
    intercept::Int
    len::Int
end
NamedNamedIndex{N,I,A} = Tuple{Symbol,NamedIndex{N,I,A}}
function _buildindices!(lastindex::Ref{Int}, x::Symbol)
    lastindex[] += one(Int)
    return lastindex[]
end
function _buildindices!(lastindex::Ref{Int}, x::NamedNamedIndex{N,I,A}) where {N,I,A}
    index = NamedIndex(lastindex[], x[2])
    lastindex[] = lastindex[] + x[2].len
    return index
end
function NamedIndex(intercept::Int, x::NamedIndex{N,I,A}) where {N,I,A}
    NamedIndex{N,I,A}(x.ax, x.names, x.indices, intercept, x.len)
end
function NamedIndex(ax::Int, names::Vararg{Union{Symbol,NamedNamedIndex}})
    lastindex = Ref(zero(ax))
    f! = Base.Fix1(_buildindices!, lastindex)
    indices = map(f!, names)
    _names = (x->x isa Tuple ? Val(x[1]) : Val(x)).(names)
    len = lastindex[]
    NamedIndex(Val(ax), _names, indices, 0, len)
end
function NamedIndex(name::Union{Symbol,NamedNamedIndex}, names::Vararg{Union{Symbol,NamedNamedIndex}}; ax::Int=1)
    NamedIndex(ax, name, names...)
end
function getproperty(index::NamedIndex, name::Symbol)
    if name == :ax
        return getfield(index, name)
    elseif name == :names
        return getfield(index, name)
    elseif name == :indices
        return getfield(index, name)
    elseif name == :intercept
        return getfield(index, name)
    elseif name == :len
        return getfield(index, name)
    else
        _getproperty(index, Val(name))
    end
end
function _getproperty(index::NamedIndex, name::Val{N}) where N
    N isa Symbol
    hasproperty(index, N) && throw(DomainError(name, "$name is a property of NamedIndex, rename it"))
    ind = findfirst(x-> x==name, index.names)
    ind === nothing && throw(DomainError(N, "NamedIndex has no property $name"))
    __getproperty(index, Val(convert(Int, ind)))
end
function __getproperty(index::NamedIndex, ::Val{N}) where N
    i = getfield(index, :indices)[N]
    intercept = getfield(index, :intercept)
    ___getproperty(i, intercept)
end
___getproperty(i::Int, intercept) = i + intercept
___getproperty(i::NamedIndex, intercept) = i

toindex(index::Int) = index
toindex(index::NamedIndex) = (index.intercept+1):(index.intercept+index.len)
struct NamedIndexedArray{AX,N,T,NI,I,A<:AbstractArray{T,N}}
    arr::A
    index::NamedIndex{NI,I,AX}
end
Base.parent(x::NamedIndexedArray) = x.arr
(i::NamedIndex)(x::A) where {A<:AbstractArray} = NamedIndexedArray(x, i)
function getproperty(x::NamedIndexedArray, name::Symbol)
    if name == :index
        return getfield(x, name)
    elseif name == :arr
        return getfield(x, name)
    elseif !(Val(name) in getfield(getfield(x, :index), :names))
        throw(DomainError(name, "NamedIndexedArray has no property $name"))
    else
        return _getproperty(x, Val(name))
    end
end
function _getproperty(x::NamedIndexedArray{AX,N}, name::Val{M}) where {AX,N,M}
    index = _getproperty(getfield(x, :index), name)
    indices = ntuple(i->Val(i) === x.index.ax ? toindex(index) : Colon(), Val(N))
    @views res = x.arr[indices...]
    # if result is not a single item - i.e. it can be further indexed - wrap with the indexing inner NamedIndex
    __getproperty(res, index)
end
__getproperty(res, index::Int)  = res
__getproperty(res, index::NamedIndex) = NamedIndexedArray(res, NamedIndex(0, index))

export NamedIndex, NamedIndexedArray

end # module NamedIndices
