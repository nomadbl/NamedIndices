module NamedIndices

import Base: parent, getproperty, propertynames, show, length, setproperty!, keys

"""
    `NamedIndex(names...; axis=1)`
Name numerical indices.
`names` can be either `Symbol` or `Pair{Symbol,NamedIndex}` or `Tuple{Symbol,NamedIndex}`.
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
julia> ni = NamedIndex(:a, :b => _ni)
julia> x = ni([1,2,3,4])
julia> x.a == 1
true
julia> x.b.a == 2
true
julia> x.b.b == 3
true
julia> x.b.c == 4
true
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
struct NamedIndex{N,A,I} # N - names, A- axis
    indices::I
    intercept::Int
    len::Int
    function NamedIndex(ax::Int, names::NTuple{N,Symbol}, indices::I, intercept::Int, len::Int) where {N,I}
        new{names,ax,I}(indices,intercept,len)
    end
end
function NamedIndex(intercept::Int, x::NamedIndex{N,A,I}) where {N,I,A}
    NamedIndex(A, N, x.indices, intercept, x.len)
end
NamedNamedIndex{N,A,I} = Union{Tuple{Symbol,NamedIndex{N,A,I}},
                               Pair{Symbol, NamedIndex{N,A,I}}}

@inline valval(::Val{N}) where N = N

keys(::NamedIndex{N}) where N = N
function _buildindices!(lastindex::Ref{Int}, x::Symbol)
    lastindex[] += one(Int)
    return lastindex[]
end
function _buildindices!(lastindex::Ref{Int}, x::NamedNamedIndex{N,A,I}) where {N,I,A}
    ni = x[2]
    index = NamedIndex(lastindex[], ni)
    lastindex[] = lastindex[] + ni.len
    return index
end

function NamedIndex(ax::Int, names::Vararg{Union{Symbol,NamedNamedIndex}})
    lastindex = Ref(zero(ax))
    f! = Base.Fix1(_buildindices!, lastindex)
    indices = map(f!, names)
    _names = (x->x isa Union{Tuple, Pair} ? x[1] : x).(names)
    len = lastindex[]
    NamedIndex(ax, _names, indices, 0, len)
end
function NamedIndex(name::Union{Symbol,NamedNamedIndex}, names::Vararg{Union{Symbol,NamedNamedIndex}}; ax::Int=1)
    NamedIndex(ax, name, names...)
end
function getproperty(index::NamedIndex, name::Symbol)
    if name == :indices
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
    hasproperty(index, N) && in(N, keys(index)) || throw(DomainError(name, "$N is a property of NamedIndex, rename it"))
    ind = findfirst(x-> x==valval(name), keys(index))
    ind === nothing && throw(DomainError(N, "NamedIndex has no property $N"))
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
function propertynames(index::NamedIndex)
    (:indices, :intercept, :len, keys(index)...)
end
function length(index::NamedIndex)
    getfield(index, :len)
end
function show(io::IO, ::MIME"text/plain", ni::NamedIndex{N,I,A}) where {N,I,A}
    print(io, "NamedIndex(axis=",A,", length=",ni.len,")")
end

struct NamedIndexedArray{AX,N,T,NI,I,A<:AbstractArray{T,N}} <: AbstractArray{T,N}
    arr::A
    index::NamedIndex{NI,AX,I}
    function NamedIndexedArray(x::A, i::NamedIndex{NI,AX,I}) where {AX,N,T,NI,I,A<:AbstractArray{T,N}}
        @assert i.len == size(x, AX) "array size $(size(x)) does not match index length $(i.len) on indexing axis $A"
        new{AX,N,T,NI,I,A}(x, i)
    end
end
Base.size(x::NamedIndexedArray) = size(parent(x))
Base.getindex(x::NamedIndexedArray, i...) = getindex(parent(x), i...)
Base.setindex!(x::NamedIndexedArray, v, i...) = setindex!(parent(x), v, i...)
Base.parent(x::NamedIndexedArray) = x.arr
(ni::NamedIndex)(x::A) where {A<:AbstractArray} = NamedIndexedArray(x, ni)
function (ni::NamedIndex{NI,AX})(::UndefInitializer, elt::Type, size::Vararg{T,N}) where {T,N,NI,AX}
    _size = ntuple(i->(i < AX ? size[i] : (i == AX ? length(ni) : size[i-1])), Val(N+1))
    x = Array{elt, N+1}(undef, _size...)
    ni(x)
end
function Base.similar(x::NamedIndexedArray{AX}, ::Type{S}, dims::Dims{N}) where {S,N,AX}
    ni = getfield(x, :index)
    _dims = ntuple(i->(i < AX ? dims[i] : dims[i+1]), Val(N-1))
    ni(undef, S, _dims...)
end

function getproperty(x::NamedIndexedArray{AX,N,T,NI,I}, name::Symbol) where {AX,N,T,NI,I}
    if name == :index
        return getfield(x, name)
    elseif name == :arr
        return getfield(x, name)
    elseif name in NI
        return _getproperty(x, Val(name))
    end
    throw(DomainError(name, "NamedIndexedArray has no property $name"))
end
function _getproperty(x::NamedIndexedArray{AX,N}, name::Val{M}) where {AX,N,M}
    index = _getproperty(getfield(x, :index), name)
    indices = ntuple(i->i == AX ? toindex(index) : Colon(), Val(N))
    @views res = x.arr[indices...]
    # if result is not a single item - i.e. it can be further indexed - wrap with the indexing inner NamedIndex
    __getproperty(res, index)
end
__getproperty(res, index::Int) = res
__getproperty(res, index::NamedIndex) = NamedIndexedArray(res, NamedIndex(0, index))

propertynames(x::NamedIndexedArray) = keys(x.index)

function setproperty!(x::NamedIndexedArray, name::Symbol, v)
    if name == :index
        return setfield!(x, name)
    elseif name == :arr
        return setfield!(x, name)
    elseif !(name in keys(getfield(x, :index)))
        throw(DomainError(name, "NamedIndexedArray has no property $name"))
    else
        _setproperty!(x, Val(name), v)
    end
end
function _setproperty!(x::NamedIndexedArray{AX,N}, name::Val{M}, v) where {AX,N,M}
    index = _getproperty(getfield(x, :index), name)
    indices = ntuple(i->i === AX ? toindex(index) : Colon(), Val(N))
    # if result is not a single item - i.e. it can be further indexed - wrap with the indexing inner NamedIndex
    __setproperty!(x, indices, index, v)
end
function __setproperty!(x, indices, index, v)
    if x.arr[indices...] isa AbstractArray
        x.arr[indices...] .= v
    else
        x.arr[indices...] = v
    end
end

function show(io::IO, ::MIME"text/plain", x::NamedIndexedArray)
    print(io, size(parent(x)), " NamedIndexedArray{", eltype(parent(x)),"}\n")
    show(io, parent(x))
end

export NamedIndex, NamedIndexedArray

end # module NamedIndices
