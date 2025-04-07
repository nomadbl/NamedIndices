module NamedIndices

using StaticArrays
import Base: parent, getproperty, propertynames, show, length, setproperty!, keys

"""
    `NamedIndex(names...; axis=1)`
Name numerical indices.
`names` can be either `Symbol`, `Pair{Symbol,NTuple{N,Int}}` ,`Pair{Symbol,NamedIndex}`,
`Pair{Symbol,Tuple{NamedIndex,NTuple{N,Int}}}.
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

Array properties can have non-unit sizes
```
julia> ni = NamedIndex(:a=>(2,2), :b)
julia> x = collect(1:5)
2×5 Matrix{Int64}:
 1  3  5  7   9
 2  4  6  8  10

julia> ni(x).b
5-element view(::Matrix{Int64}, 1, :) with eltype Int64:
 1
 3
 5
 7
 9
```
"""
struct NamedIndex{N,A,IND,INT,S,LEN,IA} # N - names, A- axis
    indices_arrays::IA
    function NamedIndex{N,A,IND,INT,S,LEN}(indices::IA) where {N,A,IND,INT,S,LEN,IA}
        new{N,A,IND,INT,S,LEN,IA}(indices)
    end
end
function _index_arrays(indices, sizes, intercept)
    indices_arrays = []
    for i in 1:length(sizes)
        push!(indices_arrays, _index_array(indices[i], sizes[i], intercept))
    end
    out = tuple(indices_arrays...)
    return out
end

function NamedIndex(ax::Int, names::NTuple{N,Symbol}, indices::I, intercept::Int, sizes::Tuple, len::Int) where {N,I}
    @assert length(sizes) == N "NamedIndex: number of symbols $N does not match length of shapes list $sizes"
    NamedIndex{names,ax,I,intercept,sizes,len}(_index_arrays(indices, sizes, intercept))
end
function NamedIndex(intercept::Int, x::NamedIndex{N,A,IND,INT,S,LEN}) where {N,A,IND,INT,S,LEN}
    NamedIndex{N,A,IND,intercept,S,LEN}(getfield(x,:indices_arrays))
end
NamedNamedIndex = Pair{Symbol, NamedIndex{N,A,IND,INT,S,LEN,IA}} where {N,A,IND,INT,S,LEN,IA}
NamedNamedIndexWithSize = Pair{Symbol, Tuple{NamedIndex{N,A,IND,INT,SZ,LEN,IA}, NTuple{S,Int}}} where {N,A,IND,INT,SZ,LEN,IA,S}
SymbolWithSize = Pair{Symbol, NTuple{S,Int}} where S

Base.length(::NamedIndex{N,A,IND,INT,S,LEN}) where {A,N,IND,INT,S,LEN} = LEN
@inline valval(::Val{N}) where N = N

keys(::NamedIndex{N}) where N = N
sizes(::NamedIndex{N,A,IND,INT,S,LEN}) where {A,N,IND,INT,S,LEN} = S
indices(::NamedIndex{N,A,IND,INT,S,LEN}) where {A,N,IND,INT,S,LEN} = IND
intercept(::NamedIndex{N,A,IND,INT,S,LEN}) where {A,N,IND,INT,S,LEN} = INT
intercept(ni::AbstractArray{NamedIndex}) = intercept(first(ni))
function sizes(::NamedIndex{N,A,IND,INT,S,LEN}, name::Symbol) where {A,N,IND,INT,S,LEN}
    ind = findfirst(x-> x==name, N)
    if ind !== nothing
        return S[ind]
    end
    throw(DomainError(name, "NamedIndex has no property $name"))
end
axis(::NamedIndex{N,A,IND,INT,S,LEN}) where {A,N,IND,INT,S,LEN} = A

_size(name::Symbol) = (1,)
_size(name::SymbolWithSize) = name.second
_size(name::NamedNamedIndex) = (1,)
_size(name::NamedNamedIndexWithSize) = name.second[2]
_name(name::Symbol) = name
_name(name::SymbolWithSize) = name.first
_name(name::NamedNamedIndex) = name.first
_name(name::NamedNamedIndexWithSize) = name.first

function _buildindices!(lastindex::Ref{Int}, x::Union{Symbol,SymbolWithSize})
    _buildindices!(lastindex, _name(x), _size(x))
end
function _buildindices!(lastindex::Ref{Int}, x::Symbol, s::NTuple{N,Int}) where N
    ind = lastindex[]
    lastindex[] += prod(s)
    return ind
end
function _buildindices!(lastindex::Ref{Int}, x::NamedNamedIndex)
    _buildindices!(lastindex, x, _size(x))
end
function _buildindices!(lastindex::Ref{Int}, x::NamedNamedIndexWithSize)
    _buildindices!(lastindex, _name(x)=>x.second[1], _size(x))
end
function _buildindices!(lastindex::Ref{Int}, x::NamedNamedIndex, s::NTuple{N,Int}) where N
    ni = x[2]
    index = NamedIndex(lastindex[]-1, ni)
    lastindex[] = lastindex[] + length(ni) * prod(s)
    return index
end

function _index_array(i::Int, shape, intercept)
    if prod(shape) == 1
        return i + intercept
    else
        res = Array{Int}(undef, shape...)
        for I in LinearIndices(shape)
            res[I] = i + intercept + I - 1
        end
        SArray{Tuple{shape...}}(res)
    end
end
function _index_array(i::NamedIndex{N,A,IND,INT}, shape, intercept) where {N,A,IND,INT}
    if prod(shape) == 1
        return i
    else
        res = Array{NamedIndex}(undef, shape...)
        for I in LinearIndices(shape)
            res[I] = NamedIndex(INT+(I-1)*length(i), i)
        end
        return SArray{Tuple{shape...}}(res)
    end
end

function NamedIndex(name::Union{Symbol,NamedNamedIndex,SymbolWithSize,NamedNamedIndexWithSize},
                    names::Vararg{Union{Symbol,NamedNamedIndex,SymbolWithSize,NamedNamedIndexWithSize}};
                    ax::Int=1)
    lastindex = Ref(one(ax))
    f! = Base.Fix1(_buildindices!, lastindex)
    indices = map(f!, (name, names...))
    len = lastindex[] - 1
    _names = map(_name, (name, names...))
    _sizes = map(_size, (name, names...))
    NamedIndex(ax, _names, indices, 0, _sizes, len)
end

getproperty(index::NamedIndex, name::Symbol) = _getproperty(index, Val(name))
function _getproperty(index::NamedIndex, name::Val{N}) where N
    hasproperty(index, N) && in(N, keys(index)) || throw(DomainError(name, "$N is a property of NamedIndex, rename it"))
    ind = findfirst(x-> x==valval(name), keys(index))
    ind === nothing && throw(DomainError(N, "NamedIndex has no property $N"))
    __getproperty(index, ind)
end
function __getproperty(index::NamedIndex{N,A,IND,INT,S,LEN}, NM) where {N,A,IND,INT,S,LEN}
    getfield(index, :indices_arrays)[NM]
end

toindex(index::Int) = index
toindex(index::AbstractArray{Int,N}) where N = index
toindex(index::UnitRange) = index
toindex(index::AbstractArray{UnitRange,N}) where N = vcat(reshape(index,:)...)
toindex(index::NamedIndex{N,A,IND,INT,S,LEN}) where {N,A,IND,INT,S,LEN} = (INT+1):(INT+LEN)
propertynames(index::NamedIndex) = keys(index)
function show(io::IO, ::MIME"text/plain", ni::NamedIndex{N,A,I}) where {N,I,A}
    print(io, "NamedIndex(axis=",A,", length=",length(ni),")")
end

struct NamedIndexedArray{AX,N,T,NI,IND,INT,S,LEN,A<:AbstractArray{T,N},IA,VS} <: AbstractArray{T,N}
    arr::A
    index::NamedIndex{NI,AX,IND,INT,S,LEN,IA}
    subarrays_tuple::VS
    function NamedIndexedArray(x::A, i::NamedIndex{NI,AX,IND,INT,S,LEN,IA}) where {AX,N,T,NI,IND,INT,S,LEN,IA,A<:AbstractArray{T,N}}
        @assert LEN == size(x, AX) "array size $(size(x)) does not match index length $(length(i)) on indexing axis $A"
        vs = _subarrays_tuple(x,i)
        new{AX,N,T,NI,IND,INT,S,LEN,A,IA,typeof(vs)}(x,i,vs)
    end
end

function _subarrays_tuple(x::AbstractArray{T,N}, namedindex::NamedIndex{NI}) where {N,T,NI}
    subarrays_array = []
    for name in NI
        push!(subarrays_array, _subarray(x, namedindex, Val(name)))
    end
    return tuple(subarrays_array...)
end
function _subarray(x::AbstractArray{T,N}, namedindex::NamedIndex, name::Val{M}) where {N,T,M}
    AX = axis(namedindex)
    ind = _getproperty(namedindex, name)
    if ind isa AbstractArray
        # return an array of arrays, so that x.name returns an array which can be further indexed without falling back to getindex above
        if eltype(ind) <: NamedIndex
            res = similar(x, NamedIndexedArray, size(ind)...)
            for I in CartesianIndices(size(ind))
                indices = ntuple(i->i == AX ? toindex(ind[I]) : Colon(), Val(N))
                @views res[I] = NamedIndexedArray(x[indices...], NamedIndex(0, ind[I]))
            end
            return SArray{Tuple{size(ind)...}}(res)
        else
            x_size = filter(a->a!=nothing, ntuple(i->i == AX ? nothing : size(x,i), Val(N)))
            new_size = (size(ind)..., x_size...)
            indices = ntuple(i->i == AX ? toindex(ind) : Colon(), Val(N))
            @views res = reshape(x[indices...], new_size)
            return res
        end
    else
        # return a single item (can be an array)
        indices = ntuple(i->i == AX ? toindex(ind) : Colon(), Val(N))
        res = view(x, indices...)
        return _subarray_rewrap(res, ind)
    end
end
# if result is not a single item - i.e. it can be further indexed - wrap with the indexing inner NamedIndex
_subarray_rewrap(res, ind::Int) = res
_subarray_rewrap(res, ind::NamedIndex) = NamedIndexedArray(res, NamedIndex(0, ind))

function index(x::NamedIndexedArray{AX,N,T,NI,IND,INT,S,LEN,A}) where {AX,N,T,NI,IND,INT,S,LEN,A<:AbstractArray{T,N}}
    getfield(x, :index)
end
Base.size(x::NamedIndexedArray) = size(parent(x))
Base.getindex(x::NamedIndexedArray, i...) = getindex(parent(x), i...)
Base.setindex!(x::NamedIndexedArray, v, i...) = setindex!(parent(x), v, i...)
Base.parent(x::NamedIndexedArray) = getfield(x, :arr)
(ni::NamedIndex)(x::A) where {A<:AbstractArray} = NamedIndexedArray(x, ni)
function (ni::NamedIndex{NI,AX})(::UndefInitializer, elt::Type, size::Vararg{T,N}) where {T,N,NI,AX}
    _size = ntuple(i->(i < AX ? size[i] : (i == AX ? length(ni) : size[i-1])), Val(N+1))
    x = Array{elt, N+1}(undef, _size...)
    ni(x)
end
function Base.similar(x::NamedIndexedArray{AX}, ::Type{S}, dims::Dims{N}) where {S,N,AX}
    ni = index(x)
    _dims = ntuple(i->(i < AX ? dims[i] : dims[i+1]), Val(N-1))
    ni(undef, S, _dims...)
end

function getproperty(x::NamedIndexedArray{AX,N,T,NI}, name::Symbol) where {AX,N,T,NI}
    if name in NI
        ind_name = findfirst(n-> n==name, NI)
        return _getproperty_rewrap(getfield(x, :subarrays_tuple)[ind_name])
    end
    throw(DomainError(name, "NamedIndexedArray has no property $name"))
end
_getproperty_rewrap(res) = res
_getproperty_rewrap(res::SubArray{T,0}) where T = res[]

propertynames(x::NamedIndexedArray) = keys(index(x))

function setproperty!(x::NamedIndexedArray, name::Symbol, v)
    if name in keys(index(x))
        _setproperty!(x, Val(name), v)
        return v
    end
    throw(DomainError(name, "NamedIndexedArray has no property $name"))
end
function _setproperty!(x::NamedIndexedArray{AX,N}, name::Val{M}, v) where {AX,N,M}
    ind = _getproperty(index(x), name)
    indices = ntuple(i->i === AX ? toindex(ind) : Colon(), Val(N))
    # if result is not a single item - i.e. it can be further indexed - wrap with the indexing inner NamedIndex
    __setproperty!(x, indices, v)
end
function __setproperty!(x, indices, v)
    if parent(x)[indices...] isa AbstractArray
        parent(x)[indices...] .= v
    else
        parent(x)[indices...] = v
    end
end

function show(io::IO, ::MIME"text/plain", x::NamedIndexedArray)
    print(io, size(parent(x)), " NamedIndexedArray{", eltype(parent(x)),"}\n")
    show(io, parent(x))
end

export NamedIndex, NamedIndexedArray, index, intercept, indices, sizes

end # module NamedIndices
