# NamedIndices.jl
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

The original array can be retrieved by using `Base.parent`
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
