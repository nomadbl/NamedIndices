using Test, NamedIndices

@testset "NamedIndex" begin
    @testset "flat index" begin
        ni = NamedIndex(:a, :b)
        x = rand(2)
        named = ni(x)
        @test named.a == x[1]
        @test named.b == x[2]
    end
    @testset "composed index" begin
        _ni = NamedIndex(:a, :b, :c)
        ni = NamedIndex(:a, (:b, _ni))
        x = ni([1,2,3,4])
        @test x.a == 1
        @test x.b.a == 2
        @test x.b.b == 3
        @test x.b.c == 4
    end
    @testset "slicing" begin
        ni = NamedIndex(:a, :b)
        x = reshape(collect(1:10), 2, 5)
        @test ni(x).a == [1,3,5,7,9]
        @test ni(x).b == [2,4,6,8,10]
    end
    @testset "type inference" begin
        ni = NamedIndex(:a, :b)
        x = rand(2)
        x = ni(x)
        f(x) = x.a
        @inferred f(x)
        @inferred f(ni)
    end
end
