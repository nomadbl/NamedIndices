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
        @testset "$nit" for nit in (Pair,)
            ni = NamedIndex(:a, nit(:b, _ni))
            x = ni([1,2,3,4])
            @test x.a == 1
            @test x.b.a == 2
            @test x.b.b == 3
            @test x.b.c == 4
        end
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
    @testset "size checking" begin
        ni = NamedIndex(:a, :b)
        x = rand(2)
        @test_nowarn ni(x)
        ni = NamedIndex(:ni=> ni, :ni2=>ni)
        @test length(ni) == 4
        ni = NamedIndex(:ni=> ni, :a)
        @test length(ni) == 5
    end
    @testset "setproperty" begin
        @testset "flat" begin
            ni = NamedIndex(:a, :b)
            x = ni(rand(Float32, 2))
            x.a = 1
            x.b = 2
            @test size(parent(x)) == (2,)
            @test parent(x) == Float32[1,2]
            @test x.a[] == 1
            @test x.b[] == 2

            ni = NamedIndex(:a, :b)
            x = ni(rand(Float32, 2, 3))
            x.a = 1
            x.b = 2
            @test size(parent(x)) == (2,3)
            @test parent(x) == Float32[1 1 1; 2 2 2]
            @test all(map(y->y==1, x.a))
            @test all(map(y->y==2, x.b))

            ni = NamedIndex(:a=>(2,), :b)
            x = ni(undef, Float32, 2)
            x.a = 1
            x.b = 2
            @test size(parent(x)) == (3,2)
            @test parent(x) == Float32[1 1; 1 1; 2 2]
            @test all(map(y->y==1, x.a))
            @test all(map(y->y==2, x.b))
        end
        @testset "composed" begin
            ni = NamedIndex(:a, :b)
            ni = NamedIndex(:ni=> ni, :a)
            x = ni(zeros(Float32, 3))
            x.ni.a = 1
            x.ni.b = 2
            x.a = 3
            @test size(parent(x)) == (3,)
            @test parent(x) == Float32[1,2,3]
            @test parent(x) == Float32[1,2,3]
            @test x.ni.a == 1
            @test x.ni.b == 2
            @test x.a == 3

            x = ni(zeros(Float32, 3,2))
            x.ni.a = 1
            x.ni.b = 2
            x.a = 3
            @test size(parent(x)) == (3,2)
            @test parent(x) == Float32[1 1; 2 2; 3 3]
            @test all(map(y->y==1, x.ni.a))
            @test all(map(y->y==2, x.ni.b))
            @test x.a == 3
        end
    end
    @testset "convenience constructor" begin
        ni = NamedIndex(:a, :b)
        x = ni(undef, Int32)
        @test size(parent(x)) == (2,)
        @test eltype(parent(x)) == Int32
        x = ni(undef, Int32, 1,2,3)
        @test size(parent(x)) == (2,1,2,3)
        @test eltype(parent(x)) == Int32
        ni = NamedIndex(:ni=> ni, :a)
        x = ni(undef, Int32, 1,2,3)
        @test size(parent(x)) == (3,1,2,3)
        @test eltype(parent(x)) == Int32
    end
end
