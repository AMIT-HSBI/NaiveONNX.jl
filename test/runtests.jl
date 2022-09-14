using SafeTestsets

#@safetestset "Data Visualization" begin include("visualizationTests.jl") end
@safetestset "Data Visualization" begin include("onnxGenTests.jl") end
