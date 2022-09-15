using SafeTestsets
using Logging
Logging.disable_logging(Logging.Info)

@safetestset "Data Visualization" begin include("visualizationTests.jl") end
@safetestset "Data Visualization" begin include("onnxGenTests.jl") end
