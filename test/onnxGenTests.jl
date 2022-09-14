using Test
using NaiveONNX

@testset "eq_928" begin
  csvFile = abspath(joinpath(@__DIR__, "csv", "eq_928.csv"))
  onnxModel = abspath(joinpath(@__DIR__, "onnx", "eq_928.onnx"))
  rm(onnxModel, force=true)
  nInputs = 1

  trainONNX(csvFile, onnxModel, nInputs; nepochs=2)
  @test isfile(onnxModel)
  rm(onnxModel, force=true)
end

"""
Filter data for `x = r*s-y <= y` to get uniqe data points.
"""
function isRight(s,r,y)
  x = r*s-y
  return x > y
end

"""
Filter training data to only contain unambiguous data points
by using only the bottom right intersection points.
"""
function filterData(data_in, data_out)
  s = [x[2] for x in data_in];
  r = [x[2] for x in data_in];
  y = [x[1] for x in data_out];

  keep = findall(i->(i==true), isRight.(s, r, y))

  filtered_data_in = data_in[keep]
  filtered_data_out = data_out[keep]
  return (filtered_data_in, filtered_data_out)
end

@testset "simpleLoop_eq14" begin
  csvFile = abspath(joinpath(@__DIR__, "csv", "simpleLoop_eq14.csv"))
  onnxModel = abspath(joinpath(@__DIR__, "onnx", "simpleLoop_eq14.onnx"))
  nInputs = 2

  trainONNX(csvFile, onnxModel, nInputs; filterFunc=filterData, nepochs=10)
  @test isfile(onnxModel)
  rm(onnxModel, force=true)
end