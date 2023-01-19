using Test
using NaiveONNX

@testset "eq_928" begin
  csvFile = abspath(joinpath(@__DIR__, "csv", "eq_928.csv"))
  onnxModel = abspath(joinpath(@__DIR__, "onnx", "eq_928.onnx"))
  rm(onnxModel, force=true)
  inputNames = ["scalableModelicaModel.simpleSpring[7].s_rel"]
  outputNames = ["scalableModelicaModel.springChain[6].spring[1].s_rel", "scalableModelicaModel.springChain[6].spring[4].s_rel", "scalableModelicaModel.springChain[6].spring[3].s_rel", "scalableModelicaModel.springChain[6].spring[2].s_rel", "scalableModelicaModel.springChain[6].spring[4].f"]


  trainONNX(csvFile, onnxModel, inputNames, outputNames; nepochs=2)
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
  s = data_in[1,:]
  r = data_in[2,:]
  y = data_out[1,:]

  keep = findall(i->(i==true), isRight.(s, r, y))

  filtered_data_in = data_in[:, keep]
  filtered_data_out = data_out[:, keep]
  return (filtered_data_in, filtered_data_out)
end

@testset "simpleLoop_eq14" begin
  csvFile = abspath(joinpath(@__DIR__, "csv", "simpleLoop_eq14.csv"))
  onnxModel = abspath(joinpath(@__DIR__, "onnx", "simpleLoop_eq14.onnx"))
  inputNames = ["s", "r"]
  outputNames = ["y"]

  trainONNX(csvFile, onnxModel, inputNames, outputNames; filterFunc=filterData, nepochs=10)
  @test isfile(onnxModel)
  rm(onnxModel, force=true)
end