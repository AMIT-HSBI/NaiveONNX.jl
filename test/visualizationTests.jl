using Test
using NaiveONNX

@testset "eq_928" begin
  csvFile = abspath(joinpath(@__DIR__, "csv", "eq_928.csv"))
  inputNames = ["scalableModelicaModel.simpleSpring[7].s_rel"]
  outputNames = ["scalableModelicaModel.springChain[6].spring[1].s_rel", "scalableModelicaModel.springChain[6].spring[4].s_rel", "scalableModelicaModel.springChain[6].spring[3].s_rel", "scalableModelicaModel.springChain[6].spring[2].s_rel", "scalableModelicaModel.springChain[6].spring[4].f"]

  data = NaiveONNX.readData(csvFile, inputNames, outputNames)
  visualizeData(data.train, data.inputNames, data.outputNames, outidx=[1,5])
end

@testset "simpleLoop_eq14" begin
  csvFile = abspath(joinpath(@__DIR__, "csv", "simpleLoop_eq14.csv"))
  inputNames = ["s", "r"]
  outputNames = ["y"]

  data = NaiveONNX.readData(csvFile, inputNames, outputNames)
  visualizeData(data.train, data.inputNames, data.outputNames)
end

@testset "simpleLoop_eq14 3D" begin
  csvFile = abspath(joinpath(@__DIR__, "csv", "simpleLoop_eq14.csv"))
  inputNames = ["s", "r"]
  outputNames = ["y"]

  data = NaiveONNX.readData(csvFile, inputNames, outputNames)
  visualizeData3D(data.train, data.inputNames, data.outputNames, [(1,2)], [1])
end
