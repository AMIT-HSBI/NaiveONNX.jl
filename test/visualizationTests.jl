using Test
using NaiveONNX

@testset "eq_928" begin
  csvFile = abspath(joinpath(@__DIR__, "csv", "eq_928.csv"))
  nInputs = 1

  data = NaiveONNX.readData(csvFile, nInputs)
  visualizeData(data.train, data.inputNames, data.outputNames, outidx=[1,5])
end

@testset "simpleLoop_eq14" begin
  csvFile = abspath(joinpath(@__DIR__, "csv", "simpleLoop_eq14.csv"))
  nInputs = 2

  data = NaiveONNX.readData(csvFile, nInputs)
  visualizeData(data.train, data.inputNames, data.outputNames)
end

@testset "simpleLoop_eq14 3D" begin
  csvFile = abspath(joinpath(@__DIR__, "csv", "simpleLoop_eq14.csv"))
  nInputs = 2

  data = NaiveONNX.readData(csvFile, nInputs)
  visualizeData3D(data.train, data.inputNames, data.outputNames, [(1,2)], [1])
end
