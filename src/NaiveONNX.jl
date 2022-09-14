module NaiveONNX

import CSV
import DataFrames
import Flux
import ONNX
import ONNXNaiveNASflux
import Plots
import StatsBase

# Add Flux.sigmoid operation for ONNXNaiveNASflux
Flux.sigmoid(pp::ONNXNaiveNASflux.AbstractProbe) = ONNXNaiveNASflux.attribfun(identity, "Sigmoid", pp)
ONNXNaiveNASflux.refresh()

include("main.jl")
export trainONNX
export visualizeData
export visualizeData3D

end # module NaiveONNX
