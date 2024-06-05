module NaiveONNX

import BSON
import CSV
import CUDA
import DataFrames
import Flux
import InvertedIndices
import ONNXNaiveNASflux
import Plots
import StatsBase

include("main.jl")
export prepareData
export trainONNX
export trainSurrogate!
export visualizeData
export visualizeData3D

end # module NaiveONNX
