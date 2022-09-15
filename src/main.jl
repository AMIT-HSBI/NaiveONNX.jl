
struct InOutData
  input::Vector{Vector{Float32}}
  output::Vector{Vector{Float32}}

  InOutData(input::Vector{Vector{Float32}}, output::Vector{Vector{Float32}}) = new(input, output)
  InOutData(data::Tuple{Vector{Vector{Float32}},Vector{Vector{Float32}}}) = new(first(data), last(data))
end

struct Data
  train::InOutData
  test::InOutData
  inputNames::Array{String}
  outputNames::Array{String}

  function Data(train_in::Vector{Vector{Float32}}, train_out::Vector{Vector{Float32}}, test_in::Vector{Vector{Float32}}, test_out::Vector{Vector{Float32}}, inputNames::Array{String}, outputNames::Array{String})
    new(InOutData(train_in, train_out), InOutData(test_in, test_out), inputNames, outputNames)
  end
  function Data(train::Tuple{Vector{Vector{Float32}},Vector{Vector{Float32}}}, test::Tuple{Vector{Vector{Float32}},Vector{Vector{Float32}}}, inputNames::Array{String}, outputNames::Array{String})
    new(InOutData(train), InOutData(test), inputNames, outputNames)
  end
end

"""
    readData(filename, nInputs; ratio=0.8)

Read CSV data and return training and testing data.

# Arguments
  - `filename::String`: Path of CSV file with training data.
  - `nInputs::Integer`: Number of input varaibles for model.

# Keywords
  - `ratio=0.8`: Ratio between training and testing data points.
              Defaults to 80% training and 20% testing.
  - `shuffle::Bool=true`: Shufle training and testing from data.
"""
function readData(filename::String, nInputs::Integer; ratio=0.8, shuffle::Bool=true)::Data
  df = CSV.read(filename, DataFrames.DataFrame)
  m = Matrix{Float32}(df)
  n = length(m[:,1])
  num_train = Integer(round(n*ratio))
  if shuffle
    trainIters = StatsBase.sample(1:n, num_train, replace = false)
  else
    trainIters = 1:num_train
  end
  testIters = setdiff(1:n, trainIters)

  train_in  = [m[i, 1:nInputs]     for i in trainIters]
  train_out = [m[i, nInputs+1:end] for i in trainIters]
  test_in   = [m[i, 1:nInputs]     for i in testIters]
  test_out  = [m[i, nInputs+1:end] for i in testIters]

  return Data(train_in, train_out, test_in, test_out, names(df)[1:nInputs], names(df)[nInputs+1:end])
end


"""
    trainSurrogate!(model, dataloader, trainData; losstol=1e-6, nepochs=10, eta=1e-3)

Train `model` on data from `dataloader`.
Stop when `losstol` or maximum number of epochs `nepochs` is reached.

# Arguments
  - `model`: Flux model to train.
  - `dataloader::Flux.DataLoader`: DataLoader with trainig data.
  - `trainData::InOutData`: Train input and output data used for computing mean loss over total train set.

# Keywords
  - `losstol::Real=1e-6`: Loss to reach for model.
  - `nepochs::Integer=10`: Number of epochs to train.
  - `eta::Real=1e-3`: η parameter for Flux.ADAM.
"""
function trainSurrogate!(model, dataloader::Flux.DataLoader, trainData::InOutData; losstol::Real=1e-6, nepochs::Integer=10, eta::Real=1e-3)
  ps = Flux.params(model)
  opt = Flux.Adam(eta)
  loss(x,y) = Flux.mse(model(x),y)
  meanloss() = sum(loss.(trainData.input, trainData.output))/length(trainData.input)
  @info "Initial loss: $(meanloss())"

  for epoch in 1:nepochs
    for (ti,to) in dataloader
      Flux.train!(loss, ps, zip(ti,to), opt)
    end
    l = meanloss()
    @info "Epoch $(epoch): Train loss=$(l)"
    if l < losstol
      break
    end
  end
end


"""
Train model on CSV data and export to ONNX.
"""
function trainONNX(csvFile::String, onnxModel::String, nInputs::Integer; filterFunc=nothing, model=nothing, losstol::Real=1e-6, nepochs=10)
  data = readData(csvFile, nInputs)
  nOutputs = length(data.train.output[1])

  if filterFunc !== nothing
    data = Data(filterFunc(data.train.input, data.train.output),
                filterFunc(data.test.input, data.test.output),
                data.inputNames, data.outputNames)
  end

  dataloader = Flux.DataLoader((data.train.input, data.train.output), batchsize=64, shuffle=true)

  if model === nothing
  model = Flux.Chain(Flux.Dense(nInputs,     nInputs*10,  Flux.σ),
                     Flux.Dense(nInputs*10,  nOutputs*10, tanh),
                     Flux.Dense(nOutputs*10, nOutputs))
  end

  trainSurrogate!(model, dataloader, data.train; losstol=losstol, nepochs=nepochs)

  mkpath(dirname(onnxModel))
  ONNXNaiveNASflux.save(onnxModel, model, (nInputs,1))
  return onnxModel
end


"""
Plot input agains outputs to visualize data.
"""
function visualizeData(data::InOutData, inputNames::Array{String}, outputNames::Array{String}; inidx=nothing, outidx=nothing)
  if inidx === nothing
    inidx = 1:length(data.input[1])
  end

  if outidx === nothing
    outidx = 1:length(data.output[2])
  end

  p = []

  inData = reduce(hcat, data.input)
  outData = reduce(hcat, data.output)

  for i in inidx
    for j in outidx
      plt = Plots.scatter(inData[i,:], outData[j,:], xlabel=inputNames[i], label=[outputNames[j]])
      push!(p, plt)
    end
  end

  Plots.plot(p..., layout=(length(inidx), length(outidx)), legend=true)
end


"""
Array of two inputs to plot against one output.
"""
function visualizeData3D(data::InOutData, inputNames::Array{String}, outputNames::Array{String}, inidx::Array{Tuple{I,I}}, outidx) where I<:Integer
  @assert length(inidx) == length(outidx) "Length of inidx not equal length of outidx"
  p = []

  inData = reduce(hcat, data.input)
  outData = reduce(hcat, data.output)

  for (out1, (in1, in2)) in enumerate(inidx)
    j = outidx[out1]
    plt = Plots.scatter(inData[in1,:], inData[in2,:], outData[j,:], xlabel=inputNames[in1], ylabel=inputNames[in2], zlabel=outputNames[j])
    push!(p, plt)
  end

  Plots.plot(p..., legend=false)
end
