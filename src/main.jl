
struct InOutData
  input::Matrix{Float32}
  output::Matrix{Float32}

  InOutData(input::Matrix{Float32}, output::Matrix{Float32}) = new(input, output)
  InOutData(data::Tuple{Matrix{Float32},Matrix{Float32}}) = new(first(data), last(data))
end

struct Data
  train::InOutData
  test::InOutData
  inputNames::Array{String}
  outputNames::Array{String}

  function Data(train_in::Matrix{Float32}, train_out::Matrix{Float32}, test_in::Matrix{Float32}, test_out::Matrix{Float32}, inputNames::Array{String}, outputNames::Array{String})
    new(InOutData(train_in, train_out), InOutData(test_in, test_out), inputNames, outputNames)
  end
  function Data(train::Tuple{Matrix{Float32},Matrix{Float32}}, test::Tuple{Matrix{Float32},Matrix{Float32}}, inputNames::Array{String}, outputNames::Array{String})
    new(InOutData(train), InOutData(test), inputNames, outputNames)
  end
end

"""
    readData(filename, inputNames, outputNames; ratio=0.8)

Read CSV data and return training and testing data.

# Arguments
  - `filename::String`:           Path of CSV file with training data.
  - `inputNames::Array{String}`:  Array with model input names.
  - `outputNames::Array{String}`:  Array with model output names.

# Keywords
  - `ratio=0.8`: Ratio between training and testing data points.
              Defaults to 80% training and 20% testing.
  - `shuffle::Bool=true`: Shufle training and testing from data.
"""
function readData(filename::String,
                  inputNames::Array{String},
                  outputNames::Array{String};
                  ratio=0.8,
                  shuffle::Bool=true)::Data

  df = CSV.read(filename, DataFrames.DataFrame; ntasks=1)

  # Assert data is in expected order
  if names(df) != vcat(inputNames, outputNames)
    throw("Order of CSV file columns doesn't match given input/output variables.")
  end
  nInputs = length(inputNames)

  M = transpose(Matrix{Float32}(df))
  n_samples = size(M,2)
  n_train = Integer(round(n_samples*ratio))
  if shuffle
    trainIters = StatsBase.sample(1:n_samples, n_train, replace = false)
  else
    trainIters = 1:n_train
  end
  testIters = setdiff(1:n_samples, trainIters)

  train_in  = M[1:nInputs, trainIters]
  train_out = M[nInputs+1:end, trainIters]
  test_in   = M[1:nInputs, testIters]
  test_out  = M[nInputs+1:end, testIters]

  return Data(train_in, train_out, test_in, test_out, names(df)[1:nInputs], names(df)[nInputs+1:end])
end


"""
    trainSurrogate!(model_cpu, trainData; losstol=1e-6, nepochs=10, eta=1e-3, useGPU=true)

Train `model` on data from `dataloader`.
Stop when `losstol` or maximum number of epochs `nepochs` is reached.

# Arguments
  - `model`: Flux model to train.
  - `trainData::InOutData`: Train input and output data used for computing mean loss over total train set.

# Keywords
  - `losstol::Real=1e-6`: Loss to reach for model.
  - `nepochs::Integer=10`: Number of epochs to train.
  - `eta::Real=1e-3`: η parameter for Flux.ADAM.
  - `useGPU`: If true use GPU to train ANN, otherwise use CPU.
"""
function trainSurrogate!(model_cpu,
                         trainData::InOutData;
                         losstol::Real = 1e-6,
                         nepochs::Integer = 10,
                         eta::Real = 1e-3,
                         useGPU::Bool = true)

  dataloader = Flux.DataLoader((trainData.input, trainData.output), batchsize=64, shuffle=true)
  nInputs = size(trainData.input,1)

  opt = Flux.Adam(eta)
  lossFunc = Flux.Losses.mse

  local model
  if useGPU
    model = Flux.gpu(model_cpu)
  else
    model = model_cpu
  end
  parameters = Flux.params(model)

  initialLoss = sum([lossFunc(model_cpu(trainData.input[:,i]), trainData.output[:,i]) for i in 1:nInputs]) / size(trainData.input,2)
  @info "Initial loss: $(initialLoss)"

  for epoch in 1:nepochs
    if useGPU
      for (xtrain_batch, ytrain_batch) in dataloader
        x, y = Flux.gpu(xtrain_batch), Flux.gpu(ytrain_batch)
        gradients = Flux.gradient(() -> lossFunc(model(x), y), parameters)
        Flux.Optimise.update!(opt, parameters, gradients)
      end

      model_cpu = Flux.cpu(model)
      l = sum([lossFunc(model_cpu(trainData.input[:,i]), trainData.output[:,i]) for i in 1:nInputs]) / size(trainData.input,2)
      @info "Epoch $(epoch): Train loss=$(l)"
      if l < losstol
        break
      end
    else
      for (x, y) in dataloader
        gradients = gradient(() -> lossFunc(model(x), y), parameters)
        Flux.Optimise.update!(opt, parameters, gradients)
      end

      l = sum([lossFunc(model(trainData.input[:,i]), trainData.output[:,i]) for i in 1:nInputs]) / size(trainData.input,2)
      @info "Epoch $(epoch): Train loss=$(l)"
      if l < losstol
        break
      end
    end
  end

  return Flux.cpu(model)
end


"""
Train model on CSV data and export to ONNX.
"""
function trainONNX(csvFile::String,
                   onnxModel::String,
                   inputNames::Array{String},
                   outputNames::Array{String};
                   filterFunc=nothing,
                   model=nothing,
                   losstol::Real=1e-6,
                   nepochs=10)

  data = readData(csvFile, inputNames, outputNames)
  nInputs = length(inputNames)
  nOutputs = length(outputNames)

  if filterFunc !== nothing
    data = Data(filterFunc(data.train.input, data.train.output),
                filterFunc(data.test.input, data.test.output),
                data.inputNames, data.outputNames)
  end

  if model === nothing
  model = Flux.Chain(Flux.Dense(nInputs,     nInputs*10,  Flux.σ),
                     Flux.Dense(nInputs*10,  nOutputs*10, tanh),
                     Flux.Dense(nOutputs*10, nOutputs))
  end

  model = trainSurrogate!(model, data.train; losstol=losstol, nepochs=nepochs, useGPU=true)

  mkpath(dirname(onnxModel))
  BSON.@save onnxModel*".bson" model
  ONNXNaiveNASflux.save(onnxModel, model, (nInputs,1))

  return model
end


"""
Plot input agains outputs to visualize data.
"""
function visualizeData(data::InOutData,
                       inputNames::Array{String},
                       outputNames::Array{String};
                       inidx=nothing,
                       outidx=nothing)
  if inidx === nothing
    inidx = 1:length(data.input[1])
  end

  if outidx === nothing
    outidx = 1:length(data.output[2])
  end

  p = []

  for i in inidx
    for j in outidx
      plt = Plots.scatter(data.input[i,:], data.output[j,:], xlabel=inputNames[i], label=[outputNames[j]])
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

  for (out1, (in1, in2)) in enumerate(inidx)
    j = outidx[out1]
    plt = Plots.scatter(data.input[in1,:], data.input[in2,:], data.output[j,:], xlabel=inputNames[in1], ylabel=inputNames[in2], zlabel=outputNames[j])
    push!(p, plt)
  end

  Plots.plot(p..., legend=false)
end
