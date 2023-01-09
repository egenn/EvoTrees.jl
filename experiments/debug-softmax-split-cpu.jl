using Revise
using Statistics
using StatsBase: sample
using EvoTrees
using BenchmarkTools

using CSV, DataFrames, MLJBase, EvoTrees
using StableRNGs

data = CSV.read(joinpath(@__DIR__, "..", "data", "debug", "pb_data.csv"), DataFrame)
y = categorical(data.target)
X = data[!, Not(:target)]

train, test = MLJBase.train_test_pairs(Holdout(), 1:size(X, 1), X, y)[1]
rng = StableRNG(6)
model = EvoTreeClassifier(nrounds = 5, lambda = 1e-5, max_depth = 7, rng = rng)
Xtrain, ytrain = MLJBase.reformat(model, selectrows(X, train), selectrows(y, train))
# MLJBase.fit(model, 1, Xtrain, ytrain);

# EvoTrees params
rng = StableRNG(6)
params_evo = EvoTreeClassifier(;
    T = Float32,
    nrounds = 5,
    lambda = 0,
    gamma = 0.0,
    eta = 0.1,
    max_depth = 7,
    min_weight = 1.0,
    rowsample = 1.0,
    colsample = 1.0,
    nbins = 64,
    rng,
)

using CategoricalArrays
x_train = Xtrain[:matrix]
y_train = CategoricalArrays.levelcode.(ytrain)

mean(y_train)
sum(ytrain .== true) ./ length(y_train)

@info "evotrees train CPU:"
params_evo.device = :cpu
@time m_evo = fit_evotree(params_evo; x_train, y_train, x_eval = x_train, y_eval = y_train, metric=:logloss, print_every_n = 1);
# @time m_evo = fit_evotree(params_evo; x_train, y_train);
# @time m_evo = fit_evotree(params_evo; x_train, y_train);

# function h1(h, hL, hR, ∑, K, nbins)
#     KK = 2 * K + 1
#     @inbounds for j in js
#         @inbounds for k = 1:KK
#             val = h[k, 1, j]
#             hL[k, 1, j] = val
#             hR[k, 1, j] = ∑[k, j] - val
#         end
#         @inbounds for bin = 2:nbins
#             @inbounds for k = 1:KK
#                 val = h[k, bin, j]
#                 hL[k, bin, j] = hL[k, bin-1, j] + val
#                 hR[k, bin, j] = hR[k, bin-1, j] - val
#             end
#         end
#     end
#     return hR
# end

# function h2(h, hL, hR, nbins)
#     cumsum!(hL, h, dims = 2)
#     hR .= view(hL, :, nbins:nbins, :) .- hL
#     return hR
# end

# nbins = 64
# js = 12
# K = 2
# h = rand(2*K+1, nbins, js)
# hL = zeros(2*K+1, nbins, js)
# hR = zeros(2*K+1, nbins, js)
# ∑ = dropdims(sum(h, dims=2), dims=2)

# x1 = h1(h, hL, hR, ∑, K, nbins)
# x2 = h2(h, hL, hR, nbins)

# minimum(x1 .- x2)
# maximum(x1 .- x2)


"""
    Reproducibility attempt

Produces consistent results...
"""
mutable struct Node
    h::Any
    hL::Any
    hR::Any
    ∑::Any
end

nbins = 16
js = 12
K = 2
x = rand(2 * K + 1, nbins)
h = zeros(2 * K + 1, nbins, js)
hL = zeros(2 * K + 1, nbins, js)
hR = zeros(2 * K + 1, nbins, js)

for j = 1:js
    ids = shuffle(1:nbins)
    h[:, :, j] .= x[:, ids]
end
sum(h, dims = 2)
∑ = sum(h, dims = 2)[:, end, end]
node = Node(h, hL, hR, ∑);

function h1_A(node, K, nbins, js)

    KK = 2 * K + 1
    h = node.h
    hL = node.hL
    hR = node.hR
    ∑ = node.∑

    # h = deepcopy(node.h)
    # hL = deepcopy(node.hL)
    # hR = deepcopy(node.hR)
    # ∑ = deepcopy(node.∑)
    @inbounds for j in js
        @inbounds for k = 1:KK
            val = h[k, 1, j]
            hL[k, 1, j] = val
            hR[k, 1, j] = ∑[k] - val
        end
        @inbounds for bin = 2:nbins
            @inbounds for k = 1:KK
                val = h[k, bin, j]
                hL[k, bin, j] = hL[k, bin-1, j] + val
                hR[k, bin, j] = hR[k, bin-1, j] - val
            end
        end
    end

    hL2 = copy(node.hL)
    hR2 = copy(node.hR)
    cumsum!(hL2, h, dims = 2)
    hR2 .= view(hL2, :, nbins:nbins, :) .- hL2

    # @info hR
    # @info hR2
    @info "max abs diff hL" maximum(abs.(hL[3, :, :] .- hL2[3, :, :]))
    @info "max abs diff hR" maximum(abs.(hR[3, :, :] .- hR2[3, :, :]))
    return (hL = hL, hL2 = hL2)
end
x = h1_A(node, K, nbins, 1:js);
