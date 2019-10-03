# prediction from single tree - assign each observation to its final leaf
function predict!(pred, tree::Tree, X::AbstractArray{T, 2}) where {T<:Real}
    @inbounds @threads for i in 1:size(X, 1)
        id = 1
        x = view(X, i, :)
        while tree.nodes[id].split
            if x[tree.nodes[id].feat] < tree.nodes[id].cond
                id = tree.nodes[id].left
            else
                id = tree.nodes[id].right
            end
        end
        pred[i] += tree.nodes[id].pred
    end
    return pred
end

# prediction from single tree - assign each observation to its final leaf
function predict(tree::Tree, X::AbstractArray{T, 2}, K) where T<:Real
    pred = zeros(SVector{K,T}, size(X, 1))
    predict!(pred, tree, X)
    return pred
end

# prediction from single tree - assign each observation to its final leaf
function predict(model::GBTree, X::AbstractArray{T, 2}) where T<:Real
    pred = zeros(SVector{model.params.K,T}, size(X, 1))
    for tree in model.trees
        predict!(pred, tree, X)
    end
    pred = reinterpret(T, pred)
    if typeof(model.params.loss) == Logistic
        @. pred = sigmoid(pred)
    elseif typeof(model.params.loss) == Poisson
        @. pred = exp(pred)
    elseif typeof(model.params.loss) == Softmax
        for row in eachrow(pred)
            row .= softmax(row)
        end
    end
    return pred
end


# prediction in Leaf - GradientRegression
# function pred_leaf(loss::S, node::TrainNode, params::EvoTreeRegressor, δ²) where {S<:GradientRegression, T<:AbstractFloat}
#     # pred = zeros(length(node.∑δ))
#     # for  i in 1:length(node.∑δ)
#     SVector(params.η) .* node.∑δ ./ (node.∑δ² .+ SVector(params.λ) .* node.∑𝑤)
#     # end
#     # return pred
# end
function pred_leaf(loss::S, node::TrainNode{L,T}, params::EvoTreeRegressor, δ²) where {S<:GradientRegression,L,T}
    - params.η .* node.∑δ ./ (node.∑δ² .+ params.λ .* node.∑𝑤)
end

# prediction in Leaf - MultiClassRegression
function pred_leaf(loss::S, node::TrainNode{L,T}, params::EvoTreeRegressor, δ²) where {S<:MultiClassRegression,L,T}
    pred = zeros(length(node.∑δ))
    for  i in 1:length(node.∑δ)
        pred[i] -= params.η * node.∑δ[i] / (node.∑δ²[i] + params.λ * node.∑𝑤[1])
    end
    return pred
end


# prediction in Leaf - L1Regression
function pred_leaf(loss::S, node::TrainNode{L,T}, params::EvoTreeRegressor, δ²) where {S<:L1Regression,L,T}
    params.η .* node.∑δ ./ (node.∑𝑤 .* (1 .+ params.λ))
end
# prediction in Leaf - L1Regression
# function pred_leaf(loss::S, node::TrainNode, params::EvoTreeRegressor, δ²) where {S<:L1Regression, T<:AbstractFloat}
#     pred = zeros(length(node.∑δ))
#     for  i in 1:length(node.∑δ)
#         pred[i] += params.η * node.∑δ[i] / (node.∑𝑤[1] * (1 + params.λ))
#     end
#     # pred = params.η * node.∑δ ./ (node.∑𝑤 * (1 + params.λ))
#     return pred
# end

# prediction in Leaf - QuantileRegression
function pred_leaf(loss::S, node::TrainNode{L,T}, params::EvoTreeRegressor, δ²) where {S<:QuantileRegression,L,T}
    pred = [params.η * quantile(reinterpret(Float64, δ²[collect(node.𝑖)]), params.α) / (1 + params.λ)]
    # pred = params.η * quantile(δ²[collect(node.𝑖)], params.α) / (1 + params.λ)
    return pred
end
