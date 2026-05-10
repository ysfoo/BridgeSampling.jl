using Test, BridgeSampling, SpecialFunctions, Turing

## Test on bivariate standard Normal distribution
samples = rand(MvNormal([1.0 0.0; 0.0 1.0]), 50_000)

log_posterior(x) = -0.5 * x' * x

Bridge = bridgesampling(samples, log_posterior, [-Inf, -Inf], [Inf, Inf])
cv = error_estimate(Bridge).cv
Analytical = log(2*pi)

@test isapprox(value(Bridge), Analytical, atol = 5cv)


## Test on y ~ Exp(1/λ), λ ~ Gamma(α, β)
λ_true = 2.0
n = 10
y = rand(Exponential(1/λ_true), n)
α = 2.0
β = 1.0

@model function exp_model(y)    
    λ ~ Gamma(α, β)
    for i in eachindex(y)
        y[i] ~ Exponential(1/λ)
    end
end

model = exp_model(y)
chain = sample(model, NUTS(), MCMCThreads(), 50_000, 2; progress=false, verbose=false)
Bridge = bridgesampling(chain, model)
cv = error_estimate(Bridge).cv
Analytical = α*log(β) - loggamma(α) + loggamma(α + n) - (α + n)*log(β + sum(y))

@test isapprox(value(Bridge), Analytical, atol = 5cv)


## Test on y ~ NegativeBinomial(r, p), p ~ Beta(α, β)
r = 5
n = 10
y = [4, 2, 0, 6, 1, 3, 2, 1, 0, 4]
α = 2.0
β = 3.0
@model function nb_model(y, r, α, β)
    p ~ Beta(α, β)                      
    for i in eachindex(y)
        y[i] ~ NegativeBinomial(r, p)
    end
end
model = nb_model(y, r, α, β)
chain = sample(model, NUTS(), MCMCThreads(), 50_000, 2; progress=false, verbose=false)
Bridge = bridgesampling(chain, model)
cv = error_estimate(Bridge).cv
Analytical = sum( loggamma.(y .+ r) .- loggamma.(y .+ 1) .- loggamma(r) ) + logbeta(α + n*r, β + sum(y)) - logbeta(α, β)

@test isapprox(value(Bridge), Analytical, atol = 5cv)
