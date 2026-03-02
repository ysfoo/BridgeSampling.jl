## 
function get_pdf_samples(samples, trans_samples, log_posterior, prop_dist, lb, ub, names)
    n = size(samples, 2)
    n == size(trans_samples,2) || throw(DimensionMismatch("samples dimensions don't match"))

    g = zeros(n)
    p = zeros(n)

    for i in 1:n
        g_sample = @views(trans_samples[:,i])
        p_sample = @views(samples[:,i])
        logdetJ = sum(logjacobian_contribution.(transform.(p_sample, lb, ub), lb, ub)) 
        g[i] = logpdf(prop_dist, g_sample)
        p[i] = isnothing(names) ? log_posterior(p_sample) + logdetJ : log_posterior(NamedTuple{Tuple(names)}(p_sample)) + logdetJ
    end

    return p, g, p-g
end


## Iterative algo
# Note that for ESS, the case of multiple chains is not dealt with correctly.
function iterative_algorithm(l₁, l₂, n₁, n₂; tol, maxiter, use_ess)

    lstar = median(l₁)
    r = exp(logsumexp(l₂) - log(n₂) - lstar)
    s₁ = n₁ / (n₁ + n₂)
    s₂ = n₂ / (n₁ + n₂)

    logml = NaN
    denomterms = zeros(n₁)
    for i = 1:maxiter
        for j1 in 1:n₁
            el1 = exp(l₁[j1] - lstar)
            denomterms[j1] = 1 / (s₁ * el1 + s₂ * r)
        end
        numterm = 0.0
        for l₂ⱼ in l₂
            el2 = exp(l₂ⱼ - lstar)
            numterm += el2 / (s₁ * el2 + s₂ * r)
        end
        rnew = (numterm/n₂) / (sum(denomterms)/n₁)
        
        logmlnew = log(rnew) + lstar
        ϵ = abs(logmlnew - logml)
        if ϵ < tol
            return logmlnew, i
        end
        logml = logmlnew
        r = rnew
        if use_ess
            n1_ess = n₁ / ess(denomterms)
            s₁ = n1_ess / (n1_ess + n₂)
            s₂ = n₂ / (n1_ess + n₂)
        end
    end
    @warn "Maximum number of iterations ($maxiter) reached before convergence under the tolerance level $tol"
    return logml, maxiter
end