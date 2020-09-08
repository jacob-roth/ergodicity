##
## dynamcis
##

function gbm(x0::R, μ::R, σ::R, nsteps::Z=1_000, Δt::R=0.01) where R <: Real where Z <: Integer
    xs = zeros(nsteps)
    ξ  = randn(nsteps)
    x  = x0
    for i in 1:nsteps
        xs[i] = x
        Δx = x * (μ * Δt + σ * ξ[i] * sqrt(Δt))
        x += Δx
    end
    return xs
end

function gbms(x0::R, μ::R, σ::R, nsteps::Z=1_000, Δt::R=0.01, nsims::Z=100) where R <: Real where Z <: Integer
    xs = []
    for i in 1:nsims
        push!(xs, gbm(x0, μ, σ, nsteps, Δt))
    end
    return hcat(xs...)
end

# function g_est(x::Array{R,2}, tgrid::Array{R,1}, t::R, n::Z, x0::R) where R <: Real where Z <: Integer
#     xx = deepcopy(x)
#     xx ./= x0
#     t_idx = findall(t .== tgrid)
#     nsamples = Int(floor(size(x,2)/n, digits=0))
#     g = sum(
#             log( sum(xx[t_idx, (i-1)*n+1:(i*n)])/n ) / t for i in 1:nsamples
#             ) / nsamples
#     return g
# end
function g_est(x::Array{R,2}, t_idx::Z, n::Z, x0::R) where R <: Real where Z <: Integer
    xx = deepcopy(x)
    xx ./= x0
    nsamples = Int(floor(size(x,2)/n, digits=0))
    g = sum( log( sum(xx[t_idx, (i-1)*n+1:(i*n)])/n ) / t_idx for i in 1:nsamples ) / nsamples
    return g
end
function g_est(x::Array{R,2}, t::R, n::Z, x0::R) where R <: Real where Z <: Integer
    xx = deepcopy(x)
    xx ./= x0
    t_idx = findall(t .== tgrid)
    nsamples = Int(floor(size(x,2)/n, digits=0))
    g = sum( log( sum(xx[t_idx, (i-1)*n+1:(i*n)])/n ) / t for i in 1:nsamples ) / nsamples
    return g
end


function cointoss(x0::R, p::Array{R,1}, v::Array{R,1}, nsteps::Z=1_000, bet_pct::R=0.5) where R <: Real where Z <: Integer
    """
    we will see how gbm is an `attractor` for different types of multiplicative growth
    """
    c = cumsum(p)
    cc = [0; c[1:end-1]]
    xs = zeros(nsteps)
    ξ  = rand(nsteps)
    x  = x0
    for i in 1:nsteps
        xs[i] = x
        idx = first(findall((ξ[i] .< c) .* (ξ[i] .>= cc)))
        Δx = (bet_pct * x) * v[idx]
        x += Δx
    end
    return xs
end

function cointosses(x0::R, p::Array{R,1}, v::Array{R,1}, nsteps::Z=1_000, bet_pct::R=0.5, nsims::Z=100) where R <: Real where Z <: Integer
    xs = []
    for i in 1:nsims
        push!(xs, cointoss(x0, p, v, nsteps, bet_pct))
    end
    return hcat(xs...)
end


##
## simulations
##

using Pkg; Pkg.activate(".")
using PyPlot
using Random
Random.seed!(12345)

## simulate
const nsteps = 10_000
const Δt = 0.01
const nsims = 1_000
const μ = 1.0
const σ = 2.0
const x0 = 100.0
tgrid = collect(range(0.0, length=nsteps, step=Δt))
xs = gbms(x0, μ, σ, nsteps, Δt, nsims)
xs[xs .< 0] .= 1e-321

## plot
fig, ax = subplots()
ax.cla()
ax.plot(xs)
ax.set_yscale("log")

const p = [0.2, 0.8]
const v = [5.1, -1.0]
const bet_pct = 0.05
ys = cointosses(x0, p, v, nsteps, bet_pct, nsims)
ys10 = cointosses(x0, p, v, 10nsteps, bet_pct, nsims)

##
## averages
##

const nt = 10
const nn = 20
ntgrid = round.(collect(range(1.0, stop=maximum(tgrid), length=nt)), digits=2)
nngrid = Int.(round.(collect(range(1, stop=nsims, length=nn)), digits=0))
gxs = zeros(nt, nn)
gys = zeros(nt, nn)
for i in enumerate(ntgrid)
    for j in enumerate(nngrid)
        gxs[i[1],j[1]] = g_est(xs, i[2], j[2], x0)
        gys[i[1],j[1]] = g_est(ys, first(findall(i[2] .== tgrid)), j[2], x0)
    end
end

gxbar = μ - σ^2/2
gxens = μ

gybar = sum(p .*v ) - (p[1]*p[2])/bet_pct