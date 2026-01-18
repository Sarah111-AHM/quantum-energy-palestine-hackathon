# quantum_simulation.jl
# Advanced quantum simulation using Julia

println("============================================")
println("        Quantum Simulation - Julia")
println("============================================")

struct QuantumSite
    id::Int
    name::String
    latitude::Float64
    longitude::Float64
    risk::Float64
    access::Float64
    priority::Float64
    population::Int
    cost::Float64
end

struct QUBOSolver
    lambda_k::Float64
    lambda_d::Float64
    lambda_b::Float64
end

function haversine_distance(lat1, lon1, lat2, lon2)
    R = 6371.0
    dlat = deg2rad(lat2 - lat1)
    dlon = deg2rad(lon2 - lon1)
    a = sin(dlat/2)^2 + cos(deg2rad(lat1)) * cos(deg2rad(lat2)) * sin(dlon/2)^2
    c = 2 * atan(√a, √(1-a))
    return R * c
end

function generate_test_sites(n::Int=20)
    println("Generating $n test sites...")
    sites = Vector{QuantumSite}()
    regions = [
        ("North Gaza", 31.55, 34.50, 0.7),
        ("Gaza City", 31.50, 34.46, 0.5),
        ("Central", 31.45, 34.40, 0.4),
        ("Khan Yunis", 31.34, 34.30, 0.6),
        ("Rafah", 31.29, 34.25, 0.8)
    ]
    types = [
        ("Hospital", 0.9, 0.6, 200000),
        ("School", 0.8, 0.7, 80000),
        ("Camp", 0.9, 0.4, 150000),
        ("Water Station", 0.8, 0.6, 120000)
    ]
    for i in 1:n
        r = mod1(i, length(regions))
        t = mod1(i, length(types))
        lat = regions[r][2] + rand(-0.01:0.0001:0.01)
        lon = regions[r][3] + rand(-0.01:0.0001:0.01)
        risk = clamp(regions[r][4] + rand(-0.15:0.01:0.15), 0.1, 0.95)
        access = clamp(types[t][3] + rand(-0.2:0.01:0.2), 0.1, 0.95)
        priority = clamp(types[t][2] + rand(-0.1:0.01:0.1), 0.3, 1.0)
        population = rand(1000:20000)
        cost = types[t][4] + rand(-30000:1000:30000)
        push!(sites, QuantumSite(i, "$(types[t][1]) $i", round(lat,6), round(lon,6),
            round(risk,3), round(access,3), round(priority,3), population, round(cost,2)))
    end
    println("Generated $(length(sites)) sites.")
    return sites
end

function calculate_site_score(site::QuantumSite, weights=Dict("risk"=>0.35,"access"=>0.25,"priority"=>0.3))
    return round(weights["risk"]*(1 - site.risk) + weights["access"]*site.access + weights["priority"]*site.priority, 4)
end

function build_qubo_matrix(sites::Vector{QuantumSite}, solver::QUBOSolver, K::Int, budget::Float64, min_distance::Float64)
    n = length(sites)
    scores = [calculate_site_score(s) for s in sites]
    Q = zeros(Float64, n, n)
    # objective
    for i in 1:n
        Q[i,i] += -scores[i]
    end
    # K-selection constraint
    for i in 1:n, j in 1:n
        Q[i,j] += i==j ? solver.lambda_k*(1 - 2*K) : 2*solver.lambda_k
    end
    # distance constraint
    for i in 1:n, j in (i+1):n
        d = haversine_distance(sites[i].latitude, sites[i].longitude, sites[j].latitude, sites[j].longitude)
        if d < min_distance
            p = solver.lambda_d*(1 - d/min_distance)
            Q[i,j] += p; Q[j,i] += p
        end
    end
    # budget constraint
    if budget > 0
        for i in 1:n
            Q[i,i] += solver.lambda_b * sites[i].cost^2 - 2 * solver.lambda_b * budget * sites[i].cost
        end
    end
    return Q
end

function simulate_qaoa(Q::Matrix{Float64}, p::Int=1, shots::Int=1000)
    n = size(Q,1)
    results = Dict{String,Float64}()
    for _ in 1:shots
        state = [rand() < 0.5 for _ in 1:n]
        energy = sum(Q[i,j]*state[i]*state[j] for i in 1:n, j in 1:n)
        results[join(Int.(state))] = energy
    end
    best_state, best_energy = findmin(collect(values(results)))
    selected_sites = findall(c->c=='1', keys(results)[best_state])
    return selected_sites, best_energy, keys(results)[best_state]
end

function compare_with_classical(sites::Vector{QuantumSite}, Q::Matrix{Float64}, K::Int)
    scores = [calculate_site_score(s) for s in sites]
    top_k = sortperm(scores, rev=true)[1:K]
    classical_score = sum(scores[top_k])
    solver = QUBOSolver(1.0,0.5,0.3)
    Q_matrix = build_qubo_matrix(sites, solver, K, 500000.0, 3.0)
    quantum_selected, quantum_energy, _ = simulate_qaoa(Q_matrix,1,500)
    quantum_score = sum(scores[quantum_selected])
    improvement = ((quantum_score - classical_score)/classical_score)*100
    return (classical=top_k, quantum=quantum_selected, classical_score=classical_score, quantum_score=quantum_score, improvement=improvement)
end

function main()
    println("Select option: 1-Quick test  2-Full analysis  3-Exit")
    choice = readline()
    if choice=="1"
        sites = generate_test_sites(3)
    elseif choice=="2"
        sites = generate_test_sites(15)
    else
        println("Exit.")
        return
    end
    Q = build_qubo_matrix(sites, QUBOSolver(1.0,0.5,0.3), 5, 500000.0, 3.0)
    selected, energy, _ = simulate_qaoa(Q,2,1000)
    println("Selected sites: ", selected)
    comparison = compare_with_classical(sites, Q, 5)
    println("Comparison: ", comparison)
end

if abspath(PROGRAM_FILE) == @__FILE__
    main()
end
