using PastaQ
using Random

# Set the random seed so the results are the same
# each time it is run
Random.seed!(1234)

N = 4

# Number to QFT
number = 10
number_bin = digits(number, base = 2, pad = N) |> reverse
number_bin_st = prod(string.(number_bin))

println("Number of qubits: ",N)
println("Number for QFT: ",number," -> ",number_bin)
println()

#
# Make state |1010⟩
#

# Start with state |0000⟩
ψ0 = qubits(N)
# Apply gates X_1, X_3
starting_state = [number_bin[n] == 1 ? ("X", n) : ("I", n) for n in 1:N]

ψ0 = runcircuit(ψ0, starting_state)

samples = getsamples(ψ0, 5)
println("Sample from the initial state |$(number_bin_st)⟩:")
display(samples)
println()

#
# Make the QFT circuit
#

circuit = qft(N)

println("QFT circuit gates:")
display(circuit)
println()

println("Running QFT...")
println()
ψ = runcircuit(ψ0, circuit)

println("Sample from QFT|$(number_bin_st)⟩:")
samples = getsamples(ψ, 5)
display(samples)
println()

#
# Run inverse QFT
#

println("Running inverse QFT...")
println()
circuit⁻¹ = qft(N; inverse = true)
ψ = runcircuit(ψ, circuit⁻¹)

println("Sample from QFT⁻¹QFT|$(number_bin_st)⟩:")
samples = getsamples(ψ, 5)
display(samples)

