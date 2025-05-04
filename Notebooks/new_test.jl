# Import required packages
using QXTools
using Pkg
Pkg.activate(@__DIR__)

using QXTns
using QXZoo
using PyCall
using QXGraphDecompositions
using LightGraphs
using DataStructures
using TimerOutputs
using ITensors
using LinearAlgebra
using NDTensors

# Ensure proper threading is available
if Threads.nthreads() == 1
    @warn "Running with only 1 thread. For better performance, start Julia with multiple threads using -t auto or -t N"
end

# Set up OpenMP environment variables
ENV["JULIA_LLVM_ARGS"] = "-enable-llvm-openmp"
# Enable multi-threading for BLAS operations
using LinearAlgebra
BLAS.set_num_threads(Threads.nthreads())

# Print system threading information
println("Julia threads: ", Threads.nthreads())
println("BLAS threads: ", BLAS.get_num_threads())

# Load custom functions from our optimized module
include("../src/functions_article_mp.jl");

# Create a GHZ circuit with 10 qubits
circuit = create_ghz_circuit(10)

# Convert the circuit to a tensor network circuit (TNC)
tnc = convert_to_tnc(circuit)

# Configure the contraction parameters
num_communities = 4  # Number of communities for multistage algorithm
input_state = "0" ^ 10  # All qubits initialized to 0
output_state = "1" ^ 10  # Target output state

# Run benchmarks for different parallel implementations
println("\n=========== BENCHMARKING DIFFERENT METHODS ===========")

# Standard sequential algorithm
println("\n1. Running sequential GN algorithm...")
result_seq = Calcul_GN_Sequencial(circuit, true)
println("Sequential result: ", result_seq)

# Standard ComPar algorithm with OpenMP
println("\n2. Running ComParCPU with OpenMP...")
result_compar = ComParCPU(circuit, input_state, output_state, num_communities;
                        timings=true, decompose=true)
println("ComParCPU result: ", result_compar)

# Parallel final contraction with OpenMP
println("\n3. Running ComParCPU_para with OpenMP acceleration...")
result_para = ComParCPU_para(circuit, input_state, output_state, num_communities;
                          timings=true, decompose=true)
println("ComParCPU_para result: ", result_para)

# Run the FastGreedy community detection version
println("\n4. Running FastGreedy community detection with OpenMP...")
result_fg = ComParCPU_GHZ(circuit, input_state, output_state;
                        timings=true, decompose=true)
println("FastGreedy result: ", result_fg)

# Run the two-level parallelism version (communities + final contraction)
println("\n5. Running two-level parallelism with OpenMP...")
result_two_level = ComParCPU_para_GHZ(circuit, input_state, output_state;
                                    timings=true, decompose=true)
println("Two-level parallelism result: ", result_two_level)

# Compare all results (they should be the same)
println("\n=========== RESULT VERIFICATION ===========")
println("All results match: ", 
        isapprox(result_seq, result_compar) && 
        isapprox(result_compar, result_para) && 
        isapprox(result_para, result_fg) &&
        isapprox(result_fg, result_two_level))

# Print detailed timing comparison
println("\n=========== SUMMARY ===========")
println("Test completed successfully!")