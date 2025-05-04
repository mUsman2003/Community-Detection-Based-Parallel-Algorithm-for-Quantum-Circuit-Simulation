# # Add necessary packages
# import Pkg

# # Install required packages if needed
# Pkg.add("QXTools")
# Pkg.add("QXGraphDecompositions")
# Pkg.add("QXZoo")
# Pkg.add("DataStructures")
# Pkg.add("QXTns")
# Pkg.add("NDTensors")
# Pkg.add("ITensors")
# Pkg.add("LightGraphs")
# Pkg.add("PyCall")
# Pkg.add("LLVMOpenMP_jll")  # Using LLVMOpenMP_jll instead of OpenMP_jll

# Using required modules
using QXTools
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
# Using Base.Threads for thread management
using Base.Threads
using LLVMOpenMP_jll  # Changed from OpenMP_jll to LLVMOpenMP_jll
using Distributed

# Remove any GPU-related code
# No ParallelStencil or CUDA initialization

# Load custom functions from the folder src
include("../src/functions_article.jl")
include("../src/TensorContraction_OpenMP.jl")

# Set Julia's threading environment variable before running
# This is in addition to OpenMP threads
ENV["JULIA_NUM_THREADS"] = 8

# You might also want to set OpenMP thread count directly
ENV["OMP_NUM_THREADS"] = 8

# Create a circuit
n = 10
circuit = create_qft_circuit(n)

# Run with LLVMOpenMP_jll
input = "0"^n
output = "0"^n
num_threads = 8  # Set to the number of cores you want to use

# Configure the contraction algorithm
num_communities = 4
input = "0"^100
output = "0"^100
convert_to_tnc(circuit; input=input, output=output, decompose=true)

println("Successfully converted to TNC")

try
    # Make sure we're using the OpenMP function with updated LLVMOpenMP_jll
    set_openmp_threads(num_threads)
    println("Confirmed OpenMP threads: ", get_openmp_threads())
    
    result = ComParCPU_OpenMP(circuit, input, output, num_communities, num_threads)
    println("Contraction result: ", result)
    println(result)
catch e
    println("Error during contraction: ", e)
    println("Error type: ", typeof(e))
    for (exc, bt) in Base.catch_stack()
        showerror(stdout, exc, bt)
        println()
    end
end