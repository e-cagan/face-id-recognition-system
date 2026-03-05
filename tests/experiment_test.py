"""
Module to run experiment test on a desired 
"""

import sys
import os
# Also "see" files on the main dir
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import time
import numpy as np
import matplotlib.pyplot as plt
import config
from modules.face_recognition import FaceRecognizer

def run_experiment():
    # 1. Initialize the recognizer using the same metric from your config
    # We use cosine distance as it's your default distance metric.
    recognizer = FaceRecognizer(distance_metric=config.DISTANCE_METRIC, threshold=config.RECOGNITION_THRESHOLD)
    
    # 2. Define the N values (Number of users in the database) to test.
    # We increase N gradually to observe the linear growth O(N).
    n_values = [10, 100, 500, 1000, 5000]
    
    # Number of trials for each N to get a reliable average
    trials = 10
    
    # Embedding dimension (Facenet uses 128 or 512, let's simulate 128-dimensional vectors)
    embedding_dim = 128 
    
    average_times = []

    print("=== O(N) Time Complexity Experiment Started ===\n")

    for n in n_values:
        print(f"--- Testing for N = {n} Users ---")
        
        # 3. Generate N dummy user embeddings to simulate database records
        # This isolates the pure algorithmic speed from SQLite disk reading speed
        stored_embeddings = [(f"user_{i}", np.random.rand(embedding_dim)) for i in range(n)]
        
        trial_times = []
        
        for trial in range(1, trials + 1):
            # 4. Generate a random dummy embedding representing a new camera frame
            query_embedding = np.random.rand(embedding_dim)
            
            # 5. Start the high-precision timer
            start_time = time.perf_counter()
            
            # Run the matching algorithm
            recognizer.find_match(query_embedding, stored_embeddings)
            
            # Stop the timer
            end_time = time.perf_counter()
            
            # 6. Calculate elapsed time in milliseconds (ms)
            elapsed_ms = (end_time - start_time) * 1000
            trial_times.append(elapsed_ms)
            
            # Print individual trial results for your Measurement Charts
            print(f"Trial {trial}: {elapsed_ms:.4f} ms")
            
        # Calculate the average time for the current N
        avg_time = sum(trial_times) / trials
        average_times.append(avg_time)
        print(f">>> Average Execution Time for N={n}: {avg_time:.4f} ms\n")
        
    # 7. Plotting the results to visualize O(N) time complexity
    print("Generating result graph...")
    plt.figure(figsize=(8, 5))
    plt.plot(n_values, average_times, marker='o', linestyle='-', color='blue', linewidth=2)
    plt.title('Time Complexity of Face Matching Algorithm - O(N)')
    plt.xlabel('Number of Stored Users (N)')
    plt.ylabel('Average Execution Time (ms)')
    plt.grid(True, linestyle='--', alpha=0.7)
    
    # Save the plot as an image to include in the design report
    plt.savefig('graphs/experimental_result_graph.png')
    print("Experiment completed successfully! Graph saved as 'graphs/experimental_result_graph.png'.")

if __name__ == "__main__":
    run_experiment()