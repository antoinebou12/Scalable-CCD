#pragma once

namespace scalable_ccd::cuda {

static constexpr int QUEUE_SIZE = 64;

// Use a sentinel value to indicate an error
// inline __device__ int2 QUEUE_ERROR() { return make_int2(-1, -1); }

/// @brief A simple lock-free ring queue for use in CUDA kernels
struct Queue {
    /// @brief Pop the top element from the queue
    /// @return The top element
    __device__ inline int2 pop()
    {
        assert(!is_empty() && "Queue is empty");
        return storage[atomicInc(&start, QUEUE_SIZE - 1)];
    }

    /// @brief Push an element to the queue
    /// @param pair The element to push
    /// @return True if the element was pushed, false if the queue is full
    __device__ inline void push(const int2 pair)
    {
        assert(!is_full() && "Queue is full");
        storage[atomicInc(&end, QUEUE_SIZE - 1)] = pair;
    }

    /// @brief Get the current size of the queue
    __device__ int size() const { return QUEUE_SIZE; }

    /// @brief Check if the queue is full
    __device__ bool is_full() const { return (end + 1) % QUEUE_SIZE == start; }

    /// @brief Check if the queue is empty
    __device__ bool is_empty() const { return end == start; }

    /// @brief Statically allocated array to hold the queue
    int2 storage[QUEUE_SIZE];

    /// @brief Index of the start of the queue
    unsigned start; // = 0;

    /// @brief Index of the end of the queue
    unsigned end; // = 0;

    /// @brief
    int nbr_per_loop;
};

} // namespace scalable_ccd::cuda