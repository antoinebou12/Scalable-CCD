#pragma once

// #include <cuda/semaphore>

namespace scalable_ccd::cuda::stq {

static constexpr int HEAP_SIZE = 64;

// Use a sentinel value to indicate an error
inline __device__ int2 QUEUE_ERROR() { return make_int2(-1, -1); }

// Prototype of a utility function to swap two integers
// __device__ void swap(Cell&x, Cell &y);

__device__ __host__ class Queue {
public:
    __device__ __host__ Queue() = default;

    __device__ int2 pop();

    // Inserts a new key 'k'
    __device__ bool push(const int2 pair);

    __device__ int size() const { return heap_size; }

    __device__ bool is_full() const { return (end + 1) % HEAP_SIZE == start; }

    __device__ bool is_empty() const { return end == start; }

public:
    // pointer to array of elements in heap
    int2 harr[HEAP_SIZE];

    // int current = 0;

    // ::cuda::binary_semaphore<::cuda::thread_scope_block> lock[HEAP_SIZE];

    // int capacity;  // maximum possible size of min heap

    // Current number of elements in min heap
    int heap_size = HEAP_SIZE;

    // Cell root; // temporary variable used for extractMin()

    // unsigned old_start;

    unsigned start;

    unsigned end;

    int nbr_per_loop;

    // int old_nbr_per_loop;

    // unsigned pop_cnt;

    // unsigned push_cnt;
};

} // namespace scalable_ccd::cuda::stq