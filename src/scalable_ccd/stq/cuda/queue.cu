#include <scalable_ccd/stq/cuda/queue.cuh>
#include <iostream>

namespace scalable_ccd::stq::gpu {

__device__ int2 QUEUE_ERROR() { return make_int2(-1, -1); }

__device__ __host__ Queue::Queue() { heap_size = HEAP_SIZE; }

__device__ int2 Queue::pop()
{
    if (!is_empty()) {
        int current = atomicInc(&start, HEAP_SIZE - 1);
        return harr[current];
    }
    // Return the sentinel value if the queue is empty
    return QUEUE_ERROR();
}

__device__ bool Queue::push(const int2 pair)
{
    if (!is_full()) {
        int current = atomicInc(&end, HEAP_SIZE - 1);
        harr[current] = pair;
        return true;
    }
    // Return false if the queue is full
    return false;
}

__device__ int Queue::size() { return heap_size; }

__device__ bool Queue::is_full() { return (end + 1) % HEAP_SIZE == start; }

__device__ bool Queue::is_empty() { return end == start; }

} // namespace scalable_ccd::stq::gpu