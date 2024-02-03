#include <scalable_ccd/cuda/broad_phase/queue.cuh>
#include <iostream>

namespace scalable_ccd::cuda {

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

} // namespace scalable_ccd::cuda