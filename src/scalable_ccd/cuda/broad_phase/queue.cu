#include "queue.cuh"

namespace scalable_ccd::cuda {

__device__ int2 Queue::pop()
{
    if (!is_empty()) {
        const int current = atomicInc(&start, QUEUE_SIZE - 1);
        return storage[current];
    }
    // Return the sentinel value if the queue is empty
    return QUEUE_ERROR();
}

__device__ bool Queue::push(const int2 pair)
{
    if (!is_full()) {
        const int current = atomicInc(&end, QUEUE_SIZE - 1);
        storage[current] = pair;
        return true;
    }
    // Return false if the queue is full
    return false;
}

} // namespace scalable_ccd::cuda