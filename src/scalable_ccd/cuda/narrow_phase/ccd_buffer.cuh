#pragma once

#include <scalable_ccd/cuda/narrow_phase/interval.cuh>

namespace scalable_ccd::cuda {

struct CCDBuffer {
    /// @brief Check if the buffer is full
    __device__ bool is_full() const
    {
        return (m_tail + 1) % m_capacity == m_head;
    }

    /// @brief Check if the buffer is empty
    __device__ bool is_empty() const { return m_tail == m_head; }

    // __device__ CCDDomain pop()
    // {
    //     if (!is_empty()) {
    //         return m_data[atomicInc(&m_head, m_capacity - 1)];
    //     }
    //     assert(false);
    // }

    __device__ CCDDomain& push(const CCDDomain& val)
    {
        if (is_full()) {
            atomicCAS(&m_overflow_flag, 0, 1);
            return;
        }
        const int i = atomicInc(&m_tail, m_capacity - 1);
        m_data[i] = val;
        return m_data[i];
    }

    __device__ CCDDomain& operator[](const int i)
    {
        return m_data[(i + m_head) % m_capacity];
    }

    __device__ void shift_queue_start()
    {
        // Update the head to the new starting position (assuming all starting
        // elements were consumed)
        m_head = (m_head + m_starting_size) % m_capacity;
        // Calculate the new starting size
        if (m_head <= m_tail) {
            m_starting_size = m_tail - m_head;
        } else {
            m_starting_size = m_capacity - m_head + m_tail;
        }
    }

    __device__ unsigned starting_size() const { return m_starting_size; }
    __device__ unsigned capacity() const { return m_capacity; }
    __device__ unsigned head() const { return m_head; }
    __device__ unsigned tail() const { return m_tail; }
    __device__ int overflow_flag() const { return m_overflow_flag; }

    // These should be private, but we need to access them from the kernel.
    // private:
    CCDDomain* m_data;
    unsigned m_starting_size = 0;
    unsigned m_capacity = 0;
    unsigned m_head = 0;
    unsigned m_tail = 0;
    int m_overflow_flag = 0;
};

__global__ void initialize_buffer(CCDBuffer* buffer)
{
    int tx = threadIdx.x + blockIdx.x * blockDim.x;
    if (tx >= buffer->starting_size())
        return;
    assert(buffer->head() == 0 && buffer->tail() >= tx);
    buffer->m_data[tx].init(tx);
}

__global__ void shift_queue_start(CCDBuffer* buffer)
{
    assert(threadIdx.x == 0 && blockDim.x == 1 && gridDim.x == 1);
    buffer->shift_queue_start();
}

} // namespace scalable_ccd::cuda