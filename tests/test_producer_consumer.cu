// #include <vector>
// #include <iostream>
// #include <bitset>
// #include <string>
// #include <numeric>
// #include <string>
// #include <functional>
// #include <cuda/pipeline>
// #include <cuda/semaphore>

// // need this to get tiled_partition > 32 threads
// #define _CG_ABI_EXPERIMENTAL // enable experimental API

// #include <cooperative_groups.h>

// #include <curand.h>
// #include <curand_kernel.h>

// #include <stq/gpu/queue.cuh>
// #include <stq/gpu/aabb.cuh>
// #include <stq/gpu/timer.cuh>
// #include <stq/gpu/utils.cuh>

// using namespace cooperative_groups;
// namespace cg = cooperative_groups;

#include <thrust/host_vector.h>
#include <thrust/device_vector.h>

__global__ void run(long* in, long* out, int N)
{
    __shared__ cuda::pipeline_shared_state<cuda::thread_scope_block, 2> pss;
    __shared__ Queue queue;
    queue.capacity = QUEUE_SIZE;
    queue.heap_size = QUEUE_SIZE;
    for (int i = threadIdx.x; i < QUEUE_SIZE; i += blockDim.x) {
        queue.lock[i].release();
        queue.storage[i].x = -1.0; // release to add
                                   // printf("Lock %i released\n", i);
    }
    __syncthreads();

    // Size must be a power of 2 and less than or equal to 32.
    cg::thread_block g = cg::this_thread_block();
    // thread_group tile32 = tiled_partition(g, 32);
    // int lane = tile32.thread_rank();
    // int tileIdx = g.thread_rank() / 32;

    // reserve shared memory for thread_block_tile usage.
    __shared__ cg::experimental::block_tile_memory<2, 1024> shared;
    cg::thread_block thb = cg::experimental::this_thread_block(shared);

    auto tilehalf = cg::experimental::tiled_partition<512>(thb);

    int lane = tilehalf.thread_rank();
    int lanerel = lane + blockIdx.x * blockDim.x;
    __syncthreads();

    int tid = threadIdx.x + blockDim.x * blockIdx.x;
    // if (tid >= N) return;
    // if in the first group + tid >= N
    if (tid >= N)
        return;

    // if (lane == 0)
    //     printf("group_rank: %i\n", tilehalf.meta_group_rank());

    if (tilehalf.meta_group_rank() == 0) {
        int curr1, curr2;
        int2 val1 = make_int2(in[tid], 0);
        int2 val2 = make_int2(0, in[tid]);
        curr1 = queue.push(lanerel, val1);
        curr2 = queue.push(lanerel, val2);
    } else {
        int2 res1 = queue.pop(lanerel % QUEUE_SIZE);
        // int2 res2 = queue.pop(lanerel % QUEUE_SIZE);
        out[tid] = res1.x;
    }

    //     int rand1, rand2;
    //     /* CUDA's random number library uses curandState_t to keep track of
    //     the seed value
    //      we will store a random state for every thread  */
    //     curandState_t state;

    //   /* we have to initialize the state */
    //     curand_init(0, /* the seed controls the sequence of random values
    //     that are produced */
    //                 0, /* the sequence number is only important with multiple
    //                 cores */ tid, /* the offset is how much extra we advance
    //                 in the sequence for each call, can be 0 */ &state);

    //   /* curand works like rand - except that it takes a state as a parameter
    //   */
    //     rand1 = int(curand(&state) % QUEUE_SIZE);
    //     rand2 = int(curand(&state) % QUEUE_SIZE);
    // if (tid == 0)
    //     printf("rand1: %i, rand2: %i\n", rand1, rand2);

    return;
}

TEST_CASE("Produce-Consumer", "[cuda][producer-consumer]")
{
    const int N = GENERATE(100, 10'000, 10'000'000);

    std::vector<long> nums(N);
    std::iota(nums.begin(), nums.end(), 0); // range 0 to N-1

    thrust::device_vector<long> d_in = nums;
    thrust::device_vector<long> d_out(N, 0);

    int block = 1024;
    int grid = (N / block + 1);
    printf("grid size: %i\n", grid);
    printf(
        "sizeof(semaphore):  %i\n",
        sizeof(cuda::binary_semaphore<cuda::thread_scope_block>));
    printf("sizeof(int2):  %i\n", sizeof(int2));

    record("run", grid, block, 8, run, d_in, d_out, N);
    cudaDeviceSynchronize();

    vector<long> out;
    out.resize(N);
    cudaMemcpy(out.data(), d_out, sizeof(long) * N, cudaMemcpyDeviceToHost);

    // int s = accumulate(out.begin(), out.end(), 0);
    int s = 0;
    for (int i = 0; i < N; i += 1) {
        // printf("%lld:%lld ", nums[i], out[i]);
        s += out[i];
    }
    printf("\n");
    printf("sum: %i\n", s);
}