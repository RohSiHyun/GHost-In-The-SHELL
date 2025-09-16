#include <cassert>
#include <cerrno>
#include <cstdint>
#include <cub/cub.cuh>
#include <cuda_runtime_api.h>
#include <jemalloc/jemalloc.h>
#include <fcntl.h>
#include <fstream>
#include <iostream>
#include <numeric>
#include <string>
#include <sys/mman.h>
#include <unistd.h>
#include <vector>

__constant__ size_t month_day_boundary[13] = {0, 31, 60, 91, 121, 152, 182, 213, 244, 274, 305, 335, 366};

__inline__ __device__ int get_month_from_day_of_year(int day_of_year) {
  const int total_months = 2;

  int month = total_months / 2;
  int upper_month = total_months;
  int lower_month = 0;

  // binary search in array
  while (lower_month <= upper_month) {
    if (day_of_year >= month_day_boundary[month])
      lower_month = month + 1;
    else if (day_of_year < month_day_boundary[month])
      upper_month = month - 1;
    month = int((upper_month + lower_month) / 2);
  }
  return month; // this is 0 based
}
__global__ void construct_yearly_histogram(float *input_data, int start_year, int end_year,
                                           size_t input_grid_height, size_t input_grid_width,
                                           size_t aligned_month_file_map_offset, float *histogram_data) {
  // end year is included
  const size_t hours_per_day = 24; // assumed 24 hr data
  const size_t days_per_leap_year = 60;
  const size_t months_per_year = 2;

  // sum will accumulate in register for the full grid
  size_t grid_pitch = input_grid_height * input_grid_width;
  size_t day_grid_pitch = hours_per_day * grid_pitch;

  // output mapping
  // total 366 * 24 * 721 * 1440 active threads
  size_t linear_day_hr_loc_idx = (size_t)blockDim.x * blockIdx.x + threadIdx.x;

  size_t max_active_threads = (days_per_leap_year * (int64_t)day_grid_pitch);

  size_t day_of_year = linear_day_hr_loc_idx / day_grid_pitch; // this is 0-based
  size_t hour_of_day = (linear_day_hr_loc_idx - (day_of_year * day_grid_pitch)) / grid_pitch;
  size_t grid_linearized_idx =
      linear_day_hr_loc_idx - (day_of_year * day_grid_pitch) - (hour_of_day * grid_pitch);

  size_t grid_y = grid_linearized_idx / input_grid_width;
  size_t grid_x = grid_linearized_idx % input_grid_width;

  // month is required as each file is mapped at a separate offset - for page boundary alignment
  size_t month = (size_t)get_month_from_day_of_year((int)day_of_year);

  if (linear_day_hr_loc_idx < max_active_threads) {
    float accum_sum = 0.0f;

    for (int i = 0; i <= (end_year - start_year); i++) {
      int year = i + start_year;

      size_t access_index = (((size_t)i * months_per_year + month) * aligned_month_file_map_offset) +
                            ((day_of_year - month_day_boundary[month]) * day_grid_pitch) +
                            (hour_of_day * grid_pitch) + grid_y * input_grid_width + grid_x;
      // leap year adjustment for feb
      if (day_of_year == 59) {
        if ((year % 4) == 0) {
          // leap year - read away
          accum_sum += input_data[access_index];
        }
      } else {
        accum_sum += input_data[access_index];
      }
    }
    // write out
    histogram_data[linear_day_hr_loc_idx] = accum_sum;
  }
}


__global__ void reduce_kernel_float(
    const float* __restrict__ d_in,
    float*       __restrict__ d_partial,
    int                        num_items)
{
    extern __shared__ float sdata[];
    unsigned int tid   = threadIdx.x;
    unsigned int start = blockIdx.x * (blockDim.x * 2);
    unsigned int idx   = start + tid;

    // 1) Each thread loads up to two elements into a register
    float sum = 0.0f;
    if (idx < (unsigned)num_items) {
        sum = d_in[idx];
    }
    if ((idx + blockDim.x) < (unsigned)num_items) {
        sum += d_in[idx + blockDim.x];
    }

    // 2) Write into shared memory
    sdata[tid] = sum;
    __syncthreads();

    // 3) Tree‐reduce in shared memory
    for (unsigned int stride = blockDim.x / 2; stride > 0; stride >>= 1) {
        if (tid < stride) {
            sdata[tid] += sdata[tid + stride];
        }
        __syncthreads();
    }

    // 4) Thread 0 writes this block’s partial sum
    if (tid == 0) {
        d_partial[blockIdx.x] = sdata[0];
    }
}



// Second‐pass kernel: one block reduces all block‐partial sums into a single float.
__global__ void final_reduce_kernel_float(
    const float* __restrict__ d_partial,
    float*       __restrict__ d_out,
    int                        n_partial)
{
    extern __shared__ float sdata2[];
    unsigned int tid = threadIdx.x;

    // 1) Load each partial sum (or zero if tid ≥ n_partial)
    float sum = 0.0f;
    if (tid < (unsigned)n_partial) {
        sum = d_partial[tid];
    }
    sdata2[tid] = sum;
    __syncthreads();

    // 2) Tree‐reduce in shared memory
    for (unsigned int stride = blockDim.x / 2; stride > 0; stride >>= 1) {
        if (tid < stride) {
            sdata2[tid] += sdata2[tid + stride];
        }
        __syncthreads();
    }

    // 3) Thread 0 writes the final result
    if (tid == 0) {
        *d_out = sdata2[0];
    }
}

__global__ void reduceSum(float* __restrict__ in,
                          float* __restrict__ out,
                          size_t n) {
    extern __shared__ float sdata[];

    unsigned int tid = threadIdx.x;
    unsigned int i   = blockIdx.x * blockDim.x + tid;

    // Load input into shared memory
    sdata[tid] = (i < n) ? in[i] : 0.0f;
    __syncthreads();

    // Parallel reduction in shared memory
    for (unsigned int s = blockDim.x / 2; s > 0; s >>= 1) {
        if (tid < s) {
            sdata[tid] += sdata[tid + s];
        }
        __syncthreads();
    }

    // Write block result to global memory
    if (tid == 0) {
        atomicAdd(out, sdata[0]);
    }
}



#define CUDA_CHECK(err)                                                                                      \
  if (err != cudaSuccess) {                                                                                  \
    std::cout << "CUDA error at " << __LINE__ << " " << cudaGetErrorString(err) << std::endl;                \
    return -1;                                                                                               \
  }

int main(int argc, char *argv[]) {


  struct timespec start_time, end_time;
  clock_gettime(CLOCK_MONOTONIC, &start_time);
  // hard coded constants for ERA5
  const int hours_per_day = 24; // assumed 24 hr data
  const int days_per_leap_year = 60;
  const int max_days_per_month = 31;
  const int months_per_year = 2;
  const int input_grid_height = 721;
  const int input_grid_width = 1440;
  int start_year = std::atoi(argv[1]);
  int end_year = std::atoi(argv[2]);
  std::string file_path = std::string(argv[3]);

  const int num_years = end_year - start_year + 1;

  size_t max_file_size =
      sizeof(float) * max_days_per_month * hours_per_day * input_grid_height * input_grid_width;

  size_t TWO_MB = 2 * 1024 * 1024;
  size_t max_aligned_file_pages = (max_file_size + TWO_MB - 1) / TWO_MB;
  size_t max_aligned_file_size = max_aligned_file_pages * TWO_MB;


  std::vector<size_t> file_sizes;
  std::vector<int> open_fds;

  // 2 MB aligned VA range to allocate
  size_t va_alloc_size = sizeof(float) * num_years * months_per_year * max_aligned_file_size;

  void *va_alloc = mmap(nullptr, va_alloc_size, PROT_NONE, MAP_PRIVATE | MAP_ANONYMOUS, -1, 0);

  void *running_address = va_alloc;

  std::string file_path_base = file_path;

  for (int y = start_year; y <= end_year; y++) {
    for (int k = 1; k <= months_per_year; k++) {
      char filestr_buf[32];
      sprintf(filestr_buf, "%d%02d.bin", y, k);
      std::string filename = file_path_base + "e5.accumulated_tp_1h." + std::string(filestr_buf);

      std::ifstream fstreamInput(filename, std::ios::binary);
      fstreamInput.seekg(0, std::ios::end);
      size_t fileByteSize = fstreamInput.tellg();
      fstreamInput.close();

      int fd = open(filename.c_str(), O_RDONLY, 0);
      if (fd == -1) {
        return -1;
      }

      char *mapped_addr = NULL;
      // probably need 2 MB pages for perf
      mapped_addr =
          (char *)mmap((void *)running_address, fileByteSize, PROT_READ, MAP_PRIVATE | MAP_FIXED, fd, 0);

      if (mapped_addr == MAP_FAILED) {
        close(fd);
        return -2;
      }

      assert(mapped_addr == (char *)running_address);
      running_address = (void *)((char *)running_address + max_aligned_file_size);

      file_sizes.push_back(fileByteSize);
      open_fds.push_back(fd);
    }
  }

  // launch kernel and feed in pointer and values
  size_t hist_bins = (size_t)days_per_leap_year * hours_per_day * input_grid_height * input_grid_width;
  size_t histogram_alloc_size = hist_bins * sizeof(float);
  float *histogram_data = NULL;
  CUDA_CHECK(cudaMalloc((void **)&histogram_data, histogram_alloc_size));
  CUDA_CHECK(cudaMemset(histogram_data, 0, histogram_alloc_size));


  dim3 block(1024, 1, 1);
  dim3 grid(1, 1, 1);

  grid.x = (hist_bins + block.x - 1) / block.x;

  construct_yearly_histogram<<<grid, block, 0, NULL>>>(
      reinterpret_cast<float *>(va_alloc), start_year, end_year, (size_t)input_grid_height,
      (size_t)input_grid_width, max_aligned_file_size / sizeof(float), histogram_data);

  CUDA_CHECK(cudaGetLastError()); // for catching errors from launch

  float time_ms = 0.0f;

  CUDA_CHECK(cudaDeviceSynchronize()); // to start reading output histogram on host

  size_t month_day_boundary[13] = {0, 31, 60, 91, 121, 152, 182, 213, 244, 274, 305, 335, 366};
  std::vector<std::vector<float>> hourly_sum_per_day;

  for (int m = 0; m < months_per_year; m++) {
    size_t start_index = month_day_boundary[m] * hours_per_day * input_grid_height * input_grid_width;
    float local_sum = 0.0f;
    for (int d = 0; d < (month_day_boundary[m + 1] - month_day_boundary[m]); d++) {
      std::vector<float> hour_sum(24);
      for (int h = 0; h < hours_per_day; h++) {
        float month_sum[16] = {0.0f};
        void *d_temp_storage = NULL;
        size_t temp_storage_bytes = 0;
        size_t num_items = (input_grid_height / 2) * input_grid_width;
        size_t strided_hourly_idx = start_index + (d * hours_per_day * input_grid_height * input_grid_width) +
                                    (h * input_grid_height * input_grid_width);
        float *array_start = &(histogram_data[strided_hourly_idx]);
        cub::DeviceReduce::Sum(d_temp_storage, temp_storage_bytes, array_start, month_sum, num_items);

        d_temp_storage = mallocx(temp_storage_bytes, 0); // use HMM B-)
        reduceSum<<<16, 16>>>((float*)d_temp_storage, array_start, num_items);
        CUDA_CHECK(cudaDeviceSynchronize());

        dallocx(d_temp_storage, 0);
        hour_sum[h] = month_sum[0];
      }
      hourly_sum_per_day.push_back(hour_sum);
      local_sum += std::accumulate(hour_sum.begin(), hour_sum.end(), 0.0f);
    }
  }

  CUDA_CHECK(cudaFree(histogram_data));

  void *unmap_address = va_alloc;
  for (int k = 1; k < argc; k++) {
    int unmap_return = munmap(unmap_address, file_sizes[k - 1]); // unmap all address
    close(open_fds[k - 1]);
    unmap_address = (void *)((char *)unmap_address + max_aligned_file_size);
  }

  int unmap_return = munmap(va_alloc, va_alloc_size); // unmap all address

  clock_gettime(CLOCK_MONOTONIC, &end_time);
  double total_time = (end_time.tv_sec - start_time.tv_sec) +
                      (end_time.tv_nsec - start_time.tv_nsec) / 1e9;
  std::cout << "Total time: " << total_time << " seconds" << std::endl;

  return 0;
}