
#define __CL_ENABLE_EXCEPTIONS
#include "CL/cl.hpp"

#include <iostream>
#include <fstream>
#include <vector>
#include <string>
#include <iterator>
#include <iomanip>
#include <random>
#include <chrono>
#include <algorithm>

cl::Platform selectPlatform()
{
    std::vector<cl::Platform> platforms;
    cl::Platform::get(&platforms);
    int i = 0;
    for (cl::Platform const& platform: platforms) {
        std::string platform_name;
        std::string platform_version;
        std::string platform_extensions;

        platform.getInfo(CL_PLATFORM_NAME, &platform_name);
        platform.getInfo(CL_PLATFORM_VERSION, &platform_version);
        platform.getInfo(CL_PLATFORM_EXTENSIONS, &platform_extensions);

        std::cout << "#" << i << std::endl;
        std::cout << "name: " << platform_name << std::endl;
        std::cout << "version: " << platform_version << std::endl;
        std::cout << "extensions: " << platform_extensions << std::endl;
        std::cout << "============" << std::endl;
        ++i;
    }
    std::cout << "Select platform: ";
    std::cin >> i;
    return platforms[i];
}

cl::Device selectDevice(std::vector<cl::Device> const& devices)
{
    int i = 0;
    for (cl::Device const& device: devices) {
        std::string device_name;
        std::string device_version;
        std::string driver_version;
        std::string device_profile;
        std::string device_built_in_kernels;
        std::string device_extensions;

        device.getInfo(CL_DEVICE_NAME, &device_name);
        device.getInfo(CL_DEVICE_VERSION, &device_version);
        device.getInfo(CL_DRIVER_VERSION, &driver_version);
        device.getInfo(CL_DEVICE_PROFILE, &device_profile);
        device.getInfo(CL_DEVICE_BUILT_IN_KERNELS, &device_built_in_kernels);
        device.getInfo(CL_DEVICE_EXTENSIONS, &device_extensions);

        std::cout << "#" << i << std::endl;
        std::cout << "name: " << device_name << std::endl;
        std::cout << "version: " << device_version << std::endl;
        std::cout << "driver version: " << driver_version << std::endl;
        std::cout << "profile: " << device_profile << std::endl;
        std::cout << "built in kernels: " << device_built_in_kernels << std::endl;
        std::cout << "device extensions: " << device_extensions << std::endl;
        ++i;
    }
    std::cout << "Select device: ";
    std::cin >> i;
    return devices[i];
}

void generate_random_data(int N, int M, std::string filename) {
    std::uniform_real_distribution<float> distribution(-10.0, 10.0);
    unsigned seed = std::chrono::system_clock::now().time_since_epoch().count();
    std::default_random_engine generator(seed);

    std::ofstream output_file(filename);
    output_file << N << " " << M << std::endl;
    for (int i = 0; i < N; ++i) {
        for (int j = 0; j < N; ++j) {
            output_file << distribution(generator) << " ";
        }
        output_file << std::endl;
    }

    for (int i = 0; i < M; ++i) {
        for (int j = 0; j < M; ++j) {
            output_file << distribution(generator) << " ";
        }
        output_file << std::endl;
    }
}

std::vector<float> cpu_conv(std::vector<float> A, std::vector<float> B, int N, int M) {
    std::vector<float> C(N * N);
    int const HM = (M - 1) / 2;
    for (int i = 0; i < N; i++) {
        for (int j = 0; j < N; j++) {
            C[i * N + j] = 0;
            for (int k = -HM; k <= HM; k++) {
                for (int l = -HM; l <= HM; l++) {
                    int const a_i = i + k;
                    int const a_j = j + l;
                    if (a_i < 0 || a_j < 0 || a_i >= N || a_j >= N)
                        continue;
                    int const b_i = HM + k;
                    int const b_j = HM + l;
                    C[i * N + j] += A[a_i * N + a_j] * B[b_i * M + b_j];
                }
            }
        }
    }
    return C;
}

int main()
{
    try {
        std::vector<cl::Device> devices;

        // select platform
        cl::Platform platform = selectPlatform();

        // select device
        platform.getDevices(CL_DEVICE_TYPE_ALL, &devices);
        cl::Device device = selectDevice(devices);

        // create context
        cl::Context context(devices);

        // create command queue
        cl::CommandQueue queue(context, device, CL_QUEUE_PROFILING_ENABLE);

        // load opencl source
        std::ifstream cl_file("matrix_conv.cl");
        std::string cl_string{std::istreambuf_iterator<char>(cl_file),
                              std::istreambuf_iterator<char>()};

        cl::Program::Sources source(1,
                                    std::make_pair(cl_string.c_str(),
                                                   cl_string.length() + 1));

        // create programm
        cl::Program program(context, source);

        // compile opencl source
        try
        {
            program.build(devices);
        }
        catch (cl::Error const & e)
        {
            std::string log_str = program.getBuildInfo<CL_PROGRAM_BUILD_LOG>(device);
            std::cout << std::endl << e.what() << " : " << e.err() << std::endl;
            std::cout << log_str;
            return 0;
        }

        // generate_random_data(1024, 64, "input.txt");

        // load data from file
        std::ifstream input_file("input.txt");
        int N;
        int M;
        input_file >> N >> M;

        std::vector<float> A(N * N); // signal
        std::vector<float> B(M * M); // template
        std::vector<float> C(N * N); // result

        for (int i = 0; i < N; ++i) {
            for (int j = 0; j < N; ++j) {
                input_file >> A[i * N + j];
            }
        }

        for (int i = 0; i < M; ++i) {
            for (int j = 0; j < M; ++j) {
                input_file >> B[i * M + j];
            }
        }

        // set block size
        size_t const block_size = 16;

        // allocate device buffer to hold message
        cl::Buffer dev_a(context, CL_MEM_READ_ONLY, sizeof(float) * N * N);
        cl::Buffer dev_b(context, CL_MEM_READ_ONLY, sizeof(float) * M * M);
        cl::Buffer dev_c(context, CL_MEM_WRITE_ONLY, sizeof(float) * N * N);

        // copy from cpu to gpu
        queue.enqueueWriteBuffer(dev_a, CL_TRUE, 0, sizeof(float) * N * N, A.data());
        queue.enqueueWriteBuffer(dev_b, CL_TRUE, 0, sizeof(float) * M * M, B.data());

        // load named kernel from opencl source
        auto matrix_conv = cl::make_kernel< cl::Buffer&
                                         , cl::Buffer&
                                         , cl::Buffer&
                                         , int
                                         , int >(program, "matrix_conv");
        size_t const global_size = (N / block_size + ((N % block_size)?1:0)) * block_size;
        auto enqueue_args = cl::EnqueueArgs(queue, cl::NDRange(global_size, global_size), cl::NDRange(block_size, block_size));
        matrix_conv(enqueue_args, dev_a, dev_b, dev_c, N, M);
        queue.enqueueReadBuffer(dev_c, CL_TRUE, 0, sizeof(float) * N * N, C.data());

        // CPU conv calculation check
//        auto C_cpu = cpu_conv(A, B, N, M);
//        if (std::equal(C.begin(), C.end(), C_cpu.begin(), [](float x, float y){return fabs(x - y) < 1e-3;})) {
//            std::cout << "Ok" << std::endl;
//        } else {
//            std::cout << "comparation failed" << std::endl;
//        }

        // write result
        std::ofstream output_file("output.txt");
        for (int i = 0; i < N; ++i) {
            for (int j = 0; j < N; ++j) {
                output_file << C[i * N + j] << " ";
            }
            output_file << std::endl;
        }
    }
    catch (cl::Error const & e) {
        std::cout << "Error: " << e.what() << " #" << e.err() << std::endl;
    }

    return 0;
}
