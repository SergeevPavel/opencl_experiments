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
#include <stdexcept>

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
    //    std::cin >> i;
    i = 1;
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
    //    std::cin >> i;
    i = 0;
    return devices[i];
}

std::vector<float> cpu_inclusive_scan(std::vector<float> const& input)
{
    std::vector<float> output(input);
    for (int i = 1; i < output.size(); i++) {
        output[i] += output[i - 1];
    }
    return output;
}

void cpu_check(std::vector<float> input, std::vector<float> output)
{
    std::vector<float> cpu_output = cpu_inclusive_scan(input);
    if (std::equal(output.begin(), output.end(), cpu_output.begin(), std::equal_to<double>())) {
        std::cout << "Ok" << std::endl;
    } else {
        std::cout << "comparation failed" << std::endl;
    }
}

size_t const block_size = 8;
cl::Program program;
cl::Context context;
cl::CommandQueue queue;

cl::Buffer small_array_scan(cl::Buffer input, size_t input_size)
{
    if (input_size > block_size) throw std::invalid_argument("input too big");

    auto kernel = cl::make_kernel< cl::Buffer&
            , cl::Buffer&
            , cl::LocalSpaceArg
            , cl::LocalSpaceArg >(program, "small_array_scan");

    cl::Buffer output(context, CL_MEM_READ_WRITE, sizeof(float) * input_size);
    auto enqueue_args = cl::EnqueueArgs(queue, cl::NDRange(input_size), cl::NDRange(input_size));
    cl::Event event = kernel(enqueue_args, input, output,
                             cl::Local(sizeof(float) * input_size), cl::Local(sizeof(float) * input_size));
    event.wait();
    return output;
}

std::pair<cl::Buffer, cl::Buffer> subblock_scan(cl::Buffer input, size_t input_size)
{
    auto kernel = cl::make_kernel< cl::Buffer&
            , cl::Buffer&
            , cl::Buffer&
            , cl::LocalSpaceArg
            , cl::LocalSpaceArg >(program, "subblock_scan");

    size_t blocks_count = input_size / block_size;
    cl::Buffer output(context, CL_MEM_READ_WRITE, sizeof(float) * input_size);
    cl::Buffer last_elements(context, CL_MEM_READ_WRITE, sizeof(float) * blocks_count);

    cl::EnqueueArgs enqueue_args = cl::EnqueueArgs(queue, cl::NDRange(input_size), cl::NDRange(block_size));
    cl::Event event = kernel(enqueue_args, input, output, last_elements,
                             cl::Local(sizeof(float) * input_size), cl::Local(sizeof(float) * input_size));
    event.wait();
    return std::make_pair(output, last_elements);
}

cl::Buffer merge(cl::Buffer input, cl::Buffer additions, size_t input_size)
{
    auto kernel = cl::make_kernel< cl::Buffer&
            , cl::Buffer&
            , cl::Buffer& >(program, "merge");

    cl::Buffer output(context, CL_MEM_READ_WRITE, sizeof(float) * input_size);

    cl::EnqueueArgs enqueue_args = cl::EnqueueArgs(queue, cl::NDRange(input_size), cl::NDRange(block_size));
    cl::Event event = kernel(enqueue_args, input, output, additions);
    event.wait();
    return output;
}

cl::Buffer inclusive_scan(cl::Buffer input, size_t input_size)
{
    if (block_size >= input_size) {
        return small_array_scan(input, input_size);
    } else {
        cl::Buffer subblocks;
        cl::Buffer last_elements;
        std::tie(subblocks, last_elements) = subblock_scan(input, input_size);
        cl::Buffer last_elements_scaned = inclusive_scan(last_elements, input_size / block_size);
        return merge(subblocks, last_elements_scaned, input_size);
    }
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
        context = cl::Context(devices);

        // create command queue
        queue = cl::CommandQueue(context, device, CL_QUEUE_PROFILING_ENABLE);

        // load opencl source
        std::ifstream cl_file("inclusive_scan.cl");

        std::string cl_string{std::istreambuf_iterator<char>(cl_file),
                    std::istreambuf_iterator<char>()};

        cl::Program::Sources source(1,
                                    std::make_pair(cl_string.c_str(),
                                                   cl_string.length() + 1));

        // create programm
        program = cl::Program(context, source);

        // compile opencl source
        try {
            program.build(devices);

            size_t const input_size = 8 * 8 * 8;

            std::vector<float> input(input_size);
            std::vector<float> output(input_size, 0);
            for (size_t i = 0; i < input_size; ++i)
            {
                input[i] = i % 10;
            }

            cl::Buffer dev_input (context, CL_MEM_READ_ONLY, sizeof(float) * input_size);
            queue.enqueueWriteBuffer(dev_input, CL_TRUE, 0, sizeof(float) * input_size, &input[0]);

            cl::Buffer dev_output = inclusive_scan(dev_input, input_size);

            queue.enqueueReadBuffer(dev_output, CL_TRUE, 0, sizeof(float) * input_size, &output[0]);
            queue.finish();

            cpu_check(input, output);

        }
        catch (cl::Error const & e) {
            std::string log_str = program.getBuildInfo<CL_PROGRAM_BUILD_LOG>(device);
            std::cout << std::endl << e.what() << " : " << e.err() << std::endl;
            std::cout << log_str;
            return 0;
        }


    }
    catch (cl::Error const & e) {
        std::cout << "Error: " << e.what() << " #" << e.err() << std::endl;
    }

    return 0;
}
