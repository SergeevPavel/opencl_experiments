
#define __CL_ENABLE_EXCEPTIONS
#include "CL/cl.hpp"

#include <iostream>
#include <fstream>
#include <vector>
#include <string>
#include <iterator>
#include <iomanip>

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
        std::ifstream cl_file("../task1/vector_add.cl");
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
            std::string log_str = program.getBuildInfo<CL_PROGRAM_BUILD_LOG>(devices[0]);
            std::cout << std::endl << e.what() << " : " << e.err() << std::endl;
            std::cout << log_str;
            return 0;
        }

        // create a message to send to kernel
        size_t const block_size = 8;
        size_t const test_array_size = 16;

        std::vector<int> a(test_array_size);
        std::vector<int> b(test_array_size);
        std::vector<int> c(test_array_size, 1);
        for (size_t i = 0; i < test_array_size; ++i)
        {
            a[i] = i;
            b[i] = i;
        }

        // allocate device buffer to hold message
        cl::Buffer dev_a(context, CL_MEM_READ_ONLY, sizeof(int) * test_array_size);
        cl::Buffer dev_b(context, CL_MEM_READ_ONLY, sizeof(int) * test_array_size);
        cl::Buffer dev_c(context, CL_MEM_WRITE_ONLY, sizeof(int) * test_array_size);

        // copy from cpu to gpu
        queue.enqueueWriteBuffer(dev_a, CL_TRUE, 0, sizeof(int) * test_array_size, &a[0]);
        queue.enqueueWriteBuffer(dev_b, CL_TRUE, 0, sizeof(int) * test_array_size, &b[0]);

        // load named kernel from opencl source
        auto vector_add = cl::make_kernel<cl::Buffer&, cl::Buffer&, cl::Buffer&, int>(program, "vector_add");
        auto enqueue_args = cl::EnqueueArgs(queue, cl::NDRange(test_array_size), cl::NDRange(block_size));
        vector_add(enqueue_args, dev_a, dev_b, dev_c, test_array_size);
        queue.enqueueReadBuffer(dev_c, CL_TRUE, 0, sizeof(int) * test_array_size, &c[0]);

        for (size_t i = 0; i < test_array_size; ++i)
           std::cout << c[i] << std::endl;
        std::cout << std::endl;
    }
    catch (cl::Error const & e) {
        std::cout << "Error: " << e.what() << " #" << e.err() << std::endl;
    }

    return 0;
}
