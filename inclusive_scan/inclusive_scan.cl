#define SWAP(a,b) {__local float * tmp=a; a=b; b=tmp;}

void threat_subblock(uint block_size, uint local_id,
		             __global float* input, __global float* output,
				     __local float* a_tmp, __local float* b_tmp)
{
    a_tmp[local_id] = b_tmp[local_id] = input[local_id];
    barrier(CLK_LOCAL_MEM_FENCE);
 
    for(uint s = 1; s < block_size; s <<= 1)
    {
        if(local_id > (s - 1))
        {
            b_tmp[local_id] = a_tmp[local_id] + a_tmp[local_id - s];
        }
        else
        {
            b_tmp[local_id] = a_tmp[local_id];
        }
        barrier(CLK_LOCAL_MEM_FENCE);
        SWAP(a_tmp, b_tmp);
    }
    output[local_id] = a_tmp[local_id];
}

__kernel void subblock_scan(__global float* input, __global float* output, __global float* last_elements,
							__local float* a_tmp, __local float* b_tmp)
{
	uint group_id = get_group_id(0);
	uint group_size = get_local_size(0);
	uint local_id = get_local_id(0);
	uint offset = group_id * group_size;
	threat_subblock(group_size, local_id, input + offset, output + offset, a_tmp + offset, b_tmp + offset);
	if (local_id + 1 == group_size)
	{
		last_elements[group_id] = output[offset + local_id];
	}
}

__kernel void small_array_scan(__global float* input, __global float* output,
		                       __local float* a_tmp, __local float* b_tmp)
{
	uint group_size = get_local_size(0);
	uint local_id = get_local_id(0);
	threat_subblock(group_size, local_id, input, output, a_tmp, b_tmp);
}

__kernel void merge(__global float* input, __global float* output, __global float* additions)
{
	uint global_id = get_global_id(0);
	uint block_size = get_local_size(0);
	if (global_id / block_size > 0) {
		output[global_id] = input[global_id] + additions[global_id / block_size - 1];
	}
	else {
		output[global_id] = input[global_id];
	}
}
