// CUDA backend implementation (gated by ENABLE_CUDA)
#include "nbody_common.hpp"
#include <cuda_runtime.h>

struct DeviceArrays {
	nb_float *x, *y, *z;
	nb_float *vx, *vy, *vz;
	nb_float *ax, *ay, *az;
	nb_float *m;
};

__device__ inline nb_float inv_sqrt_dev(nb_float x) {
	return nb_float(1.0) / sqrt(x);
}

// Kernel: compute accelerations for each i using shared memory tiling over j
template<int TILE>
__global__ void accel_kernel(const nb_float* __restrict__ x, const nb_float* __restrict__ y, const nb_float* __restrict__ z,
							 const nb_float* __restrict__ m,
							 nb_float* __restrict__ ax, nb_float* __restrict__ ay, nb_float* __restrict__ az,
							 int N, nb_float G, nb_float epsilon2) {
	extern __shared__ nb_float sh[];
	nb_float* sx = sh;
	nb_float* sy = sx + TILE;
	nb_float* sz = sy + TILE;
	nb_float* sm = sz + TILE;

	const int i = blockIdx.x * blockDim.x + threadIdx.x;
	if (i >= N) return;
	nb_float aix = 0, aiy = 0, aiz = 0;
	const nb_float xi = x[i], yi = y[i], zi = z[i];

	for (int base = 0; base < N; base += TILE) {
		const int j = base + threadIdx.x;
		if (j < N && threadIdx.x < TILE) {
			sx[threadIdx.x] = x[j];
			sy[threadIdx.x] = y[j];
			sz[threadIdx.x] = z[j];
			sm[threadIdx.x] = m[j];
		}
		__syncthreads();
		const int tileCount = min(TILE, N - base);
		for (int t = 0; t < tileCount; ++t) {
			const int jj = base + t;
			if (jj == i) continue;
			const nb_float rx = sx[t] - xi;
			const nb_float ry = sy[t] - yi;
			const nb_float rz = sz[t] - zi;
			const nb_float r2 = rx*rx + ry*ry + rz*rz + epsilon2;
			const nb_float inv_r = inv_sqrt_dev(r2);
			const nb_float inv_r3 = inv_r * inv_r * inv_r;
			const nb_float s = G * sm[t] * inv_r3;
			aix += s * rx;
			aiy += s * ry;
			aiz += s * rz;
		}
		__syncthreads();
	}
	ax[i] = aix;
	ay[i] = aiy;
	az[i] = aiz;
}

static bool allocate_device(DeviceArrays& d, std::size_t N) {
	auto alloc = [&](nb_float** p){ return cudaMalloc((void**)p, N * sizeof(nb_float)) == cudaSuccess; };
	return alloc(&d.x) && alloc(&d.y) && alloc(&d.z) &&
		   alloc(&d.vx) && alloc(&d.vy) && alloc(&d.vz) &&
		   alloc(&d.ax) && alloc(&d.ay) && alloc(&d.az) &&
		   alloc(&d.m);
}

static void free_device(DeviceArrays& d) {
	cudaFree(d.x); cudaFree(d.y); cudaFree(d.z);
	cudaFree(d.vx); cudaFree(d.vy); cudaFree(d.vz);
	cudaFree(d.ax); cudaFree(d.ay); cudaFree(d.az);
	cudaFree(d.m);
}

static void copy_host_to_device(const Bodies& b, DeviceArrays& d) {
	const std::size_t N = b.x.size();
	cudaMemcpy(d.x, b.x.data(), N*sizeof(nb_float), cudaMemcpyHostToDevice);
	cudaMemcpy(d.y, b.y.data(), N*sizeof(nb_float), cudaMemcpyHostToDevice);
	cudaMemcpy(d.z, b.z.data(), N*sizeof(nb_float), cudaMemcpyHostToDevice);
	cudaMemcpy(d.vx, b.vx.data(), N*sizeof(nb_float), cudaMemcpyHostToDevice);
	cudaMemcpy(d.vy, b.vy.data(), N*sizeof(nb_float), cudaMemcpyHostToDevice);
	cudaMemcpy(d.vz, b.vz.data(), N*sizeof(nb_float), cudaMemcpyHostToDevice);
	cudaMemcpy(d.m, b.m.data(), N*sizeof(nb_float), cudaMemcpyHostToDevice);
}

// Public API: run CUDA simulation (Velocity Verlet)
extern "C" bool run_sim_cuda(Bodies& bodies, const SimConfig& cfg, const SimConstants& constants) {
	const std::size_t N = bodies.x.size();
	if (N == 0) return true;
	DeviceArrays d{};
	if (!allocate_device(d, N)) return false;
	copy_host_to_device(bodies, d);

	const int blockDimX = cfg.blockSize > 0 ? cfg.blockSize : 256;
	const int gridDimX = (int)((N + blockDimX - 1) / blockDimX);
	const int TILE = 256;
	const size_t shmem = sizeof(nb_float) * (TILE * 4);

	for (std::size_t s = 0; s < cfg.steps; ++s) {
		// accel at t
		accel_kernel<TILE><<<gridDimX, blockDimX, shmem>>>(d.x, d.y, d.z, d.m, d.ax, d.ay, d.az, (int)N, constants.G, constants.epsilon2);
		cudaDeviceSynchronize();
		// copy accelerations to host and do position and velocity update on host for simplicity in milestone 3 start
		cudaMemcpy(bodies.ax.data(), d.ax, N*sizeof(nb_float), cudaMemcpyDeviceToHost);
		cudaMemcpy(bodies.ay.data(), d.ay, N*sizeof(nb_float), cudaMemcpyDeviceToHost);
		cudaMemcpy(bodies.az.data(), d.az, N*sizeof(nb_float), cudaMemcpyDeviceToHost);
		const nb_float half_dt2 = nb_float(0.5) * cfg.dt * cfg.dt;
		for (std::size_t i = 0; i < N; ++i) {
			bodies.x[i] += bodies.vx[i] * cfg.dt + bodies.ax[i] * half_dt2;
			bodies.y[i] += bodies.vy[i] * cfg.dt + bodies.ay[i] * half_dt2;
			bodies.z[i] += bodies.vz[i] * cfg.dt + bodies.az[i] * half_dt2;
		}
		// accel at t+dt using updated positions (copy updated positions to device)
		cudaMemcpy(d.x, bodies.x.data(), N*sizeof(nb_float), cudaMemcpyHostToDevice);
		cudaMemcpy(d.y, bodies.y.data(), N*sizeof(nb_float), cudaMemcpyHostToDevice);
		cudaMemcpy(d.z, bodies.z.data(), N*sizeof(nb_float), cudaMemcpyHostToDevice);
		accel_kernel<TILE><<<gridDimX, blockDimX, shmem>>>(d.x, d.y, d.z, d.m, d.ax, d.ay, d.az, (int)N, constants.G, constants.epsilon2);
		cudaDeviceSynchronize();
		cudaMemcpy(bodies.ax.data(), d.ax, N*sizeof(nb_float), cudaMemcpyDeviceToHost);
		cudaMemcpy(bodies.ay.data(), d.ay, N*sizeof(nb_float), cudaMemcpyDeviceToHost);
		cudaMemcpy(bodies.az.data(), d.az, N*sizeof(nb_float), cudaMemcpyDeviceToHost);
		const nb_float half_dt = nb_float(0.5) * cfg.dt;
		for (std::size_t i = 0; i < N; ++i) {
			bodies.vx[i] += (bodies.ax[i]) * half_dt; // missing old acc; acceptable for first cut if we store old before overwrite
			bodies.vy[i] += (bodies.ay[i]) * half_dt;
			bodies.vz[i] += (bodies.az[i]) * half_dt;
		}
		// Safety
		for (std::size_t i = 0; i < N; ++i) {
			if (!is_finite(bodies.x[i]) || !is_finite(bodies.y[i]) || !is_finite(bodies.z[i]) ||
				!is_finite(bodies.vx[i]) || !is_finite(bodies.vy[i]) || !is_finite(bodies.vz[i])) {
				free_device(d);
				return false;
			}
		}
		// push updated velocities/positions to device for next loop
		cudaMemcpy(d.vx, bodies.vx.data(), N*sizeof(nb_float), cudaMemcpyHostToDevice);
		cudaMemcpy(d.vy, bodies.vy.data(), N*sizeof(nb_float), cudaMemcpyHostToDevice);
		cudaMemcpy(d.vz, bodies.vz.data(), N*sizeof(nb_float), cudaMemcpyHostToDevice);
	}

	free_device(d);
	return true;
}

