#include <cuda_runtime.h>
#include <sys/types.h>
#include <unistd.h>

#include <chrono>
#include <cstdio>
#include <string>
#include <thread>
#include <vector>

static int fail(cudaError_t err, const char *op) {
    if (err == cudaSuccess) {
        return 0;
    }
    std::fprintf(stderr, "%s failed: %s\n", op, cudaGetErrorString(err));
    return 1;
}

int main() {
    constexpr size_t n = 16 * 1024 * 1024;
    unsigned char *d = nullptr;
    if (fail(cudaMalloc(reinterpret_cast<void **>(&d), n), "cudaMalloc")) {
        return 1;
    }
    if (fail(cudaMemset(d, 0x5a, n), "cudaMemset")) {
        return 2;
    }
    if (fail(cudaDeviceSynchronize(), "cudaDeviceSynchronize")) {
        return 3;
    }

    pid_t pid = getpid();
    std::printf("READY %d %p\n", static_cast<int>(pid), static_cast<void *>(d));
    std::fflush(stdout);

    std::string go = "/tmp/gpucr_restore_go_" + std::to_string(static_cast<int>(pid));
    for (int i = 0; i < 600 && access(go.c_str(), F_OK) != 0; ++i) {
        std::this_thread::sleep_for(std::chrono::milliseconds(100));
    }

    std::vector<unsigned char> h(n);
    if (fail(cudaMemcpy(h.data(), d, n, cudaMemcpyDeviceToHost), "cudaMemcpy")) {
        return 4;
    }
    for (size_t i = 0; i < n; ++i) {
        if (h[i] != 0x5a) {
            std::fprintf(stderr, "VERIFY_FAIL at %zu got %u\n", i, h[i]);
            return 5;
        }
    }
    if (fail(cudaFree(d), "cudaFree")) {
        return 6;
    }
    std::puts("VERIFY_OK");
    return 0;
}
