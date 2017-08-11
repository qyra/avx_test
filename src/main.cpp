#include <iostream>
#include <immintrin.h>
#include <pmmintrin.h>
#include <math.h>
#include <chrono>

using Clock = std::chrono::steady_clock;
using std::chrono::time_point;
using std::chrono::duration_cast;
using std::chrono::nanoseconds;
using namespace std::literals::chrono_literals;

float sum_of_squares(const int n, const float* x, const float* y)
{
    float result = 0.f;
    for(int i = 0; i < n; ++i){
        const float num = x[i] - y[i];
        result += num * num;
    }
    return result;
}

inline float hsum_m128_elements(__m128 v) {
    __m128 shuf = _mm_movehdup_ps(v);        // broadcast elements 3,1 to 2,0
    __m128 sums = _mm_add_ps(v, shuf);
    shuf        = _mm_movehl_ps(shuf, sums); // high half -> low half
    sums        = _mm_add_ss(sums, shuf);
    return        _mm_cvtss_f32(sums);
}

float sse_sum_of_squares(int n, const float* x, const float* y)
{
    __m128 sum_set = _mm_setzero_ps();

    for (; n>3; n-=4) {
        const __m128 a = _mm_loadu_ps(x);
        const __m128 b = _mm_loadu_ps(y);
        const __m128 a_minus_b = _mm_sub_ps(a,b);
        const __m128 a_minus_b_squared = _mm_mul_ps(a_minus_b, a_minus_b);
        sum_set = _mm_add_ps(sum_set, a_minus_b_squared);
        x+=4;
        y+=4;
    }

    //~ const __m128 shuffle1 = _mm_shuffle_ps(sum, sum, _MM_SHUFFLE(1,0,3,2));
    //~ const __m128 sum1 = _mm_add_ps(sum, shuffle1);
    //~ const __m128 shuffle2 = _mm_shuffle_ps(sum1, sum1, _MM_SHUFFLE(2,3,0,1));
    //~ const __m128 sum2 = _mm_add_ps(sum1, shuffle2);
    // with SSE3, we could use hadd_ps, but the difference is negligible

    //~ The haddps instruction performs a horizontal add, meaning that adjacent elements in the same operand are added together. Each 128-bit argument is considered as four 32-bit floating-point elements, numbered from 0 to 3, with 3 being the high-order element. The result of the operation on operand a (A3, A2, A1, A0) and operand b (B3, B2, B1, B0) is (B3 + B2, B1 + B0, A3 + A2, A1 + A0).
    float result = hsum_m128_elements(sum_set);

    if (n)
        result += sum_of_squares(n, x, y);    // remaining 1-3 entries

    return result;
}

static inline float norm(int n, const float* x, const float* y)
{
#ifdef __SSE__
    return sqrt(sse_sum_of_squares(n, x, y));
#else
    return sqrt(sum_of_squares(n, x, y));
#endif
}

static inline float norm_slow(int n, const float* x, const float* y)
{
    return sqrt(sum_of_squares(n, x, y));
}


int main()
{
    int dims = 200;
    int test_count = 100000;
    float a[dims];
    float b[dims];
    time_point<Clock> start;

    start = Clock::now();
    float total = 0.f;
    for(int i = 0; i < test_count; ++i){
        //Set a, b such that all elements are 1 apart.
        for(int j = 0; j < dims; ++j){
            a[j] = i;
            b[j] = i + 1;
        }
        total += 10;
    }
    nanoseconds loop_time = duration_cast<nanoseconds>(Clock::now() - start);
    //~ std::cout << "Decoy result = " << total << std::endl;
    std::cout << "Time to just run the loop = " << loop_time.count() << "ns" << std::endl;

    start = Clock::now();
    total = 0.f;
    for(int i = 0; i < test_count; ++i){
        //Set a, b such that all elements are 1 apart.
        for(int j = 0; j < dims; ++j){
            a[j] = i;
            b[j] = i + 1;
        }
        //sum of squared diffs is dims, so res should always be 4.47213595499958.
        float res = norm(dims, a, b);
        total += res;
    }
    nanoseconds avx_diff = duration_cast<nanoseconds>(Clock::now() - start);
    std::cout << "Result using AVX = " << total << std::endl;
    std::cout << "Time with AVX = " <<avx_diff.count() - loop_time.count() << "ns" << std::endl;

    start = Clock::now();
    total = 0.f;
    for(int i = 0; i < test_count; ++i){
        //Set a, b such that all elements are 1 apart.
        for(int j = 0; j < dims; ++j){
            a[j] = i;
            b[j] = i + 1;
        }
        //sum of squared diffs is dims, so res should always be 4.47213595499958.
        float res = norm_slow(dims, a, b);
        total += res;
    }
    nanoseconds no_avx_diff = duration_cast<nanoseconds>(Clock::now() - start);
    std::cout << "Result without AVX = " << total << std::endl;
    std::cout << "Time without AVX = " <<no_avx_diff.count() - loop_time.count() << "ns" << std::endl;

    std::cout <<"AVX Speedup: " << (double)no_avx_diff.count()/(double)avx_diff.count()*100 << "%" << std::endl;

    return 0;
}
