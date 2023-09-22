#include <iostream>
#include <omp.h>

using namespace std;

int main()
{
  int N = 100;
  int nthreads, tid, idx;
  float a[N], b[N], c[N];

#pragma omp parallel master
  {
    nthreads = omp_get_num_threads();
    cout << "Num. Threads " << nthreads << endl;
  }
#pragma omp parallel for
  for (idx = 0; idx < N; idx++)
  {
    a[idx] = b[idx] = 1.0;
  }

#pragma omp parallel for private(tid)

  for (idx = 0; idx < N; idx++)
  {
    c[idx] = a[idx] + b[idx];
    tid = omp_get_thread_num();
    cout << "Thread n. " << tid << ", c[" << idx << "]=" << c[idx] << endl;
  }

  return 0;
}