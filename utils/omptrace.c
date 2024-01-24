#define _GNU_SOURCE

#include <stdio.h>
#include <stdarg.h>
#include <malloc.h>
#include <dlfcn.h>
#include <execinfo.h>
#include <time.h>
#include <sys/syscall.h>
#include <signal.h>
#define gettid() ((int)syscall(SYS_gettid))

//static __thread int inside_gomp = 0;
static __thread double lst_end = 0, omp_total = 0.0, lst_rst = 0.0, gomp_parallel_start_time = 0.0;
static __thread long omp_calls = 0;
static __thread int inside_gomp_parallel = 0;

double get_time() {
  static int init_done = 0;
  static struct timespec stp = {0,0};
  struct timespec tp;
  clock_gettime(CLOCK_REALTIME, &tp);
  //clock_gettime(CLOCK_PROCESS_CPUTIME_ID, &tp);

  if(!init_done) {
    init_done = 1;
    stp = tp;
  }
  double ret = (tp.tv_sec - stp.tv_sec) * 1e3 + (tp.tv_nsec - stp.tv_nsec)*1e-6;
  return ret;
}

void reset_stats(double cur_time)
{
  lst_rst = cur_time;
  omp_total = 0.0;
  omp_calls = 0;
}

void handle_sigint(int sig) 
{ 
  double cur_time = get_time();
  printf("omp: %.3f   total: %.3f  calls: %ld\n", omp_total, cur_time - lst_rst, omp_calls);
  reset_stats(cur_time);

} 

static void (*real_GOMP_parallel)(void (*) (void *), void*, unsigned, unsigned int)=NULL;
static void (*real_GOMP_parallel_start)(void (*) (void *), void*, unsigned)=NULL;
static void (*real_GOMP_parallel_end)(void)=NULL;
//static void (*real_kmpc_fork_call)(void*, int, void*, ...)=NULL;
static int (*real_kmp_fork_call)(void *, int, int, int, void *, void *, va_list *)=NULL;
//int __kmp_fork_call(void *loc, int gtid, int call_context, int argc, void *microtask, void *invoker, va_list *ap)

static void omptrace_init(void)
{
    real_GOMP_parallel = dlsym(RTLD_NEXT, "GOMP_parallel");
    if (NULL == real_GOMP_parallel) {
        //fprintf(stderr, "Error in `dlsym`: %s\n", dlerror());
    }
    real_kmp_fork_call = dlsym(RTLD_NEXT, "__kmp_fork_call");
    if (NULL == real_kmp_fork_call) {
        //fprintf(stderr, "Error in `dlsym`: %s\n", dlerror());
    }
    real_GOMP_parallel_start = dlsym(RTLD_NEXT, "GOMP_parallel_start");
    real_GOMP_parallel_end = dlsym(RTLD_NEXT, "GOMP_parallel_end");
#if 0
    real_kmpc_fork_call = dlsym(RTLD_NEXT, "__kmpc_fork_call");
    if (NULL == real_kmpc_fork_call) {
        //fprintf(stderr, "Error in `dlsym`: %s\n", dlerror());
    }
#endif
    reset_stats(get_time());
    signal(SIGUSR1, handle_sigint);
    fprintf(stderr, "omptrace_init called\n");
}

static inline void print_trace(int skip)
{
    void* callstack[6];
    int i, frames = backtrace(callstack, 6);
    char** strs = backtrace_symbols(callstack, frames);
    for (i = skip; i < frames; ++i) {
            fprintf(stdout, "OMP STACK %d: %s\n", i, strs[i]);
    }
    free(strs);
}

static inline void print_func(void *func)
{
    void* callstack[1] = {func};
    int i;
    char** strs = backtrace_symbols(callstack, 1);
    for (i = 0; i < 1; ++i) {
            fprintf(stdout, "OMP STACK %d: %s\n", i, strs[i]);
    }
    free(strs);
}

void GOMP_parallel_start (void (*fn) (void *), void *data, unsigned num_threads)
{
  if(real_GOMP_parallel==NULL) {
    omptrace_init();
  }
  if (num_threads != 1) gomp_parallel_start_time = get_time();
  real_GOMP_parallel_start(fn, data, num_threads);
}

void GOMP_parallel_end (void)
{
  if(real_GOMP_parallel==NULL) {
    omptrace_init();
  }
  real_GOMP_parallel_end();
  if (inside_gomp_parallel) return;
  if (gomp_parallel_start_time > 0) {
    double t1 = get_time();
    lst_end = t1;
    omp_total += t1 - gomp_parallel_start_time;
    omp_calls++;
    gomp_parallel_start_time = 0;
  }
}

void GOMP_parallel (void (*fn) (void *), void *data, unsigned num_threads, unsigned int flags)
{
    if(real_GOMP_parallel==NULL) {
        omptrace_init();
    }
    //printf("Calling GOMP_parallel num_threads = %d\n", num_threads);
    //inside_gomp = 1;
    //print_trace(2);
    inside_gomp_parallel = 1;
    double t0 = get_time();
    real_GOMP_parallel(fn, data, num_threads, flags);
    double t1 = get_time();
    inside_gomp_parallel = 0;
    //inside_gomp = 0;
    //printf("GOMP_parallel: %8g ms num_threads: %d tid: %d\n", (t1-t0), num_threads, gettid());
    if(num_threads != 1) {
      //printf("GOMP_parallel: %8g ms ST: %12.3f ET: %12.3f GT: %8g num_threads: %d tid: %d\n", (t1-t0), t0, t1, (t0-lst_end), num_threads, gettid());
      //print_func((void*)fn);
      lst_end = t1;
      omp_total += t1 - t0;
      omp_calls++;
    }
}

#if 0
void __kmpc_fork_call (void *loc, int nargs, void *microtask, ...)
{
    if(real_kmpc_fork_call==NULL) {
        omptrace_init();
    }
    printf("Calling __kmpc_fork_call\n");
    va_list ap;
    va_start(ap, microtask);
    real_kmpc_fork_call(loc, nargs, microtask, &ap);
    va_end(ap);
}
#endif

int __kmp_fork_call(void *loc, int gtid, int call_context, int argc, void *microtask, void *invoker, va_list *ap)
{
    if(real_kmp_fork_call==NULL) {
        omptrace_init();
    }
    if(call_context == 0) return real_kmp_fork_call(loc, gtid, call_context, argc, microtask, invoker, ap);
    //printf("Calling __kmp_fork_call\n");
    //print_trace(3);
    double t0 = get_time();
    int ret = real_kmp_fork_call(loc, gtid, call_context, argc, microtask, invoker, ap);
    double t1 = get_time();
    //printf("__kmp_fork_call: %8g ms ST: %8g ET: %8g omp_compiler = %s tid: %d, gtid: %d argc: %d\n", (t1-t0), t0, t1, (call_context==0 ? "gcc" : "icc"), gettid(), gtid, argc);
    //printf("__kmp_fork_call: %8g ms ST: %12.3f ET: %12.3f GT: %8g tid: %8d, gtid: %d\n", (t1-t0), t0, t1, (t0-lst_end), gettid(), gtid);
    lst_end = t1;
    omp_total += t1 - t0;
    omp_calls++;
    return ret;
}
