#include <pthread.h>
#include <stdio.h>
#include <stdlib.h>

int n_threads = 4;
long long int n_tosses = 1e8;
long long int total_in_circle;
pthread_mutex_t mutexsum; // pthread 互斥鎖

// 每個 thread 要做的任務
void *count_pi(void *args)
{
    long cur_thread = (long)args;
    float x = 0.0f, y = 0.0f;
    long long int in_circle = 0;
    unsigned int seed = time(NULL);

    // 將原本的 PI 算法切成好幾份
    for (long long int i = n_tosses / n_threads; i >= 0; i--)
    {
        x = rand_r(&seed) / ((float)RAND_MAX); //thread-safe random number generator
        y = rand_r(&seed) / ((float)RAND_MAX); //thread-safe random number generator
        if (x * x + y * y <= 1.0f)
        {
            in_circle++;
        }
    }

    // **** critical section ****
    // 一次只允許一個 thread 存取
    pthread_mutex_lock(&mutexsum);
    total_in_circle += in_circle;
    pthread_mutex_unlock(&mutexsum);
    // *****************

    printf("Thread %ld:  local=%lld global=%.lld\n", cur_thread, in_circle, total_in_circle);

    pthread_exit((void *)0);
}

int main(int argc, char *argv[])
{
    if (argc > 1)
    {
        n_threads = atoi(argv[1]);
        n_tosses = atoll(argv[2]);
    }

    printf("n_threads=%d, n_tosses=%lld\n", n_threads, n_tosses);
    pthread_t callThd[n_threads]; // 宣告建立 pthread

    clock_t starttime, endtime;
    starttime = clock();
    // 初始化互斥鎖
    pthread_mutex_init(&mutexsum, NULL);

    // 設定 pthread 性質是要能 join
    pthread_attr_t attr;
    pthread_attr_init(&attr);
    pthread_attr_setdetachstate(&attr, PTHREAD_CREATE_JOINABLE);

    for (int i = 0; i < n_threads; i++)
    {
        // 建立一個 thread，執行 count_pi 任務，傳入 arg[i] 指標參數
        pthread_create(&callThd[i], NULL, count_pi, (void *)(size_t)i);
    }

    // 回收性質設定
    pthread_attr_destroy(&attr);

    void *status;
    for (int i = 0; i < n_threads; i++)
    {
        // 等待每一個 thread 執行完畢
        pthread_join(callThd[i], NULL);
    }
    endtime = clock();
    float pi_estimated = 4 * total_in_circle / ((float)n_tosses);
    // 所有 thread 執行完畢，印出 PI
    printf("Pi =  %.10lf \n", pi_estimated);

    double diff = endtime - starttime; // ms
    printf("runtime: %f  sec \n", diff / CLOCKS_PER_SEC);

    // 回收互斥鎖
    pthread_mutex_destroy(&mutexsum);
    // 離開
    pthread_exit(NULL);
}