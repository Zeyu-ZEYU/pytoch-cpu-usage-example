# pytoch-cpu-usage-example
A reproducible example of an issue of PyTorch's CPU usage.

## Envrionment
- CPU: 1 Intel(R) Xeon(R) Silver 4210 CPU @ 2.20GHz (20 logic cores)
- 512GB main memory

## Run
```bash
python3 ./train_cifar_sgd.py  # run local training only using CPU

python3 ./sgd_profiler.py  # run CPU profiler for train_cifar_sgd.py on the same host
```

Let the code run for a while, and then kills ./train_cifar_sgd.py.

In ./logs there are ckpt.log and cpu_usage.log. The file ckpt.log has timestamps for the start and end of parameter update; cpu_usage.log has CPU usage which is recorded every 0.015s.

### Part of ckpt.log
```txt
2022-11-23 14:47:17,532 - ckpt - sleep_start
 # time.sleep(2)
2022-11-23 14:47:19,534 - ckpt - update_start
 # optimizer.step()
 # optimizer.zero_grad()
2022-11-23 14:47:19,551 - ckpt - update_end
 # time.sleep(2)
2022-11-23 14:47:21,554 - ckpt - sleep_end
```

### Part of cpu_usage.log
```txt
2022-11-23 14:47:19,492 - sgd_profiler - 0.0
2022-11-23 14:47:19,512 - sgd_profiler - 0.0
2022-11-23 14:47:19,533 - sgd_profiler - 0.0
 # Start of optimizer.step() and zero_grad()
2022-11-23 14:47:19,553 - sgd_profiler - 1783.5
 # End of optimizer.step() and zero_grad()
 # time.sleep(2) starts
2022-11-23 14:47:19,574 - sgd_profiler - 1882.1
2022-11-23 14:47:19,594 - sgd_profiler - 1889.2
2022-11-23 14:47:19,614 - sgd_profiler - 1938.4
2022-11-23 14:47:19,635 - sgd_profiler - 1889.9
2022-11-23 14:47:19,655 - sgd_profiler - 1890.0
2022-11-23 14:47:19,675 - sgd_profiler - 1939.9
2022-11-23 14:47:19,695 - sgd_profiler - 1939.7
2022-11-23 14:47:19,715 - sgd_profiler - 1840.0
2022-11-23 14:47:19,736 - sgd_profiler - 1890.1
2022-11-23 14:47:19,756 - sgd_profiler - 49.6
2022-11-23 14:47:19,777 - sgd_profiler - 0.0
2022-11-23 14:47:19,797 - sgd_profiler - 0.0
2022-11-23 14:47:19,818 - sgd_profiler - 0.0
```

After optimizer.step() and zero_grad() end and time.sleep(2) starts, there is still high CPU usage. I think openMP threads are doing parallel optimization for CPU tensors during parameter udpate. But why after parameter udpate is done, threads are still running for a while? I have no idea on it.
