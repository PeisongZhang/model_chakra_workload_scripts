# 20260416 

## analytical

- [x] 分析analytical customize topology仿真的准确性
  - [x] 和analytical原生版本比较原生CCL Ring All-Gather执行准确性
  - [x] 和ns3比较原生CCL Ring All-Gather执行(async Ring)        1258780(analytical) vs 1264384(ns3)
  - [ ] ~~和analytical原生版本比较llama3_8b执行准确性~~
  - [x] 和ns3比较llama3_8b(inter dc localsgd)执行准确性  136325230394(analytical) vs 136036790518(ns3)
  
- [x] analytical customize topology split flow
  - [x] implement
  - [x] test

- [x] analytical customize topology configuration file
  - [x] how to use: custom topology and topology file
  - [x] simplify

## astra sim simulation index

- [x] wall time: cycles
- [x] comm time
- [x] GPU time
- [x] compute-communication overlap: GPU time + Comm time - wall time
- [x] average compute utilization: %   $ \frac {\sum({perf_i} \times {duration_i})} {\sum{duration_i}  \times {perf_{peak}} }  $
- [ ] average memory utilization: %
- [ ] average operation intensity: float

## question: when will the ns3 backend degrade to analytical backend?

- [ ] how to change the parameter of ns3, to make it behaves like analytical backend, is it the MTU?
