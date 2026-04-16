# SOW 1.1 All-to-All集合通信性能仿真
使用ASTRA-sim仿真自定义的All-to-All集合通信算法，网络拓扑和算法都支持自定义的实现

## 环境准备
1. Linux OS: Ubuntu22.04/24.04 X86
2. Setup
```bash
sudo apt -y update
sudo apt -y install coreutils wget vim git
sudo apt -y install gcc-11 g++-11 make cmake 
sudo apt -y install clang-format 
sudo apt -y install libboost-dev libboost-program-options-dev
sudo apt -y install python3.10 python3-pip
sudo apt -y install libprotobuf-dev protobuf-compiler
sudo apt -y install openmpi-bin openmpi-doc libopenmpi-dev
```

```bash
## use virtual environment (conda/mamba)
pip3 install --upgrade pip
pip3 install protobuf==5.28.2
pip3 install --upgrade protobuf    #ignore pip error
pip3 install graphviz pydot
```

3. compile ASTRA-sim
```bash
cd astra-sim
git submodule update --init --recursive 
./build/astra_analytical/build.sh
./build/astra_ns3/build.sh -c
```

## 编译MSCCLang All-to-All算法
1. MSCCLang to XML
```bash
## install msccl-tools
cd collectiveapi/msccl-tools
pip install -r requirements.txt
pip install .

## complile custom all-to-all algorithm
## Python DSL to XML
cd examples/mscclang 
bash ./compile_a2a.sh
```

2. XML to Execution trace(protobuf format)
```bash
## install chakra
### cd project home
cd chakra
pip install -r requirements-dev.txt
pip install .

## convert XML all-to-all collective to ET(Protobuf format)
cd ../collectiveapi/chakra_converter
pip3 install --upgrade protobuf    #ignore pip error
bash ./a2a_convert.sh
```

## 仿真
1. 运行仿真
```bash
### cd project home
cd astra-sim
## setup ns3
touch extern/network_backend/ns-3/scratch/output/flow.txt
touch extern/network_backend/ns-3/scratch/output/trace.txt

## simulate custom all-to-all collective with ASTRA-sim ns-3
cd examples/run_scripts/ns3/
bash ./run_a2a.sh
```

2. 查看仿真结果
```bash
bash ./a2a_report.sh

bash ./a2a_report.sh > report.txt

## report.txt --> report.md (use some AI Agent tools to do this)
```