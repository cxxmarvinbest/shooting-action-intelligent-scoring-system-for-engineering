
# python

#### 步骤1：安装pybind11

```bash

## 安装opencv
sudo apt-get install libopencv-dev

#方式一 克隆pybind11源码
git clone https://github.com/pybind/pybind11.git
cd pybind11

#方式二 下载pybind11压缩包复制进去解压
  pybind11-3.0.4.zip
  pybind11-master.zip

#先确认工具若无则安装 
  sudo apt install unzip -y

#二选一
  unzip pybind11-3.0.4.zip
  cd pybind11-3.0.4

  unzip pybind11-master.zip
  cd pybind11-master

# 创建构建目录并编译安装
mkdir build && cd build
cmake .. -DPYBIND11_TEST=OFF
sudo make install
```

#### 步骤2：编译 mpp

#命名行编译

mkdir build && cd build
cmake ..
make -j8

#脚本

python3 mpp_build_pybind.py

#运行

python3 main.py