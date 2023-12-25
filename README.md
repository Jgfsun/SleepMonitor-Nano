自动睡眠分期系统实现：使用OpenBCI设备通过Lab Streaming Layer (LSL)与部署在Jetson NANO上的深度神经网络进行数据通信并分期结果显示，从而构建睡眠分期实时监测系统。
**Python 版本2.7.17**

```
sudo apt install python2-pip
```

​	**1.安装tensorflow（适用于NANO）**

```
pip2 install --pre --extra-index-url https://developer.download.nvidia.com/compute/redist/jp/v40/tensorflow-gpu/
```

​	若安装报错，则去网址下载到本地安装

```
pip2 install 文件路径/文件名.whl
```

​	2.[NX安装方法](https://forums.developer.nvidia.com/t/official-tensorflow-for-jetson-agx-xavier/65523)选择对应版本的安装教程就好

**2.安装pyqt5,在nano上安装比较特殊（qt5在ubuntu默认库中）**

```
sudo apt-get install qt5-default
sudo apt-get install python-pyqt5
apt install python2-pyqtgraph	# 如果是python3则是python3-pyqtgraph
```

​	如果不是nano/nx则使用`pip install python-qt5`命令安装

**3.安装其他环境包（可能不全，但是主要的包都包含在里面，其他的可能需要自己安装）**

```
pip2 install -r requirements.txt
```

**4.覆盖pylsl下的libls64.so文件**
	原因：nano是arm架构的，直接pip的是x86架构的，不兼容。
	复制代码参考：（liblsl64文件下两个覆盖掉原来的liblsl64.so即可）

```
cp -rf /home/johnny/Desktop/sleepeegnet/liblsl-1.13.0-b7/build/install/LSL/lib/. /usr/local/lib/python2.7/dist-packages/pylsl
```

​	4.1 移到NX报错：总线错误（核心已转储）

​	原因：复制的代码是nano上编译的，我移动到nx上需要重新编译。编译过程：[参考链接](https://github.com/labstreaminglayer/liblsl-Python/issues/16)
​				重新编译后查看USB端口ls -l /dev/ttyUSB*,运行lsl_data.py检查LSL

​	4.2 首次使用Ubuntu时，usb设备对用户不开放权限，导致usb转串口数据无法读写。
​	解决方法：
​		单次访问权限解决方法：sudo chmod -R 777 /dev/ttyUSB0。
​		永久权限：~~sudo gedit /etc/group；在dialout: x : 20: 后加上username（好像不可以）~~

**5.代码结构**

​	ui3.py 界面渲染

​	eeg_ui3.py 界面绘图

​	lsl_data.py nano与OpenBCI建立连接

​	inlet.py 数据读取并建立数据缓存

​	network2.py 运行分期网络

​	utils.py 一些函数方法

**6.测试代码**

​	eeg_ui_test.py 测试界面能否使用，读取的数据是本地eeg_data.npz的数据

​	ui_test.py 测试界面直接读取LSL流的数据（注意，Python的scipy版本包对不对，如果版本太低会报错）

**7.运行代码**

​	①python lsl_data.py与OpenBCI建立连接

​	②新开一个终端：python network2.py 运行网络，显示GUI。

