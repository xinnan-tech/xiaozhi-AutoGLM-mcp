# xiaozhi-AutoGLM-mcp

## 项目介绍

这是一个基于 [Open-AutoGLM](https://github.com/zai-org/Open-AutoGLM) 构建的小智MCP服务，作为MCP服务接入[小智AI](https://xiaozhi.me/)、[xiaozhi-esp32-server](https://github.com/xinnan-tech/xiaozhi-esp32-server) 被小智调用。

## 环境准备

### 1. ADB (Android Debug Bridge)

1. 下载官方 ADB [安装包](https://developer.android.com/tools/releases/platform-tools?hl=zh-cn)，并解压到自定义路径
2. 配置环境变量

- MacOS 配置方法：在 `Terminal` 或者任何命令行工具里

  ```bash
  # 假设解压后的目录为 ~/Downlaods/platform-tools。如果不是请自行调整命令。
  export PATH=${PATH}:~/Downloads/platform-tools
  ```

- Windows 配置方法：可参考 [第三方教程](https://blog.csdn.net/x2584179909/article/details/108319973) 进行配置。

### 2. Android 7.0+ 的设备或模拟器，并启用 `开发者模式` 和 `USB 调试`

1. 开发者模式启用：通常启用方法是，找到 `设置-关于手机-版本号` 然后连续快速点击 10
   次左右，直到弹出弹窗显示“开发者模式已启用”。不同手机会有些许差别，如果找不到，可以上网搜索一下教程。
2. USB 调试启用：启用开发者模式之后，会出现 `设置-开发者选项-USB 调试`，勾选启用
3. 部分机型在设置开发者选项以后, 可能需要重启设备才能生效. 可以测试一下: 将手机用USB数据线连接到电脑后, `adb devices`
   查看是否有设备信息, 如果没有说明连接失败.

**请务必仔细检查相关权限**

![权限](resources/screenshot-20251209-181423.png)

### 3. 安装 ADB Keyboard（用于文本输入）

下载 [安装包](https://github.com/senzhk/ADBKeyBoard/blob/master/ADBKeyboard.apk) 并在对应的安卓设备中进行安装。
注意，安装完成后还需要到 `设置-输入法` 或者 `设置-键盘列表` 中启用 `ADB Keyboard` 才能生效

## 部署准备工作

### 1. 安装依赖

```bash
conda remove -n autoglm --all -y
conda create -n autoglm python=3.10 -y
conda activate autoglm

pip install -r requirements.txt 
```

### 2. 配置 ADB

确认 **USB数据线具有数据传输功能**, 而不是仅有充电功能

确保已安装 ADB 并使用 **USB数据线** 连接设备：

```bash
# 检查已连接的设备
adb devices

# 输出结果应显示你的设备，如：
# List of devices attached
# emulator-5554   device
```

### 3. 配置视觉模型和小智MCP接入点
3.1 复制 `config.yaml` 为 `.config.yaml` 文件。

3.2 登录`小智AI`或者`你私有化部署的智控台`，获取智能体的MCP接入点地址。

3.3 编辑 `.config.yaml` 文件，将获取到的MCP接入点地址替换到 `mcp_endpoint` 字段中。

3.4 如果你的 `.config.yaml`选择了 `ChatGLMVLLM` 作为视觉语言大模型，那么请前往[智谱AI](https://open.bigmodel.cn/usercenter/proj-mgmt/apikeys)平台，获取你的密钥。请主要账户余额充足，否则会导致调用失败。

3.5 编辑 `.config.yaml` 文件，将获取到的密钥替换到 `VLLM`下的 `ChatGLMVLLM`下的 `api_key` 字段中。

### 4. 运行服务
#### Linux/macOS:

```bash
./start_mcp_service.sh
```

#### Windows:

```batch
start_mcp_service.bat
```