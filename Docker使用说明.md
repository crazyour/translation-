# 实时英日翻译：从零安装 Docker 并运行

本项目的标准交付方式是 Docker：朋友不需要安装 Python，也不使用 DMG。从安装 Docker、获取代码、构建镜像到首次运行，按下面的顺序操作即可。

> 平台限制：macOS 和 Windows 的 Docker Desktop 可以运行命令行翻译，但默认无法把宿主机麦克风直接映射给 Linux 容器。需要麦克风和 Tk 窗口的实时模式只建议在 Linux 宿主机上使用本 Docker 配置。

首次启动会自动下载模型，不需要手工把模型复制进项目。默认 Qwen3 4B 翻译模型约 2.2–2.4 GB；实时模式还会下载语音识别和断句模型，请预留数 GB 磁盘空间。下载可以断点续传，完成后程序会校验必要文件。Docker 模式下模型保存在命名卷 `translator-models` 中，重建容器不会重复下载。

## 第 1 步：安装 Docker

安装完成后必须先启动 Docker Desktop，并等到状态显示 Docker Engine 正在运行，再执行项目命令。下面只需选择朋友电脑对应的系统。

### macOS

已经安装 Homebrew 时，在终端执行：

```sh
brew install --cask docker
open -a Docker
```

第一次打开需要按界面提示授权。然后检查：

```sh
docker --version
docker compose version
```

没有 Homebrew 时，按 [Docker 官方 macOS 安装说明](https://docs.docker.com/desktop/setup/install/mac-install/) 下载与芯片匹配的 `Docker.dmg`。

### Windows 10/11

以管理员身份打开 PowerShell。如果尚未安装 WSL 2，先执行并按提示重启：

```powershell
wsl --install
```

然后安装并启动 Docker Desktop：

```powershell
winget install --exact --id Docker.DockerDesktop
Start-Process "C:\Program Files\Docker\Docker\Docker Desktop.exe"
```

检查安装：

```powershell
docker --version
docker compose version
```

如果 `winget` 不可用，请使用 [Docker 官方 Windows 安装说明](https://docs.docker.com/desktop/setup/install/windows-install/)。

### Ubuntu Linux

下面使用 Docker 官方软件源：

```sh
sudo apt update
sudo apt install -y ca-certificates curl
sudo install -m 0755 -d /etc/apt/keyrings
sudo curl -fsSL https://download.docker.com/linux/ubuntu/gpg -o /etc/apt/keyrings/docker.asc
sudo chmod a+r /etc/apt/keyrings/docker.asc
```

继续添加软件源并安装 Engine 与 Compose：

```sh
echo "Types: deb
URIs: https://download.docker.com/linux/ubuntu
Suites: $(. /etc/os-release && echo "${UBUNTU_CODENAME:-$VERSION_CODENAME}")
Components: stable
Architectures: $(dpkg --print-architecture)
Signed-By: /etc/apt/keyrings/docker.asc" | sudo tee /etc/apt/sources.list.d/docker.sources >/dev/null

sudo apt update
sudo apt install -y docker-ce docker-ce-cli containerd.io docker-buildx-plugin docker-compose-plugin
sudo usermod -aG docker "$USER"
newgrp docker
docker run --rm hello-world
```

`docker` 用户组可控制 Docker daemon，权限接近管理员；只应把可信用户加入该组。

完整要求与其他 Linux 发行版请看 [Docker 官方 Ubuntu 安装说明](https://docs.docker.com/engine/install/ubuntu/)。

## 第 2 步：获取代码并进入目录

如果收到的是 ZIP 压缩包，macOS/Linux 终端执行：

```sh
cd "$HOME/Downloads"
unzip "翻译软件.zip" -d "翻译软件"
cd "翻译软件"
```

Windows PowerShell 执行：

```powershell
Set-Location "$HOME\Downloads"
Expand-Archive -Path ".\翻译软件.zip" -DestinationPath ".\翻译软件"
Set-Location ".\翻译软件"
```

如果代码放在 Git 仓库，则执行：

```sh
git clone <仓库地址> "翻译软件"
cd "翻译软件"
```

确认当前目录可以看到 `Dockerfile`、`compose.yaml` 和 `realtime.py`。

## 第 3 步：构建镜像

在项目目录构建镜像：

```sh
docker compose build
```

这一步会安装 Python、CPU 版 PyTorch、音频库和 llama.cpp。第一次构建可能需要较长时间；以后只有代码或依赖变化时才需要重建。模型不在构建阶段下载。

## 第 4 步：首次运行

### macOS、Windows、Linux：先测试命令行翻译

翻译一句话：

```sh
docker compose run --rm translator translate "Good morning"
```

第一次执行时会自动下载 Qwen3 4B GGUF 模型（约 2.4 GB），下载完成后加载模型并输出中文。请不要关闭终端。

不提供句子时会进入交互模式：

```sh
docker compose run --rm translator translate
```

### Linux：启动实时翻译窗口

Docker 需要连接宿主机的 X11 和 `/dev/snd`。先安装 `xhost`：

```sh
sudo apt install -y x11-xserver-utils
```

再临时允许容器中的 root 用户连接 X11：

```sh
xhost +si:localuser:root
docker compose -f compose.yaml -f compose.linux.yaml run --rm translator realtime
xhost -si:localuser:root
```

列出容器可见的麦克风：

```sh
docker compose -f compose.yaml -f compose.linux.yaml run --rm translator list-mics
```

首次实时启动还会自动下载 SenseVoiceSmall 和 WtP。如果桌面使用纯 Wayland、声卡由特殊音频服务器独占，可能还需要宿主机侧的 XWayland 或音频转接配置。Docker Desktop（macOS/Windows）默认无法把宿主机麦克风直接映射给 Linux 容器，因此不能用本配置运行实时收音。

## 第 5 步：以后再次运行

不需要重新构建，在项目目录直接重复运行命令即可。Docker 命名卷 `translator-models` 会保留已下载模型。

查看命名卷：

```sh
docker volume ls
```

## 常见问题

- `docker: command not found`：Docker 尚未安装或终端没有重新打开。
- `Cannot connect to the Docker daemon`：Docker Desktop 尚未启动完成；打开 Docker Desktop，等待 Engine 运行后重试。
- 模型下载中断：直接重新执行原命令，下载器会尝试续传。
- `no space left on device`：清理 Docker Desktop 磁盘空间，建议至少留出 10 GB。
- Windows 提示 WSL 版本过旧：以管理员 PowerShell 执行 `wsl --update`，重启 Docker Desktop。

## 模型如何自动下载

实时模式执行 `realtime.py` 后会调用 `launcher.ensure_models()`，依次检查当前配置需要的三个模型：

1. 缺少 SenseVoiceSmall 时运行 `download_asr.py`。
2. 缺少 WtP 断句模型时运行 `download_sentence_model.py`。
3. 缺少所选翻译模型时运行 `download_model.py`。

每一项下载后都会再次检查关键配置和权重文件。Docker 命令行 `translate` 模式也会先检查并下载 Qwen3 4B 模型。

## 实现方法：三阶段本地流水线

```text
麦克风音频
    ↓
① SenseVoiceSmall：语音识别，得到并缓存原文
    ↓
② WtP：判断语义是否完整，在自然句界处分段
    ↓
③ Qwen3 4B：结合最近上下文翻译成中文并显示
```

### 1. 识别与原文输出

程序以 16 kHz 单声道采集音频，通过能量阈值、语音前置保留和静音时长形成语音片段。`SenseVoiceSmall` 把英语或日语语音识别成文字；识别结果先进入待处理缓冲区，而不是一有停顿就立刻翻译。

### 2. 完整性判断

`wtp-bert-mini` 检查缓冲区里的文本是否已经形成语义完整的句子。它会忽略不可靠的 ASR 标点，使用后文提示和日语悬空助词/连接词规则避免把半句话送去翻译；句子过长时才在较自然的位置拆分。

### 3. 本地 LLM 翻译

完整句子交给 `Qwen3-4B-Instruct-2507`，默认使用 4-bit 量化版本。翻译提示要求模型自动识别英语、日语或混合输入，只输出简明中文；最近几句原文与译文会作为上下文，改善代词和连续对话的一致性。全部推理都在本机进行，音频和识别文本不会发送给在线翻译 API。

### 默认使用的三个本地模型

| 阶段 | 模型 | 作用 | 大约大小 |
|---|---|---|---:|
| 语音识别 | SenseVoiceSmall | 英语/日语音频转文字 | 900 MB |
| 完整性判断 | wtp-bert-mini（ONNX） | 语义断句与自然边界选择 | 16 MB |
| LLM 翻译 | Qwen3-4B-Instruct-2507 4-bit | 英/日/混合文本翻译为中文 | 2.2–2.4 GB |

Apple Silicon 原生模式使用 MLX 4-bit 模型；Docker/Linux CPU 模式使用 GGUF Q4_K_M 模型。设置中还保留 Qwen2.5 1.5B 快速模式和 NLLB，但默认流程使用上表的 4B 翻译模型。
