# 实时英日翻译

这是一个完全本地运行的英语/日语实时语音转中文工具。

处理流程：

```text
SenseVoiceSmall 语音识别
          ↓
WtP 判断句子是否完整并自然断句
          ↓
Qwen3 4B 本地翻译成中文
```

## 第一次运行

本项目统一使用 Docker 交付，不使用 DMG，也不要求朋友预先安装 Python。第一次需要完成以下步骤：

1. 安装并启动 Docker。
2. 解压或克隆这个代码目录。
3. 在代码目录执行 `docker compose build`。
4. 执行项目运行命令；程序会自动下载当前模式需要的本地模型。

首次执行命令行翻译会自动下载 Qwen3 4B；首次启动实时模式会自动下载 SenseVoiceSmall、WtP 和 Qwen3 4B 三个本地模型。首次下载时请联网并预留数 GB 空间，下载完成后可以离线运行。

从一台没有安装过 Docker 的电脑开始，请按 [Docker 使用说明](Docker使用说明.md) 逐步操作。

> 注意：macOS/Windows 的 Docker Desktop 适合运行命令行翻译，但默认无法把宿主机麦克风直接交给 Linux 容器。带麦克风和 Tk 窗口的实时模式仅提供 Linux Docker 运行配置。

日常界面参数说明见 [设置说明](设置说明.md)。
# translation-
