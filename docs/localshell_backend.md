LocalShellBackend = FilesystemBackend + 本地 shell 执行（无沙箱、完全宿主机权限）

| 分类   | 名称                        | 类型 / 签名                                                                                                 | 说明                                                                                               | 使用示例                                               |
| ---- | ------------------------- | ------------------------------------------------------------------------------------------------------- | ------------------------------------------------------------------------------------------------ | -------------------------------------------------- |
| 类    | `LocalShellBackend`       | `class LocalShellBackend(FilesystemBackend, SandboxBackendProtocol)`                                    | 在 `FilesystemBackend` 基础上扩展 **本地 Shell 命令执行能力**。提供文件系统操作 + 本地 shell 执行能力。**无沙箱、无隔离，直接在宿主机运行命令。** | `backend = LocalShellBackend(root_dir="/project")` |
| 常量   | `DEFAULT_EXECUTE_TIMEOUT` | `int = 120`                                                                                             | 默认 shell 命令执行超时时间（秒）                                                                             | 默认 2 分钟                                            |
| 构造函数 | `__init__`                | `(root_dir=None, virtual_mode=None, timeout=120, max_output_bytes=100000, env=None, inherit_env=False)` | 初始化 Backend，配置文件系统和 shell 执行环境                                                                   | `backend = LocalShellBackend(root_dir="/app")`     |
| 属性   | `id`                      | `property -> str`                                                                                       | 返回 backend 唯一 ID                                                                                 | `local-a3b9d1c2`                                   |
| 方法   | `execute`                 | `execute(command: str, timeout: int \| None = None) -> ExecuteResponse`                                 | 执行 shell 命令（宿主机）                                                                                 | `backend.execute("ls -la")`                        |


| 参数                 | 类型                    | 默认值      | 说明                    | 示例                        |
| ------------------ | --------------------- | -------- | --------------------- | ------------------------- |
| `root_dir`         | `str \| Path \| None` | 当前目录     | 文件系统操作与 shell 命令的工作目录 | `/home/user/project`      |
| `virtual_mode`     | `bool`                | `False`  | 是否启用虚拟路径模式（只影响文件系统操作） | `True`                    |
| `timeout`          | `int`                 | `120`    | 默认 shell 执行超时时间（秒）    | `timeout=300`             |
| `max_output_bytes` | `int`                 | `100000` | 命令输出最大字节数（超过会截断）      | `max_output_bytes=200000` |
| `env`              | `dict[str,str]`       | `{}`     | shell 执行环境变量          | `{"PATH":"/usr/bin"}`     |
| `inherit_env`      | `bool`                | `False`  | 是否继承当前进程环境变量          | `True`                    |
