BaseSandbox 把大多数复杂文件操作转换成一段 python3 脚本，然后通过 execute() 在 sandbox 里运行。
BaseSandbox 的核心思想：把所有文件操作转换为一条 shell 命令（多数是 python3 脚本），再通过 execute() 在 sandbox 环境中执行。

SandboxBackendProtocol
        ▲
        │
    BaseSandbox (abstract)
        ▲
        │
  YourSandboxImplementation


  | 方法                                                           | 类型       | 参数                                                                                | 返回类型                         | 功能说明                    | 实现方式                   |
| ------------------------------------------------------------ | -------- | --------------------------------------------------------------------------------- | ---------------------------- | ----------------------- | ---------------------- |
| `execute(command, timeout=None)`                             | **抽象**   | `command: str`<br>`timeout: int \| None`                                          | `ExecuteResponse`            | 在 sandbox 内执行 shell 命令  | 由子类实现                  |
| `id`                                                         | **抽象属性** | -                                                                                 | `str`                        | sandbox 唯一标识            | 由子类实现                  |
| `upload_files(files)`                                        | **抽象**   | `files: list[(path, bytes)]`                                                      | `list[FileUploadResponse]`   | 上传多个文件到 sandbox，需支持部分成功 | 由子类实现                  |
| `download_files(paths)`                                      | **抽象**   | `paths: list[str]`                                                                | `list[FileDownloadResponse]` | 下载多个文件，需支持部分成功          | 由子类实现                  |
| `ls_info(path)`                                              | 已实现      | `path: str`                                                                       | `list[FileInfo]`             | 获取目录文件列表及是否为目录          | `python3 + os.scandir` |
| `read(file_path, offset=0, limit=2000)`                      | 已实现      | `file_path: str`<br>`offset: int`<br>`limit: int`                                 | `str`                        | 读取文件内容并带行号              | shell + python3        |
| `write(file_path, content)`                                  | 已实现      | `file_path: str`<br>`content: str`                                                | `WriteResult`                | 创建新文件                   | heredoc + python       |
| `edit(file_path, old_string, new_string, replace_all=False)` | 已实现      | `file_path: str`<br>`old_string: str`<br>`new_string: str`<br>`replace_all: bool` | `EditResult`                 | 替换文件字符串                 | python脚本执行             |
| `grep_raw(pattern, path=None, glob=None)`                    | 已实现      | `pattern: str`<br>`path: str \| None`<br>`glob: str \| None`                      | `list[GrepMatch] \| str`     | 搜索文件内容                  | `grep -rHnF`           |
| `glob_info(pattern, path="/")`                               | 已实现      | `pattern: str`<br>`path: str`                                                     | `list[FileInfo]`             | glob 文件匹配               | python `glob.glob`     |


import subprocess
from deepagents.backends.sandbox.base import BaseSandbox
from deepagents.backends.protocol import ExecuteResponse, FileUploadResponse, FileDownloadResponse


class DockerSandbox(BaseSandbox):

    def __init__(self, container_name: str):
        self.container_name = container_name

    @property
    def id(self) -> str:
        return f"docker:{self.container_name}"

    def execute(self, command: str, *, timeout: int | None = None) -> ExecuteResponse:
        result = subprocess.run(
            [
                "docker",
                "exec",
                self.container_name,
                "bash",
                "-lc",
                command,
            ],
            capture_output=True,
            text=True,
            timeout=timeout,
        )

        return ExecuteResponse(
            output=result.stdout + result.stderr,
            exit_code=result.returncode,
            truncated=False,
        )

    def upload_files(self, files):
        responses = []
        for path, data in files:
            try:
                subprocess.run(
                    ["docker", "cp", "-", f"{self.container_name}:{path}"],
                    input=data,
                    check=True,
                )
                responses.append(FileUploadResponse(path=path))
            except Exception as e:
                responses.append(FileUploadResponse(path=path, error=str(e)))
        return responses

    def download_files(self, paths):
        responses = []
        for path in paths:
            try:
                result = subprocess.run(
                    ["docker", "cp", f"{self.container_name}:{path}", "-"],
                    capture_output=True,
                )
                responses.append(FileDownloadResponse(path=path, content=result.stdout))
            except Exception as e:
                responses.append(FileDownloadResponse(path=path, error=str(e)))
        return responses