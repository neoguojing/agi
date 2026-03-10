
BackendProtocol 是 Agent 文件系统能力的统一抽象层，使不同存储（内存 / 本地 / 云 / sandbox）能够用同一套接口操作文件。

| 方法                                                           | 类型    | 参数                                       | 返回值                          | 说明               |
| ------------------------------------------------------------ | ----- | ---------------------------------------- | ---------------------------- | ---------------- |
| `ls_info(path)`                                              | sync  | `path:str`                               | `list[FileInfo]`             | 列出目录中的文件及元信息     |
| `als_info(path)`                                             | async | `path:str`                               | `list[FileInfo]`             | `ls_info` 异步版本   |
| `read(file_path, offset=0, limit=2000)`                      | sync  | `file_path:str` `offset:int` `limit:int` | `str`                        | 读取文件内容（带行号）      |
| `aread(...)`                                                 | async | 同上                                       | `str`                        | `read` 异步版本      |
| `grep_raw(pattern, path=None, glob=None)`                    | sync  | `pattern:str` `path:str?` `glob:str?`    | `list[GrepMatch] \| str`     | 在文件中搜索文本         |
| `agrep_raw(...)`                                             | async | 同上                                       | 同上                           | `grep_raw` 异步版本  |
| `glob_info(pattern, path="/")`                               | sync  | `pattern:str` `path:str`                 | `list[FileInfo]`             | 按 glob 模式查找文件    |
| `aglob_info(...)`                                            | async | 同上                                       | `list[FileInfo]`             | `glob_info` 异步版本 |
| `write(file_path, content)`                                  | sync  | `file_path:str` `content:str`            | `WriteResult`                | 写入新文件（文件存在则失败）   |
| `awrite(...)`                                                | async | 同上                                       | `WriteResult`                | `write` 异步版本     |
| `edit(file_path, old_string, new_string, replace_all=False)` | sync  | 见参数                                      | `EditResult`                 | 字符串替换编辑文件        |
| `aedit(...)`                                                 | async | 同上                                       | `EditResult`                 | `edit` 异步版本      |
| `upload_files(files)`                                        | sync  | `list[(path, bytes)]`                    | `list[FileUploadResponse]`   | 批量上传文件           |
| `aupload_files(files)`                                       | async | 同上                                       | 同上                           | 异步版本             |
| `download_files(paths)`                                      | sync  | `list[str]`                              | `list[FileDownloadResponse]` | 批量下载文件           |
| `adownload_files(paths)`                                     | async | 同上                                       | 同上                           | 异步版本             |


| 数据结构                   | 字段           | 类型                        | 说明                      |
| ---------------------- | ------------ | ------------------------- | ----------------------- |
| `FileDownloadResponse` | path         | str                       | 请求下载的文件路径               |
|                        | content      | bytes | None              | 文件内容                    |
|                        | error        | FileOperationError | None | 错误码                     |
| `FileUploadResponse`   | path         | str                       | 上传文件路径                  |
|                        | error        | FileOperationError | None | 错误码                     |
| `WriteResult`          | error        | str | None                | 错误信息                    |
|                        | path         | str | None                | 写入文件路径                  |
|                        | files_update | dict | None               | LangGraph checkpoint 更新 |
| `EditResult`           | error        | str | None                | 错误信息                    |
|                        | path         | str | None                | 编辑文件路径                  |
|                        | files_update | dict | None               | checkpoint 更新           |
|                        | occurrences  | int | None                | 替换次数                    |


SandboxBackendProtocol 在 BackendProtocol 基础上增加 代码执行能力。

| 方法                               | 类型       | 参数                           | 返回值               | 说明             |
| -------------------------------- | -------- | ---------------------------- | ----------------- | -------------- |
| `id`                             | property | -                            | str               | sandbox 实例唯一ID |
| `execute(command, timeout=None)` | sync     | `command:str` `timeout:int?` | `ExecuteResponse` | 执行 shell 命令    |
| `aexecute(...)`                  | async    | 同上                           | `ExecuteResponse` | 异步版本           |


BackendFactory = 一个用 runtime 创建 backend 的函数。
create_agent
     │
     │ backend_factory
     ▼
FilesystemMiddleware
     │
     │ runtime 创建
     ▼
backend_factory(runtime)
     │
     ▼
FilesystemBackend
     │
     ▼
Agent tools
     │
     ▼
read / write / grep


def backend_factory(runtime: ToolRuntime):
    workspace = runtime.state.get("workspace", "/tmp")
    return FilesystemBackend(root=workspace)

agent = create_agent(
    model,
    middleware=[FilesystemMiddleware(backend=backend_factory)]
)


def backend_factory(runtime: ToolRuntime):
    thread_id = runtime.thread_id
    return FilesystemBackend(
        root=f"/sandbox/{thread_id}"
    )