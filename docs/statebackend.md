StateBackend 是存储在 Agent 内存 state 里的临时文件系统，而 FilesystemBackend 是操作真实操作系统磁盘的文件系统。

| 接口                                                           | 功能          | 主要参数                         | 返回                           | 说明                   |
| ------------------------------------------------------------ | ----------- | ---------------------------- | ---------------------------- | -------------------- |
| `__init__(runtime)`                                          | 初始化 backend | `runtime: ToolRuntime`       | None                         | 用于访问 `runtime.state` |
| `ls_info(path)`                                              | 列出目录文件      | `path: str`                  | `list[FileInfo]`             | 非递归列出目录文件和子目录        |
| `read(file_path, offset=0, limit=2000)`                      | 读取文件内容      | `file_path`                  | `str`                        | 返回带行号的文件内容           |
| `write(file_path, content)`                                  | 创建文件        | `file_path`, `content`       | `WriteResult`                | 文件存在则报错              |
| `edit(file_path, old_string, new_string, replace_all=False)` | 替换文件内容      | `old_string`, `new_string`   | `EditResult`                 | 返回替换次数               |
| `grep_raw(pattern, path=None, glob=None)`                    | 文本搜索        | `pattern`                    | `list[GrepMatch]`            | 在 state 文件中搜索字符串     |
| `glob_info(pattern, path="/")`                               | 按 glob 查找文件 | `pattern`                    | `list[FileInfo]`             | 返回匹配文件信息             |
| `upload_files(files)`                                        | 上传文件        | `files: list[(path, bytes)]` | `FileUploadResponse`         | ❌ 当前未实现              |
| `download_files(paths)`                                      | 下载文件        | `paths: list[str]`           | `list[FileDownloadResponse]` | 返回文件 bytes           |
