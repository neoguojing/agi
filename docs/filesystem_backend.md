FilesystemBackend 为 Agent 提供完整的本地文件系统操作能力（读、写、搜索、上传、下载）。

| 接口       | 方法签名                                                                        | 功能说明           | 关键参数                                                                 | 返回值                          | 示例                                                                 |
| -------- | --------------------------------------------------------------------------- | -------------- | -------------------------------------------------------------------- | ---------------------------- | ------------------------------------------------------------------ |
| **初始化**  | `FilesystemBackend(root_dir=None, virtual_mode=False, max_file_size_mb=10)` | 创建文件系统 backend | `root_dir` 根目录<br>`virtual_mode` 虚拟路径模式<br>`max_file_size_mb` 最大搜索文件 | `FilesystemBackend`实例        | `python backend = FilesystemBackend("./repo", virtual_mode=True) ` |
| **列出目录** | `ls_info(path)`                                                             | 列出目录下文件（非递归）   | `path` 目录路径                                                          | `list[FileInfo]`             | `python backend.ls_info("/src") `                                  |
| **读取文件** | `read(file_path, offset=0, limit=2000)`                                     | 读取文件内容（带行号）    | `file_path` 文件路径<br>`offset` 起始行<br>`limit` 最大行数                     | `str`                        | `python backend.read("main.py",0,20) `                             |
| **创建文件** | `write(file_path, content)`                                                 | 创建新文件          | `file_path` 文件路径<br>`content` 文件内容                                   | `WriteResult`                | `python backend.write("hello.txt","hello world") `                 |
| **编辑文件** | `edit(file_path, old_string, new_string, replace_all=False)`                | 替换文件中的字符串      | `old_string` 旧文本<br>`new_string` 新文本<br>`replace_all` 是否替换全部         | `EditResult`                 | `python backend.edit("main.py","hello","hi",True) `                |
| **文本搜索** | `grep_raw(pattern, path=None, glob=None)`                                   | 搜索文件中的字符串      | `pattern` 搜索内容<br>`path` 搜索路径<br>`glob` 文件过滤                         | `list[GrepMatch]`            | `python backend.grep_raw("import","/src","*.py") `                 |
| **文件匹配** | `glob_info(pattern, path="/")`                                              | 使用 glob 匹配文件   | `pattern` glob规则                                                     | `list[FileInfo]`             | `python backend.glob_info("*.py","/src") `                         |
| **上传文件** | `upload_files(files)`                                                       | 批量上传文件         | `files = [(path, bytes)]`                                            | `list[FileUploadResponse]`   | `python backend.upload_files([("a.txt",b"hello")]) `               |
| **下载文件** | `download_files(paths)`                                                     | 批量下载文件         | `paths` 文件路径列表                                                       | `list[FileDownloadResponse]` | `python backend.download_files(["a.txt"]) `                        |

class FilesystemBackend(BackendProtocol):

    def __init__(self, root_dir=None, virtual_mode=None, max_file_size_mb=10):
        初始化文件系统后端并设置路径范围与大小限制

    def _resolve_path(self, key: str) -> Path:
        将输入路径解析为受规则约束的绝对路径

    def _to_virtual_path(self, path: Path) -> str:
        将真实路径转换为相对 root_dir 的虚拟路径

    def ls_info(self, path: str) -> list[FileInfo]:
        列出指定目录下的文件和子目录信息

    def read(self, file_path: str, offset: int = 0, limit: int = 2000) -> str:
        按行读取文件内容并返回带行号的文本

    def write(self, file_path: str, content: str) -> WriteResult:
        创建新文件并写入内容（不覆盖已有文件）

    def edit(self, file_path: str, old_string: str, new_string: str, replace_all: bool = False) -> EditResult:
        在文件中执行字符串替换操作

    def grep_raw(self, pattern: str, path: str | None = None, glob: str | None = None) -> list[GrepMatch] | str:
        在指定路径下搜索包含目标字符串的文本行

    def _ripgrep_search(self, pattern: str, base_full: Path, include_glob: str | None):
        使用 ripgrep 执行高性能文本搜索

    def _python_search(self, pattern: str, base_full: Path, include_glob: str | None):
        使用 Python 实现文本搜索作为 ripgrep 的降级方案

    def glob_info(self, pattern: str, path: str = "/") -> list[FileInfo]:
        根据 glob 模式递归匹配文件并返回信息

    def upload_files(self, files: list[tuple[str, bytes]]) -> list[FileUploadResponse]:
        批量写入多个文件到文件系统

    def download_files(self, paths: list[str]) -> list[FileDownloadResponse]:
        批量读取多个文件的二进制内容