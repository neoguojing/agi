StoreBackend = 基于 LangGraph BaseStore 的持久化、跨会话虚拟文件系统。

| 分类          | 名称                               | 类型 / 参数                            | 返回                       | 说明                                |
| ----------- | -------------------------------- | ---------------------------------- | ------------------------ | --------------------------------- |
| 基本信息        | 组件                               | `StoreBackend`                     | -                        | DeepAgents 文件存储 Backend           |
| 基本信息        | 继承                               | `BackendProtocol`                  | -                        | 实现统一文件操作接口                        |
| 基本信息        | 存储介质                             | `LangGraph BaseStore`              | -                        | 使用 LangGraph Store 持久化            |
| 基本信息        | 存储类型                             | 持久化                                | -                        | 数据可跨会话、跨线程                        |
| 基本信息        | 文件系统                             | 虚拟文件系统                             | -                        | 不依赖真实磁盘                           |
| 基本信息        | 多租户隔离                            | `namespace`                        | tuple[str,...]           | 通过 namespace 隔离用户 / agent         |
| 基本信息        | Async支持                          | 是                                  | -                        | 提供 async API                      |
| 核心概念        | Namespace                        | `tuple[str,...]`                   | -                        | Store 存储路径前缀                      |
| 核心概念        | NamespaceFactory                 | `Callable(ctx)`                    | tuple                    | 根据 runtime 动态生成 namespace         |
| 核心概念        | BackendContext                   | `{state, runtime}`                 | -                        | namespace 生成上下文                   |
| 内部数据        | FileData                         | `{content,created_at,modified_at}` | dict                     | Store 内部文件格式                      |
| 内部数据        | content                          | `list[str]`                        | -                        | 文件按行存储                            |
| 内部数据        | created_at                       | `str`                              | -                        | 创建时间                              |
| 内部数据        | modified_at                      | `str`                              | -                        | 修改时间                              |
| 文件操作        | `ls_info(path)`                  | path:str                           | `list[FileInfo]`         | 列出目录文件（非递归）                       |
| 文件操作        | `read(file_path, offset, limit)` | path:str                           | str                      | 读取文件内容                            |
| 文件操作        | `write(file_path, content)`      | path:str                           | `WriteResult`            | 创建新文件                             |
| 文件操作        | `edit(file_path, old, new)`      | path:str                           | `EditResult`             | 替换字符串                             |
| 查询          | `grep_raw(pattern, path, glob)`  | pattern:str                        | `list[GrepMatch]`        | 文本搜索                              |
| 查询          | `glob_info(pattern, path)`       | glob pattern                       | `list[FileInfo]`         | 文件 glob 搜索                        |
| 批量操作        | `upload_files(files)`            | `[(path, bytes)]`                  | `FileUploadResponse[]`   | 批量上传文件                            |
| 批量操作        | `download_files(paths)`          | `list[str]`                        | `FileDownloadResponse[]` | 批量下载文件                            |
| Async API   | `aread`                          | 同 read                             | str                      | 异步读取                              |
| Async API   | `awrite`                         | 同 write                            | WriteResult              | 异步写文件                             |
| Async API   | `aedit`                          | 同 edit                             | EditResult               | 异步编辑                              |
| 运行依赖        | runtime.store                    | `BaseStore`                        | -                        | LangGraph 提供的存储                   |
| Namespace策略 | 自定义                              | `namespace=lambda ctx: (...)`      | -                        | 推荐方式                              |
| Namespace策略 | legacy                           | 自动读取 `assistant_id`                | -                        | 未来废弃                              |
| 文件读取特点      | 行号                               | 自动添加                               | -                        | read 返回带行号                        |
| 文件系统特点      | 目录结构                             | 通过 path 模拟                         | -                        | Store 不是真目录                       |
| 搜索实现        | grep                             | 遍历 store item                      | -                        | 内存匹配                              |
| 搜索实现        | glob                             | path pattern                       | -                        | 模拟文件系统 glob                       |
| 数据分页        | `_search_store_paginated`        | page_size                          | list[Item]               | 自动分页查询 store                      |
| 典型用途        | Agent共享文件                        | -                                  | -                        | 多 agent 使用                        |
| 典型用途        | 跨会话记忆                            | -                                  | -                        | 文件长期保存                            |
| 典型用途        | LLM工具文件                          | -                                  | -                        | agent 工具读写                        |
| 一句话总结       | StoreBackend                     | -                                  | -                        | **基于 LangGraph Store 的持久化虚拟文件系统** |
