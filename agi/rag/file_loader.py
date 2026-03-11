from langchain_community.document_loaders import (
    PyPDFLoader, CSVLoader, Docx2txtLoader, TextLoader, 
    UnstructuredMarkdownLoader, JSONLoader, YoutubeLoader
)

class LoaderFactory:
    _MAPPING = {
        "pdf": PyPDFLoader,
        "csv": CSVLoader,
        "docx": Docx2txtLoader,
        "doc": Docx2txtLoader,
        "md": UnstructuredMarkdownLoader,
        "json": JSONLoader,
        "txt": TextLoader
    }

    @classmethod
    def get_loader(cls, file_path: str, source_type: SourceType, **kwargs):
        if source_type == SourceType.YOUTUBE:
            return YoutubeLoader.from_youtube_url(file_path, add_video_info=True)
        if source_type == SourceType.WEB:
            return SafeWebBaseLoader(file_path) # 自定义包装类
        
        ext = file_path.split(".")[-1].lower()
        loader_cls = cls._MAPPING.get(ext, TextLoader)
        return loader_cls(file_path)