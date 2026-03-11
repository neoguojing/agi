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
    def get_loader(cls, file_path: str, **kwargs):
        ext = file_path.split(".")[-1].lower()
        loader_cls = cls._MAPPING.get(ext, TextLoader)
        return loader_cls(file_path)