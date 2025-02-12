"""A module for managing vector Storage and retrieval."""

import os
from pathlib import Path
from typing import Dict, List, Optional, Sequence, Union

import pandas as pd
from llama_index.core import SimpleDirectoryReader, StorageContext
from llama_index.core.extractors import (
    KeywordExtractor,
    QuestionsAnsweredExtractor,
    SummaryExtractor,
    TitleExtractor,
)
from llama_index.core.ingestion import IngestionPipeline
from llama_index.core.node_parser import SentenceSplitter, TokenTextSplitter
from llama_index.core.schema import BaseNode, Document, TextNode
from llama_index.core.storage.docstore import BaseDocumentStore, SimpleDocumentStore
from llama_index.core.storage.docstore.types import RefDocInfo
from llama_index.core.storage.index_store import SimpleIndexStore
from llama_index.core.storage.index_store.types import BaseIndexStore
from llama_index.core.vector_stores import SimpleVectorStore
from pandas import DataFrame

from llama_utils.utils.errors import StorageNotFoundError
from llama_utils.utils.helper_functions import generate_content_hash

EXTRACTORS = dict(
    text_splitter=TokenTextSplitter,
    title=TitleExtractor,
    question_answer=QuestionsAnsweredExtractor,
    summary=SummaryExtractor,
    keyword=KeywordExtractor,
)
ID_MAPPING_FILE = "metadata_index.csv"


class Storage:
    """A class to manage vector Storage and retrieval.

    The Storage class is used to manage the storage and retrieval of documents. It provides methods to add documents to the
    store, read documents from a directory, and extract information from the documents.
    """

    def __init__(
        self,
        storage_context: StorageContext = None,
    ):
        """Initialize the Storage.

        The constructor method takes a llama_index.core.StorageContext object that is a native llamaIndex object
        and a metadata table (pandas.DataFrame-optional) as input.

        Parameters
        ----------
        storage_context: str, optional, default is None.
            the StorageContext object that is created by LlamaIndex (a native llamaIndex object).

        """
        if not isinstance(storage_context, StorageContext):
            raise ValueError(
                f"Storage class should be instantiated using StorageContext object, given: {storage_context}"
            )

        self._store = storage_context

    @classmethod
    def create(cls) -> "Storage":
        """Create a new in-memory Storage.

        Returns
        -------
        Storage:
            The storage Context.

        Examples
        --------
        You can create a new storage (in-memory) using the `create` method as follows:

        ```python
        >>> store = Storage.create()
        >>> print(store)
        <BLANKLINE>
                Documents: 0
                Indexes: 0
        <BLANKLINE>

        ```
        """
        storage = cls._create_simple_storage_context()
        return cls(storage)

    @staticmethod
    def _create_simple_storage_context() -> StorageContext:
        """Create a simple Storage context.

        Returns
        -------
        StorageContext:
            A storage context with docstore, vectore store, and index store.
        """
        return StorageContext.from_defaults(
            docstore=SimpleDocumentStore(),
            vector_store=SimpleVectorStore(),
            index_store=SimpleIndexStore(),
        )

    @staticmethod
    def _create_metadata_index():
        """Create a metadata-based index."""
        return pd.DataFrame(columns=["file_name", "doc_id"])

    @property
    def store(self) -> StorageContext:
        """Get the Storage context."""
        return self._store

    @property
    def docstore(self) -> BaseDocumentStore:
        """Get the document store."""
        return self.store.docstore

    @property
    def vector_store(self):
        """Get the vector store."""
        return self.store.vector_store

    @property
    def index_store(self) -> BaseIndexStore:
        """Get the index store."""
        return self.store.index_store

    def save(self, store_dir: str):
        """Save the storage to a directory.

        Parameters
        ----------
        store_dir: str
            The directory to save the store.

        Examples
        --------
        You can save a storage to a directory as follows:
        ```python
        >>> store = Storage.create()
        >>> store.save("examples/paul-graham-essay-storage-example")

        ```
        The following files will be created in the specified directory:
        - metadata_index.csv
        - docstore.json
        - default__vector_store.json
        - index_store.json
        - graph_store.json
        - image__vector_store.json
        """
        self.store.persist(persist_dir=store_dir)
        file_path = os.path.join(store_dir, ID_MAPPING_FILE)
        save_metadata_index(self.document_metadata(as_dataframe=True), file_path)

    @classmethod
    def load(cls, store_dir: str) -> "Storage":
        """Load the store from a directory.

        Parameters
        ----------
        store_dir: str
            The directory containing the store.

        Returns
        -------
        Storage:
            The loaded storage.

        Raises
        ------
        StorageNotFoundError
            If the storage is not found at the specified directory.

        Examples
        --------
        You can load a storage from a directory as follows:
        ```python
        >>> store = Storage.load("examples/paul-graham-essay-storage")
        >>> print(store) # doctest: +SKIP
        <BLANKLINE>
                    Documents: 53
                    Indexes: 2
        <BLANKLINE>
        >>> metadata = store.document_metadata(as_dataframe=True)
        >>> print(metadata.head()) # doctest: +SKIP
                                     doc_id                              node_id              file_name
        0   a25111e2e59f81bb7a0e3efb4825...  cadde590b82362fc7a5f8ce0751c5b30b...  paul_graham_essay.txt
        1   a25111e2e59f81bb7a0e3efb4825...  0567f3a9756983e1d040ec332255db945...  paul_graham_essay.txt
        2   a25111e2e59f81bb7a0e3efb4825...  d5542515414f1bf30f6c21f0796af8bde...  paul_graham_essay.txt
        3   a25111e2e59f81bb7a0e3efb4825...  120b69658a6c69ab8de3167b5ed0db779...  paul_graham_essay.txt
        >>> docstore = store.docstore # doctest: +SKIP
        <llama_index.core.storage.docstore.simple_docstore.SimpleDocumentStore at 0x20444d31be0>
        >>> vector_store = store.vector_store
        >>> print(type(vector_store))
        <class 'llama_index.core.vector_stores.simple.SimpleVectorStore'>

        ```
        """
        if not Path(store_dir).exists():
            raise StorageNotFoundError(f"Storage not found at {store_dir}")
        storage = StorageContext.from_defaults(persist_dir=store_dir)
        return cls(storage)

    def __str__(self):
        """Return a string representation of the storage."""
        message = f"""
        Documents: {len(self.docstore.docs)}
        Indexes: {len(self.index_store.index_structs())}
        """
        return message

    def __repr__(self):
        """Return a string representation of the storage."""
        message = f"""
        Documents: {len(self.docstore.docs)}
        Indexes: {len(self.index_store.index_structs())}
        """
        return message

    @property
    def node_metadata(self) -> pd.DataFrame:
        """Get the metadata index."""
        return create_metadata_index_existing_docs(self.docstore.docs)

    def document_metadata(
        self, as_dataframe: Optional[bool] = False
    ) -> Union[Dict[str, RefDocInfo], DataFrame]:
        r"""Document metadata.

        Get the metadata of all the documents in the docstore.

        Parameters
        ----------
        as_dataframe: bool, optional, default is False.
            True to return the metadata as a DataFrame.

        Returns
        -------
        Dict[str, RefDocInfo] or DataFrame
            The metadata of all the documents in the docstore.

        Examples
        --------
        You can get the document metadata as a dictionary using the `metadata` method with the default parameter values:
        ```python
        >>> store = Storage.load("examples/paul-graham-essay-storage")
        >>> metadata = store.document_metadata()

        ```
        The `metadata` is a dictionary with the document ID as the key and the document metadata as the value:
        ```python
        >>> documents_id = list(metadata.keys())
        >>> print(documents_id) # doctest: +SKIP
        ['a25111e2e59f81bb7a0e3efb48255f4a5d4f722aaf13ffd112463fb98c227092']
        >>> print(metadata) # doctest: +SKIP
        {
            'a25111e2e59f81bb7a0e3efb48255f4a5d4f722aaf13ffd112463fb98c227092':
                RefDocInfo(
                    node_ids=[
                        'cadde590b82362fc7a5f8ce0751c5b30b11c0f81369df7d83a76956bf22765b7',
                        '0567f3a9756983e1d040ec332255db94521ed5dc1b03fc7312f653c0e670a0bf',
                        'd5542515414f1bf30f6c21f0796af8bde4c513f2e72a2df21f0810f10826252f',
                        '120b69658a6c69ab8de3167b5ed0db77941a2b487e94d5d0e64a0d2d2805a4b7'
                    ],
                    metadata={
                        'file_path': 'examples\\data\\paul_graham_essay.txt',
                        'file_name': 'paul_graham_essay.txt',
                        'file_type': 'text/plain',
                        'file_size': 75395,
                        'creation_date': '2024-10-24',
                        'last_modified_date': '2024-09-16',
                        'document_title': 'Based on the candidate titles and content, I would suggest a***.'
                    }
                )
            }
        ```
        To get the metadata as a DataFrame, you can set the `as_dataframe` parameter to True:
        ```python
        >>> metadata = store.document_metadata(as_dataframe=True)
        >>> print(metadata) # doctest: +SKIP
                                                       doc_id                                            node_id
        0   a25111e2e59f81bb7a0e3efb48255f4a5d4f722aaf13ff...  cadde590b82362fc7a5f8ce0751c5b30b11c0f81369df7...
        1   a25111e2e59f81bb7a0e3efb48255f4a5d4f722aaf13ff...  0567f3a9756983e1d040ec332255db94521ed5dc1b03fc...
        2   a25111e2e59f81bb7a0e3efb48255f4a5d4f722aaf13ff...  d5542515414f1bf30f6c21f0796af8bde4c513f2e72a2d...
        ```
        """
        ref_doc_info: dict = self.docstore.get_all_ref_doc_info()
        if as_dataframe:
            doct_node_ids_dict = {}
            file_name_doc = {}
            doc_ids = list(ref_doc_info.keys())
            for doc_id in doc_ids:
                # get the DocRefInfo object for the first document
                doc_ref = ref_doc_info[doc_id].to_dict()
                # get the node ids for the first document
                node_ids = doc_ref["node_ids"]
                file_name_doc[doc_id] = doc_ref["metadata"].get("file_name")
                doct_node_ids_dict[doc_id] = node_ids

            df = pd.DataFrame(
                list(doct_node_ids_dict.items()), columns=["doc_id", "node_id"]
            )
            df = df.explode("node_id", ignore_index=True)

            # merge the file name with the doc_id
            file_name_df = pd.DataFrame(
                file_name_doc.items(), columns=["doc_id", "file_name"]
            )
            df = df.merge(file_name_df, on="doc_id", how="left", validate="many_to_one")
            data = df
        else:
            data = ref_doc_info
        return data

    def node_id_list(self) -> List[str]:
        """Get the metadata of the nodes in the docstore.

        Returns
        -------
        Dict[str, Dict[str, Any]]
            The metadata of the nodes in the docstore.

        Examples
        --------
        You can get the metadata of the nodes in the docstore using the `nodes_metadata` method:
        ```python
        >>> store = Storage.load("examples/paul-graham-essay-storage")
        >>> nodes_metadata = store.node_id_list()
        >>> print(nodes_metadata) # doctest: +SKIP
        [
            'cadde590b82362fc7a5f8ce0751c5b30b11c0f81369df7d83a76956bf22765b7',
            '0567f3a9756983e1d040ec332255db94521ed5dc1b03fc7312f653c0e670a0bf',
            'd5542515414f1bf30f6c21f0796af8bde4c513f2e72a2df21f0810f10826252f',
            ...
        ]
        ```
        """
        return list(self.docstore.docs.keys())

    def delete_document(self, doc_id: str):
        """Delete a document from the docstore.

        Parameters
        ----------
        doc_id: str
            The ID of the document to delete.

        Returns
        -------
        None

        Examples
        --------
        You can delete a document from the document store and all the nodes that are related to it using the
        `delete_document` method by providing the `document_id`:

        ```python
        >>> store = Storage.load("examples/paul-graham-essay-storage")
        >>> document_metadata = store.document_metadata
        >>> document_id = list(document_metadata().keys())[0]
        >>> print(document_id) # doctest: +SKIP
        a25111e2e59f81bb7a0e3efb48255f4a5d4f722aaf13ffd112463fb98c227092
        >>> store.delete_document(document_id)

        ```

        Now if you check the document_metadata, you will find that the document is deleted:

        ```python
        >>> print(store.document_metadata())
        {}

        ```
        """
        if doc_id not in self.document_metadata().keys():
            raise ValueError(f"Document with ID {doc_id} not found.")
        self.docstore.delete_ref_doc(doc_id)

    def delete_node(self, node_id: str):
        """Delete a node from the docstore.

        Parameters
        ----------
        node_id: str
            The ID of the node to delete.

        Returns
        -------
        None

        Examples
        --------
        You can delete a node from the document store using the `delete_node` method by providing the `node_id`:
        ```
        >>> store = Storage.load("examples/paul-graham-essay-storage")
        >>> node_id = store.node_id_list()[0]
        >>> print(node_id)
        cadde590b82362fc7a5f8ce0751c5b30b11c0f81369df7d83a76956bf22765b7
        >>> store.delete_node(node_id)

        ```
        """
        self.docstore.delete_document(node_id)

    def add_documents(
        self,
        docs: Sequence[Union[Document, TextNode]],
        generate_id: bool = True,
        update: bool = False,
    ):
        r"""Add node/documents to the store.

        The `add_documents` method adds a node to the store. The node's id is a sha256 hash generated based on the
        node's text content. if the `update` parameter is True and the nodes already exist the existing node will
        be updated.

        Parameters
        ----------
        docs: Sequence[TextNode/Document]
            The node/documents to add to the store.
        generate_id: bool, optional, default is False.
            True if you want to generate a sha256 hash number as a doc_id based on the content of the nodes.
        update: bool, optional, default is True.
            True to update the document in the docstore if it already exist.

        Returns
        -------
        None

        Examples
        --------
        - First create the storage object:
        ```python
        >>> store = Storage.create()

        - Then you can add documents to the store using the `add_documents` method:

        >>> data_path = "examples/data/essay"
        >>> documents = Storage.read_documents(data_path)
        >>> store.add_documents(documents)
        >>> print(store)
        <BLANKLINE>
                Documents: 1
                Indexes: 0
        <BLANKLINE>

        - once the documents are added successfully, they are added also to the metadata index.

        >>> metadata = store.node_metadata
        >>> print(metadata) # doctest: +SKIP
                        file_name                                            node_id
        0   paul_graham_essay.txt  cadde590b82362fc7a5f8ce0751c5b30b11c0f81369df7...

        >>> docstore = store.docstore
        >>> print(docstore.docs) # doctest: +SKIP

        {
            'a25111e2e59f81bb7a0e3efb48255f4a5d4f722aaf13ffd112463fb98c227092':
                Document(
                    id_='a25111e2e59f81bb7a0e3efb48255f4a5d4f722aaf13ffd112463fb98c227092',
                    embedding=None,
                    metadata={
                        'file_path': 'examples\\data\\essay\\paul-graham-essay.txt',
                        'file_name': 'paul-graham-essay.txt',
                        'file_type': 'text/plain',
                        'file_size': 75395,
                        'creation_date': '2024-10-25',
                        'last_modified_date': '2024-09-16'
                    },
                    excluded_embed_metadata_keys=['file_name'],
                    excluded_llm_metadata_keys=['file_name'],
                    relationships={},
                    text='What I Worked On February 2021 Before college the two ...',
                    mimetype='text/plain',
                    start_char_idx=None,
                    end_char_idx=None,
                    text_template='{metadata_str}\n\n{content}',
                    metadata_template='{key}: {value}',
                    metadata_seperator='\n'
                )
        }
        ```
        """
        for doc in docs:
            # change the id to a sha256 hash if it is not already
            if generate_id:
                doc.node_id = generate_content_hash(doc.text)

            if not self.docstore.document_exists(doc.node_id) or update:
                self.docstore.add_documents([doc], allow_update=update)
            else:
                print(f"Document with ID {doc.node_id} already exists. Skipping.")

    @staticmethod
    def read_documents(
        path: Union[str, Path],
        split_into_nodes: bool = False,
        chunk_size: int = 1024,
        chunk_overlap: int = 100,
        show_progress: bool = False,
        num_workers: int = None,
        recursive: bool = False,
        **kwargs,
    ) -> List[Union[Document, TextNode]]:
        r"""Read documents from a directory and optionally splits them into TextNodes.

        the `read_documents` method reads documents from a directory and returns a list of documents.
        the `doc_id` is sha256 hash number generated based on the document's text content.

        Parameters
        ----------
        path: str
            path to the directory containing the documents. or a single file path.
        split_into_nodes : bool, optional
            If True, splits documents into smaller TextNodes (default: False).
        chunk_size : int, optional, default is 1024
            The size of each text chunk. The `chunk_size` Defines the size of each split chunk (in characters).
            - Larger values preserve more context but increase memory usage.
            - Smaller values improve retrieval specificity but may lose context.
        chunk_overlap : int, optional, default is 100.
            The overlap size between chunks. The `chunk_overlap` ensures adjacent chunks have overlapping content to
            preserve context.
            - Higher overlap improves retrieval coherence but increases redundancy.
            - Lower overlap minimizes redundancy but might break context.
        show_progress: bool, optional, default is False.
            True to show progress bar.
        num_workers: int, optional, default is None.
            The number of workers to use for loading the data.
        recursive: bool, optional, default is False.
            True to read from subdirectories.

        Returns
        -------
        Sequence[Union[Document, TextNode]]
            The documents/nodes read from the store.

        Raises
        ------
        FileNotFoundError
            If the directory is not found.

        Notes
        -----
        The best values for `chunk_size` and `chunk_overlap` depend on the type of documents and retrieval needs.

        | **Use Case**                                              | **Recommended `chunk_size`** | **Recommended `chunk_overlap`** |
        |-----------------------------------------------------------|------------------------------|---------------------------------|
        | **Short documents (FAQs, structured text, emails)**       | `256 - 512`                  | `50`                           |
        | **General-purpose (news articles, reports, blog posts)**  | `512 - 1024`                 | `50 - 100`                     |
        | **Long documents (PDFs, research papers, books)**         | `1024 - 2048`                | `100 - 200`                    |
        | **Dense technical/scientific papers**                     | `1500 - 2500`                | `200 - 300`                    |
        | **Conversational AI (chat context memory)**               | `512 - 1024`                 | `50 - 150`                     |
        | **Legal/contract documents (precise context retention)**  | `2048 - 4096`                | `200 - 400`                    |

        Best Practices:
        - For most cases, `chunk_size=1024` and `chunk_overlap=100` work well.
        - If working with FAQs, decrease `chunk_size` to `512`.
        - If preserving long-form LLM context, use `2048+` with higher overlap.

        Examples
        --------
        You can read documents from a directory as follows:
        ```python
        >>> from llama_utils.retrieval.storage import Storage
        >>> data_path = "examples/data/essay"
        >>> docs = Storage.read_documents(data_path)
        >>> print(docs) # doctest: +SKIP
        [
            Document(
                id_='a25111e2e59f81bb7a0e3efb48255**',
                embedding=None,
                metadata={
                    'file_path': 'examples/data/essay/paul-graham-essay.txt',
                    'file_name': 'paul-graham-essay.txt',
                    'file_type': 'text/plain',
                    'file_size': 75395,
                    'creation_date': '2024-10-25',
                    'last_modified_date': '2024-09-16'
                },
                excluded_embed_metadata_keys=['file_name'],
                excluded_llm_metadata_keys=['file_name'],
                relationships={},
                text='What I Worked On\n\nFebruary 2021\n\nBefore college the two main things ****',
                mimetype='text/plain',
                start_char_idx=None,
                end_char_idx=None,
        4        text_template='{metadata_str}\n\n{content}',
                metadata_template='{key}: {value}',
                metadata_seperator='\n'
            )
        ]
        ```

        You can also use the `split_into_nodes` parameter to split the documents into smaller nodes:

        ```python
        >>> docs = Storage.read_documents(data_path, split_into_nodes=True)
        >>> print(len(docs))
        20
        >>> print(docs[0]) # doctest: +SKIP
        Node ID: bfdf48ff-3c2b-49a8-ae25-7a837158d2dc
        Text: What I Worked On    February 2021    Before college the two main
        things I worked on, outside of school, were writing and programming. I
        didn't write essays. I wrote what beginning writers were supposed to
        write then, and probably still are: short stories. My stories were
        awful. They had hardly any plot, just characters with strong feelings,
        whic...
        ```
        """
        if isinstance(path, str):
            path = Path(path)

        if not path.exists():
            raise FileNotFoundError(f"Directory not found: {path}")

        if path.is_file():
            reader = SimpleDirectoryReader(input_files=[path])
        else:
            reader = SimpleDirectoryReader(path, recursive=recursive, **kwargs)

        documents = reader.load_data(
            show_progress=show_progress, num_workers=num_workers, **kwargs
        )

        if split_into_nodes:
            # Split documents into nodes.
            splitter = SentenceSplitter(
                chunk_size=chunk_size, chunk_overlap=chunk_overlap
            )
            documents = splitter.get_nodes_from_documents(documents)

        for doc in documents:
            # exclude the file name from the llm metadata to avoid affecting the llm by weird file names
            doc.excluded_llm_metadata_keys = ["file_name"]
            # exclude the file name from the embeddings' metadata to avoid affecting the llm by weird file names
            doc.excluded_embed_metadata_keys = ["file_name"]
            # Generate a hash based on the document's text content
            content_hash = generate_content_hash(doc.text)
            # Assign the hash as the doc_id
            doc.metadata["content-hash"] = content_hash

        return documents

    def get_nodes_by_file_name(
        self, file_name: str, exact_match: bool = False
    ) -> List[BaseNode]:
        r"""Get nodes by file name.

        Parameters
        ----------
        file_name: str
            The file name to search for.
        exact_match: bool, optional, default is False
            True to search for an exact match, False to search for a partial match.

        Returns
        -------
        List[TextNode]
            The nodes with the specified file name.

        Examples
        --------
        - First read the storage context from a directory:
        ```python
        >>> storage_dir = "examples/paul-graham-essay-storage"
        >>> store = Storage.load(storage_dir)
        >>> print(store)
        <BLANKLINE>
                Documents: 53
                Indexes: 2
        <BLANKLINE>

        - The storage context contains the following data:

        >>> print(store.node_metadata.head(3))
                       file_name                                            node_id
        0  paul_graham_essay.txt  cadde590b82362fc7a5f8ce0751c5b30b11c0f81369df7...
        1  paul_graham_essay.txt  0567f3a9756983e1d040ec332255db94521ed5dc1b03fc...
        2  paul_graham_essay.txt  d5542515414f1bf30f6c21f0796af8bde4c513f2e72a2d...


        You can get all the nodes for file `paul_graham_essay.txt` as follows:


        >>> nodes = store.get_nodes_by_file_name("paul_graham_essay.txt")
        >>> nodes[0] # doctest: +SKIP
        TextNode(
            id_='cadde590b82362fc7a5f8ce0751c5b30b11c0f81369df7d83a76956bf22765b7',
            embedding=None,
            metadata={
                'file_path': 'examples\\data\\paul_graham_essay.txt',
                'file_name': 'paul_graham_essay.txt',
                'file_type': 'text/plain',
                'file_size': 75395,
                'creation_date': '2024-10-24',
                'last_modified_date': '2024-09-16',
                'document_title': 'Based on the candidate titles and content, I would suggest a comprehensive title
                    that captures the essence of the text. Here\'s a potential title:\n\n"From Early Days ***'
            },
            excluded_embed_metadata_keys=['file_name'],
            excluded_llm_metadata_keys=['file_name'],
            relationships={
                <NodeRelationship.SOURCE: '1'>:
                RelatedNodeInfo(
                    node_id='a25111e2e59f81bb7a0e3efb48255f4a5d4f722aaf13ffd112463fb98c227092',
                    node_type=<ObjectType.DOCUMENT: '4'>,
                    metadata={
                        'file_path': 'examples\\data\\paul_graham_essay.txt',
                        'file_name': 'paul_graham_essay.txt',
                        'file_type': 'text/plain',
                        'file_size': 75395,
                        'creation_date': '2024-10-24',
                        'last_modified_date': '2024-09-16'
                    },
                    hash='2a494d84cd0ab1e73396773258b809a47739482c90b80d5f61d374e754c3ef06'
                ),
                <NodeRelationship.NEXT: '3'>: RelatedNodeInfo(node_id='15478c7a-fdab-40c8-92e7-42973b9d3b28', node_type=<ObjectType.TEXT: '1'>, metadata={}, hash='424546c0aa78015988ced235522cdd238633d5edc1b92667cbdcda44d72613ec')}, text='What I Worked On\r\n\r\nFebruary 2021\r\n\r\nBefore college the two main things I worked on, outside of school, were writing and programming. I didn\'t write essays. I wrote what beginning writers were supposed to write then, and probably still are: short stories. My stories were awful. They had hardly any plot, just characters with strong feelings, which I imagined made them deep.\r\n\r\nThe first programs I tried writing were on the IBM 1401 that our school district used for what was then called "data processing." This was in 9th grade, so I was 13 or 14. The school district\'s 1401 happened to be in the basement of our junior high school, and my friend Rich Draves and I got permission to use it. It was like a mini Bond villain\'s lair down there, with all these alien-looking machines — CPU, disk drives, printer, card reader — sitting up on a raised floor under bright fluorescent lights.\r\n\r\nThe language we used was an early version of Fortran. You had to type programs on punch cards, then stack them in the card reader and press a button to load the program into memory and run it. The result would ordinarily be to print something on the spectacularly loud printer.\r\n\r\nI was puzzled by the 1401. I couldn\'t figure out what to do with it. And in retrospect there\'s not much I could have',
                mimetype='text/plain',
                start_char_idx=4,
                end_char_idx=2027,
                text_template='[Excerpt from document]\n{metadata_str}\nExcerpt:\n-----\n{content}\n-----\n',
                metadata_template='{key}: {value}', metadata_seperator='\n'
                )
        ```
        """
        metadata_index = self.node_metadata
        if exact_match:
            doc_ids = metadata_index.loc[
                metadata_index["file_name"] == file_name, "node_id"
            ].values
        else:
            doc_ids = metadata_index.loc[
                metadata_index["file_name"].str.contains(file_name, regex=True),
                "node_id",
            ].values
        docs = self.docstore.get_nodes(doc_ids)
        return docs

    @staticmethod
    def apply_extractors(
        documents: List[Union[Document, BaseNode]],
        extractors: Dict[str, Dict[str, int]] = None,
    ) -> Sequence[BaseNode]:
        r"""Extract information from a list of documents using predefined extractors.

        Parameters
        ----------
        documents : List[Union[Document, BaseNode]]
            List of documents or nodes to process. Each document should be an instance of `Document` or `BaseNode`.
        extractors : Dict[str, Dict[str, Any]], optional
            A dictionary defining the information extraction configuration. If not provided, default extractors will be used.

            - Example format for `info`

             .. code-block:: rst

                {
                    "text_splitter": {"separator": " ", "chunk_size": 512, "chunk_overlap": 128},
                    "title": {"nodes": 5},
                    "question_answer": {"questions": 3},
                    "summary": {"summaries": ["prev", "self"]},
                    "keyword": {"keywords": 10},
                    "entity": {"prediction_threshold": 0.5}
                }


        Returns
        -------
        Sequence[BaseNode]
            A sequence of processed nodes with extracted metadata. Extracted data is stored in the node's `metadata`
            field under the following keys:

                - "document_title": Extracted title.
                - "questions_this_excerpt_can_answer": Extracted questions.
                - "summary": Extracted summaries.
                - "keywords": Extracted keywords.
                - "entities": Extracted entities.

        Examples
        --------
        First create a config loader object:
        ```python
        >>> from llama_utils.utils.config_loader import ConfigLoader
        >>> config_loader = ConfigLoader()

        ```

        You can extract information from a single document as follows:

        ```python
        >>> docs = [Document(text="Sample text", metadata={})]
        >>> extractors_info = {
        ...     "text_splitter": {"separator": " ", "chunk_size": 512, "chunk_overlap": 128},
        ...     "title": {"nodes": 5},
        ...     "summary": {"summaries": ["prev", "self"]}
        ... }
        >>> extracted_nodes = Storage.apply_extractors(docs, extractors_info) # doctest: +SKIP
        Parsing nodes: 100%|██████████| 1/1 [00:00<00:00, 1000.31it/s]
        100%|██████████| 1/1 [00:05<00:00,  5.82s/it]
        100%|██████████| 1/1 [00:00<00:00,  1.54it/s]
        >>> len(extracted_nodes) # doctest: +SKIP
        1
        >>> print(extracted_nodes[0].document_metadata) # doctest: +SKIP
        {
            'document_title': "I'm excited to help! Unfortunately, there doesn't seem to be any text provided.
                Please go ahead and share the sample text, and I'll do my best to give you a comprehensive title
                that summarizes all the unique entities, titles, or themes found in it.",
            'section_summary': "I apologize, but since there is no provided text, I have nothing to summarize.
                Please provide the sample text, and I'll be happy to help you summarize the key topics and
                entities!"
        }
        ```
        You can extract information from a list of documents as follows:

        ```python
        >>> data_path = "examples/data/essay"
        >>> docs = Storage.read_documents(data_path)
        >>> extractors_info = {
        ...     "text_splitter": {"separator": " ", "chunk_size": 512, "chunk_overlap": 128},
        ...     "title": {"nodes": 5},
        ...     "question_answer": {"questions": 1},
        ... }

        >>> extracted_docs = Storage.apply_extractors(docs, extractors_info) # doctest: +SKIP
        Parsing nodes: 100%|██████████| 1/1 [00:00<00:00,  4.52it/s]
        100%|██████████| 5/5 [00:15<00:00,  3.19s/it]
        100%|██████████| 53/53 [03:46<00:00,  4.27s/it]
         26%|██▋       | 14/53 [00:48<02:08,  3.29s/it]
        100%|██████████| 53/53 [00:47<00:00,  1.13it/s]
        >>> len(extracted_docs) # doctest: +SKIP
        53
        >>> print(extracted_docs[0]) # doctest: +SKIP
        Node ID: 9b4fca22-7f1f-4876-bb71-d4b29500daa3
        Text: What I Worked On    February 2021    Before college the two main
        things I worked on, outside of school, were writing and programming. I
        didn't write essays. I wrote what beginning writers were supposed to
        write then, and probably still are: short stories. My stories were
        awful. They had hardly any plot, just characters with strong feelings,
        whic...
        >>> print(extracted_docs[0].extra_info) # doctest: +SKIP
        {
            'file_path': 'examples\\data\\essay\\paul-graham-essay.txt',
            'file_name': 'paul-graham-essay.txt',
            'file_type': 'text/plain',
            'file_size': 75395,
            'creation_date': '2024-10-25',
            'last_modified_date': '2024-09-16',
            'document_title': 'After reviewing the potential titles and themes mentioned in the context,
                I would suggest the following comprehensive title \n\n"A Personal Odyssey ***,'.
            'questions_this_excerpt_can_answer': "Based on the provided context, here's a question that this
                context can specifically answer:\n\nWhat was Paul Graham's experience with the IBM ***",
            'section_summary': 'Here is a summary of the key topics and entities in the section:\n\n**Key
                Topics:**\n\n1. Paul Graham\'s early experiences with writing and programming.\n2. His work on ***',
            'excerpt_keywords': 'Here are three unique keywords for this document:\n\nPaul Graham, IBM 1401,
                Microcomputers'
        }
        ```
        """
        extractors = EXTRACTORS.copy() if extractors is None else extractors

        extractors = [
            EXTRACTORS[key](**val)
            for key, val in extractors.items()
            if key in EXTRACTORS
        ]
        pipeline = IngestionPipeline(transformations=extractors)

        nodes = pipeline.run(
            documents=documents,
            in_place=True,
            show_progress=True,
        )
        return nodes


def read_metadata_index(path: str) -> pd.DataFrame:
    """Read the ID mapping from a JSON file."""
    file_path = os.path.join(path, ID_MAPPING_FILE)
    data = pd.read_csv(file_path, index_col=0)
    return data


def save_metadata_index(data: pd.DataFrame, path: str):
    """Save the ID mapping to a JSON file."""
    data.to_csv(path, index=True)


def create_metadata_index_existing_docs(docs: Dict[str, BaseNode]):
    """Create a metadata index for existing documents."""
    metadata_index = {}
    i = 0
    for key, val in docs.items():
        if "file_name" in val.metadata:
            file_name = val.metadata["file_name"]
        elif "file_path" in val.metadata:
            file_name = Path(val.metadata["file_path"]).name
        else:
            file_name = f"doc_{i}"

        metadata_index[i] = {
            "file_name": file_name,
            "node_id": generate_content_hash(val.text),
        }
        i += 1
    df = pd.DataFrame.from_dict(metadata_index, orient="index")
    return df
