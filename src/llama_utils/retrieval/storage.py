"""A module for managing vector Storage and retrieval."""

import os
from pathlib import Path
from typing import Sequence, Union, List, Dict
import pandas as pd
from pandas import DataFrame
from llama_index.core.storage.docstore import SimpleDocumentStore, BaseDocumentStore
from llama_index.core.storage.index_store import SimpleIndexStore
from llama_index.core.vector_stores import SimpleVectorStore
from llama_index.core.storage.index_store.types import BaseIndexStore
from llama_index.core import StorageContext
from llama_index.core.schema import Document, TextNode, BaseNode
from llama_index.core import SimpleDirectoryReader
from llama_index.core.node_parser import TokenTextSplitter
from llama_index.core.extractors import (
    TitleExtractor,
    QuestionsAnsweredExtractor,
    KeywordExtractor,
    SummaryExtractor,
)
from llama_index.core.ingestion import IngestionPipeline
from llama_utils.utils.helper_functions import generate_content_hash
from llama_utils.utils.errors import StorageNotFoundError


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
        metadata_index: DataFrame = None,
    ):
        """Initialize the Storage.

        The constructor method takes a llama_index.core.StorageContext object that is a native llamaIndex object
        and and a metadata table (pandas.DataFrame-optional) as input.

        Parameters
        ----------
        storage_context: str, optional, default is None.
            the StorageContext object that is created by LlamaIndex (a native llamaIndex object).

        metadata_index: DataFrame, optional, default=None
            The metadata index for the documents.

            ,file_name,doc_id
            0,paul_graham_essay.txt,cadde590b82362fc7a5f8ce0751c5b30b11c0f81369df7d83a76956bf22765b7
            1,paul_graham_essay.txt,0567f3a9756983e1d040ec332255db94521ed5dc1b03fc7312f653c0e670a0bf
            2,paul_graham_essay.txt,d5542515414f1bf30f6c21f0796af8bde4c513f2e72a2df21f0810f10826252f
        """
        if not isinstance(storage_context, StorageContext):
            raise ValueError(
                f"Storage class should be instantiated using StorageContext object, given: {storage_context}"
            )

        self._store = storage_context
        if isinstance(metadata_index, pd.DataFrame):
            self._metadata_index = metadata_index
        elif metadata_index is None:
            self._metadata_index = create_metadata_index_existing_docs(
                self._store.docstore.docs
            )
        else:
            raise ValueError(
                f"Invalid Storage backend: {storage_context}. Must be a string or StorageContext."
            )

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

            >>> store = Storage.create()
            >>> print(store)
            <BLANKLINE>
                        Documents: 0
                        Indexes: 0
            <BLANKLINE>
        """
        storage = cls._create_simple_storage_context()
        metadata_index = cls._create_metadata_index()
        return cls(storage, metadata_index)

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

        >>> store = Storage.create()
        >>> store.save("examples/paul-graham-essay-storage-example")

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
        save_metadata_index(self.metadata_index, file_path)

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
        >>> store = Storage.load("examples/paul-graham-essay-storage")
        >>> print(store) # doctest: +SKIP
        <BLANKLINE>
                    Documents: 53
                    Indexes: 2
        <BLANKLINE>
        >>> metadata = store.metadata_index
        >>> print(metadata) # doctest: +SKIP
                        file_name                                             doc_id
        0   paul_graham_essay.txt  cadde590b82362fc7a5f8ce0751c5b30b11c0f81369df7...
        1   paul_graham_essay.txt  0567f3a9756983e1d040ec332255db94521ed5dc1b03fc...
        2   paul_graham_essay.txt  d5542515414f1bf30f6c21f0796af8bde4c513f2e72a2d...
        3   paul_graham_essay.txt  120b69658a6c69ab8de3167b5ed0db77941a2b487e94d5...
        >>> docstore = store.docstore # doctest: +SKIP
        <llama_index.core.storage.docstore.simple_docstore.SimpleDocumentStore at 0x20444d31be0>
        >>> vector_store = store.vector_store
        >>> print(type(vector_store))
        'llama_index.core.vector_stores.simple.SimpleVectorStore'>
        """
        if not Path(store_dir).exists():
            raise StorageNotFoundError(f"Storage not found at {store_dir}")
        storage = StorageContext.from_defaults(persist_dir=store_dir)
        metadata_index = read_metadata_index(path=store_dir)
        return cls(storage, metadata_index)

    def __str__(self):
        message = f"""
        Documents: {len(self.docstore.docs)}
        Indexes: {len(self.index_store.index_structs())}
        """
        return message

    @property
    def metadata_index(self) -> pd.DataFrame:
        """Get the metadata index."""
        return self._metadata_index

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

            >>> store = Storage.create()

        - Then you can add documents to the store using the `add_documents` method:

            >>> data_path = "examples/data/essay"
            >>> docs = Storage.read_documents(data_path)
            >>> store.add_documents(docs)
            >>> print(store) # doctest: +SKIP
            <BLANKLINE>
                        Documents: 1
                        Indexes: 0
            <BLANKLINE>

        - once the documents are added successfully, they are added also to the metadata index.

            >>> metadata = store.metadata_index
            >>> print(metadata)
                            file_name                                             doc_id
            0   paul_graham_essay.txt  cadde590b82362fc7a5f8ce0751c5b30b11c0f81369df7...

            >>> docstore = store.docstore
            >>> print(docstore.docs)

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
        """
        new_entries = []
        file_names = []
        # Create a metadata-based index
        for doc in docs:
            # change the id to a sha256 hash if it is not already
            if generate_id:
                doc.node_id = generate_content_hash(doc.text)

            if not self.docstore.document_exists(doc.node_id) or update:
                self.docstore.add_documents([doc], allow_update=update)
                # Update the metadata index with file name as key and doc_id as value
                file_name = os.path.basename(doc.metadata["file_path"])
                if file_name in file_names:
                    file_name = f"{file_name}_{len(file_names)}"
                new_entries.append({"file_name": file_name, "doc_id": doc.node_id})
                file_names.append(file_name)
            else:
                print(f"Document with ID {doc.node_id} already exists. Skipping.")

        # Convert new entries to a DataFrame and append to the existing metadata DataFrame
        if new_entries:
            new_entries_df = pd.DataFrame(new_entries)
            self._metadata_index = pd.concat(
                [self._metadata_index, new_entries_df], ignore_index=True
            )

    @staticmethod
    def read_documents(
        path: str,
        show_progres: bool = False,
        num_workers: int = None,
        recursive: bool = False,
        **kwargs,
    ) -> List[Union[Document, TextNode]]:
        """Read documents from a directory.

        the `read_documents` method reads documents from a directory and returns a list of documents.
        the `doc_id` is sha256 hash number generated based on the document's text content.

        Parameters
        ----------
        path: str
            path to the directory containing the documents.
        show_progres: bool, optional, default is False.
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

        Examples
        --------
        You can read documents from a directory as follows:

            >>> data_path = "examples/data/essay"
            >>> docs = Storage.read_documents(data_path)
            >>> print(docs) # DOCTEST: +SKIP
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
                    text_template='{metadata_str}\n\n{content}',
                    metadata_template='{key}: {value}',
                    metadata_seperator='\n'
                )
            ]
        """
        if not Path(path).exists():
            raise FileNotFoundError(f"Directory not found: {path}")

        reader = SimpleDirectoryReader(path, recursive=recursive, **kwargs)
        documents = reader.load_data(
            show_progress=show_progres, num_workers=num_workers, **kwargs
        )

        for doc in documents:
            # exclude the file name from the llm metadata in order to avoid affecting the llm by weird file names
            doc.excluded_llm_metadata_keys = ["file_name"]
            # exclude the file name from the embeddings metadata in order to avoid affecting the llm by weird file names
            doc.excluded_embed_metadata_keys = ["file_name"]
            # Generate a hash based on the document's text content
            content_hash = generate_content_hash(doc.text)
            # Assign the hash as the doc_id
            doc.doc_id = content_hash

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
        First read the storage context from a directory:
            >>> storage_dir = "examples/paul-graham-essay-storage"
            >>> store = Storage.load(storage_dir)
            >>> print(store) # doctest: +SKIP
            <BLANKLINE>
                        Documents: 53
                        Indexes: 2
            <BLANKLINE>

        The storage context contains the following data:

            >>> print(store.metadata_index.head(3))
                           file_name                                             doc_id
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

        """
        if exact_match:
            doc_ids = self.metadata_index.loc[
                self.metadata_index["file_name"] == file_name, "doc_id"
            ].values
        else:
            doc_ids = self.metadata_index.loc[
                self.metadata_index["file_name"].str.contains(file_name, regex=True),
                "doc_id",
            ].values
        docs = self.docstore.get_nodes(doc_ids)
        return docs

    @staticmethod
    def apply_extractors(
        documents: List[Union[Document, BaseNode]],
        extractors: Dict[str, Dict[str, int]] = None,
    ) -> Sequence[BaseNode]:
        """Extract information from a list of documents using predefined extractors.

        Parameters
        ----------
        documents : List[Union[Document, BaseNode]]
            List of documents or nodes to process. Each document should be an instance of `Document` or `BaseNode`.
        extractors : Dict[str, Dict[str, Any]], optional
            A dictionary defining the information extraction configuration. If not provided, default extractors will be used.

            - Example format for `info`

            .. code-block:: python

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

            >>> from llama_utils.utils.config_loader import ConfigLoader
            >>> config_loader = ConfigLoader()

        You can extract information from a single document as follows:

            >>> docs = [Document(text="Sample text", metadata={})]
            >>> extractors_info = {
            ...     "text_splitter": {"separator": " ", "chunk_size": 512, "chunk_overlap": 128},
            ...     "title": {"nodes": 5},
            ...     "summary": {"summaries": ["prev", "self"]}
            ... }
            >>> extracted_nodes = Storage.apply_extractors(docs, extractors_info)
            Parsing nodes: 100%|██████████| 1/1 [00:00<00:00, 1000.31it/s]
            100%|██████████| 1/1 [00:05<00:00,  5.82s/it]
            100%|██████████| 1/1 [00:00<00:00,  1.54it/s]
            >>> len(extracted_nodes)
            1
            >>> print(extracted_nodes[0].metadata) # doctest: +SKIP
            {
                'document_title': "I'm excited to help! Unfortunately, there doesn't seem to be any text provided.
                    Please go ahead and share the sample text, and I'll do my best to give you a comprehensive title
                    that summarizes all the unique entities, titles, or themes found in it.",
                'section_summary': "I apologize, but since there is no provided text, I have nothing to summarize.
                    Please provide the sample text, and I'll be happy to help you summarize the key topics and
                    entities!"
            }

        You can extract information from a list of documents as follows:

            >>> data_path = "examples/data/essay"
            >>> docs = Storage.read_documents(data_path)
            >>> extractors_info = {
            >>>     "text_splitter": {"separator": " ", "chunk_size": 512, "chunk_overlap": 128},
            >>>     "title": {"nodes": 5},
            >>>     "question_answer": {"questions": 1},
            >>>     "summary": {"summaries": ["prev", "self"]},
            >>>     "keyword": {"keywords": 3},
            >>>     "entity": {"prediction_threshold": 0.5},
            >>> }

            >>> extracted_docs = Storage.apply_extractors(docs, extractors_info) # doctest: +SKIP
            Parsing nodes: 100%|██████████| 1/1 [00:00<00:00,  4.52it/s]
            100%|██████████| 5/5 [00:15<00:00,  3.19s/it]
            100%|██████████| 53/53 [03:46<00:00,  4.27s/it]
             26%|██▋       | 14/53 [00:48<02:08,  3.29s/it]
            100%|██████████| 53/53 [00:47<00:00,  1.13it/s]
            >>> len(extracted_docs)
            53
            >>> print(extracted_docs[0])
            Node ID: 9b4fca22-7f1f-4876-bb71-d4b29500daa3
            Text: What I Worked On    February 2021    Before college the two main
            things I worked on, outside of school, were writing and programming. I
            didn't write essays. I wrote what beginning writers were supposed to
            write then, and probably still are: short stories. My stories were
            awful. They had hardly any plot, just characters with strong feelings,
            whic...
            >>> print(extracted_docs[0].extra_info)
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
    metadata_index = {}
    i = 0
    for key, val in docs.items():
        metadata_index[i] = {
            "file_name": val.metadata["file_name"],
            "doc_id": generate_content_hash(val.text),
        }
        i += 1
    df = pd.DataFrame.from_dict(metadata_index, orient="index")
    return df
