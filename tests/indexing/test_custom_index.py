import pytest
from llama_index.core import VectorStoreIndex
from llama_index.core.data_structs.data_structs import IndexDict
from llama_index.core.embeddings.mock_embed_model import MockEmbedding
from llama_index.core.schema import Document, TextNode
from llama_index.core.vector_stores import SimpleVectorStore

from llama_utils.indexing.custom_index import CustomIndex


class TestclassMethods:

    def test_create_from_document(self, document: Document):
        """
        The test checks if the CustomIndex object is created from a list of a Document and a string objects.

        the CustomIndex object should have the following attributes:
        - id: str
        - metadata: IndexDict
            - index metadata data object from llamaindex.core.data_structs.data_structs
            - The metadata should have a nodes_dict with the node id as the key and the node id as the value.
            Hint: the node id is not the same as the document id, so the node_dict will have a different key and
            value, than the doc_ids.
        - doc_ids: List[str]
            the value should have two strings in a list, the string in the input is converted into a document and
            the other document.
        - embeddings: Dict[str, List[float]]
            the value should be a dictionary with the node id as the key and the embedding as the value.
        - vector_store: SimpleVectorStore
            a Simple vector store object.
        - index: VectorStoreIndex
        """
        content_id = [
            "8323ac870e04bcf4b64eb04624001a025027d8f797414072df1b81e087f74fb3",
            "06c83efcd22e3b755ca95ffe22954a304740027769fb541a126a5d23b6f5ac1f",
        ]
        doc_list = [document, "any text"]
        index = CustomIndex.create_from_documents(doc_list)
        assert isinstance(index, CustomIndex)
        string = f"\n        Index ID: {index.id}\n        Number of Document: {len(index.doc_ids)}\n        "
        assert index.__str__() == string
        assert index.__repr__() == string
        assert isinstance(index.index, VectorStoreIndex)
        assert isinstance(index.id, str)
        assert isinstance(index.vector_store, SimpleVectorStore)
        assert index.doc_ids == content_id
        assert isinstance(index.embeddings, dict)
        assert list(index.embeddings.keys()) == content_id
        # the embedding is a list of 768 floats based on the BERT model
        assert len(index.embeddings[content_id[0]]) == 768

    def test_create_from_nodes(self, text_node: TextNode):
        """
        The test checks if the CustomIndex object is created from a list of one TextNode object.

        the CustomIndex object should have the following attributes:
        - id: str
        - metadata: IndexDict
            - index metadata data object from llamaindex.core.data_structs.data_structs
            - The metadata should habe a nodes_dict with the node id as the key and the node id as the value.
        - doc_ids: List[str]
            the value should be an empty list, as the input is a node not a document.
        - embeddings: Dict[str, List[float]]
            the value should be a dictionary with the node id as the key and the embedding as the value.
        - vector_store: SimpleVectorStore
            a Simple vector store object.
        - index: VectorStoreIndex
        """
        index = CustomIndex.create_from_nodes([text_node])
        assert isinstance(index, CustomIndex)
        assert len(index.embeddings) == 1
        assert list(index.index.docstore.docs.keys()) == [text_node.id_]
        assert index.doc_ids == []
        assert index.metadata.nodes_dict == {text_node.id_: text_node.id_}

    def test_create_custome_index_wrong_input(self):
        with pytest.raises(ValueError):
            CustomIndex("wrong input")


def test_metadata(text_node: TextNode):
    index = CustomIndex.create_from_nodes([text_node])
    metadata = index.metadata
    assert isinstance(metadata, IndexDict)
    assert list(metadata.nodes_dict.keys()) == [text_node.id_]


class TestProperties:
    def test_embedding_model(self, vector_store_index: VectorStoreIndex):
        c_index = CustomIndex(vector_store_index, embedding_model=MockEmbedding)
        assert c_index.embedding_model == MockEmbedding
        c_index.embedding_model = MockEmbedding(embed_dim=768)
        assert c_index.embedding_model.model_name == "unknown"

    def test_node_ids(self, vector_store_index: VectorStoreIndex):
        c_index = CustomIndex(vector_store_index, embedding_model=MockEmbedding)
        assert c_index.node_id_list == ["d2"]


class TestAddDocument:

    docs = [Document(text="text", id_="id")]
    nodes = [TextNode(text="text", id_="node")]
    node_conetent_hash = (
        "982d9e3eb996f559e633f4d194def3761d909f5a3b647d1a851fead67c32c9d1"
    )

    def test_document(self, vector_store_index: VectorStoreIndex):
        """
        The test checks if the add_documents method adds a document to the index.

        Parameters
        ----------
        vector_store_index: VectorStoreIndex
            The vector_store_index is created from one TextNode object.

        The CustomIndex object should have the following attributes:
        - id: str
        - metadata: IndexDict
            - index metadata data object from llamaindex.core.data_structs.data_structs
            - The metadata should have a nodes_dict with the node id as the key and the node id as the value.
        - doc_ids: List[str]
            the value should have one strings in a list, as the input vector_store_index has one node and no documents.
        - embeddings: Dict[str, List[float]]
            the value should be a dictionary with the node id as the key and the embedding as the value.
        - vector_store: SimpleVectorStore
            a Simple vector store object.
        - index: VectorStoreIndex

        """
        c_index = CustomIndex(vector_store_index)
        c_index.add_documents(self.docs)
        assert len(c_index.doc_ids) == 1
        assert c_index.doc_ids == [self.docs[0].id_]
        assert len(c_index.embeddings) == 2
        # the node_dict will have a node id that represent the document that was added but the node id is different
        # than the document id that is generated by the `add_documents` method in the CustomIndex class. or the
        # id of the document that was added.
        assert list(c_index.metadata.nodes_dict.keys())[0] == "d2"

    def test_add_document_wrong_input(self, vector_store_index: VectorStoreIndex):
        c_index = CustomIndex(vector_store_index)
        with pytest.raises(ValueError):
            c_index.add_documents("wrong input")

    def test_nodes(self, vector_store_index: VectorStoreIndex):
        """
        The test checks if the add_nodes method adds a document to the index.

        Parameters
        ----------
        vector_store_index: VectorStoreIndex
            The vector_store_index is created from one TextNode object.

        The CustomIndex object should have the following attributes:
        - id: str
        - metadata: IndexDict
            - index metadata data object from llamaindex.core.data_structs.data_structs
            - The metadata should have a nodes_dict with the node id as the key and the node id as the value.
        - doc_ids: List[str]
            the value should have one strings in a list, as the input vector_store_index has one node and no documents.
        - embeddings: Dict[str, List[float]]
            the value should be a dictionary with the node id as the key and the embedding as the value.
        - vector_store: SimpleVectorStore
            a Simple vector store object.
        - index: VectorStoreIndex
        """
        c_index = CustomIndex(vector_store_index)
        c_index.add_documents(self.nodes)
        assert len(c_index.doc_ids) == 0
        # the node_dict will have a node id that represent the document that was added but the node id is different
        # than the document id that is generated by the `add_documents` method in the CustomIndex class. or the
        # id of the document that was added.
        assert list(c_index.metadata.nodes_dict.keys())[1] == self.node_conetent_hash
