# Design and Architecture Diagrams

## 1. Class Diagram

```mermaid
classDiagram
    class IndexManager {
        +List~str~ ids
        +List~CustomIndex~ indexes
        +load_from_storage(storage: Storage) IndexManager
        +create_from_storage(storage: Storage) IndexManager
    }
    class CustomIndex {
        +VectorStoreIndex index
        +IndexDict metadata
        +List~str~ doc_ids
        +str id
        +create_from_documents(document: List~Union~Document, str~, generate_id: bool) CustomIndex
        +create_from_nodes(nodes: List~TextNode~) CustomIndex
    }

    class VisionModel {
        +List~str~ images
        +str prompt_template
        +encode_image(image: np.ndarray, image_format: str) str
        +encode_images() List~str~
        +trigger_model(max_tokens: int, temperature: float, detail: str) str
        +to_markdown(path: str)
    }
    class Storage {
        +StorageContext store
        +BaseDocumentStore docstore
        +BaseIndexStore index_store
        +vector_store
        +save(store_dir: str)
        +load(store_dir: str) Storage
        +add_documents(docs: Sequence~Union~Document, TextNode~, generate_id: bool, update: bool)
        +read_documents(path: str) List~Union~Document, TextNode~
    }
    class ConfigLoader {
        +Settings settings
        +llm
        +embedding
    }
    class Logger {
        +__init__(name: str, level: int, file_name: str)
    }
    class StorageNotFoundError {
        +__init__(error_message: str)
    }
    IndexManager --> CustomIndex
    IndexManager --> Storage
    CustomIndex --> VectorStoreIndex
    CustomIndex --> BasePydanticVectorStore
    CustomIndex --> Document
    CustomIndex --> TextNode
    VisionModel --> AzureOpenAI
    VisionModel --> ConfigLoader
    VisionModel --> Storage
    Storage --> StorageContext
    Storage --> Document
    Storage --> TextNode
    ConfigLoader --> Settings
    ConfigLoader --> HuggingFaceEmbedding
    Logger --> logging
    StorageNotFoundError --> Exception
```

### Visibility Symbols in Mermaid Class Diagrams

- `+` : **Public** – The member is accessible from outside the class.
- `-` : **Private** – The member is accessible only within the class.
- `#` : **Protected** – The member is accessible within the class and its subclasses.



## 2. Module Dependency Diagram

```mermaid
graph TD
    IndexManager -->|Imports| CustomIndex
    IndexManager -->|Imports| Storage
    CustomIndex -->|Imports| VectorStoreIndex
    CustomIndex -->|Imports| Document
    CustomIndex -->|Imports| TextNode
    VisionModel -->|Imports| AzureOpenAI
    VisionModel -->|Uses| ConfigLoader
    Storage -->|Uses| StorageContext
    Storage -->|Uses| Document
    ConfigLoader -->|Uses| HuggingFaceEmbedding
    ConfigLoader -->|Uses| OllamaLLM
    Logger -->|Depends| logging
```

## 3. Sequence Diagram

```mermaid
sequenceDiagram
    actor User
    participant IndexManager
    participant CustomIndex
    participant Storage
    User ->> IndexManager: Create IndexManager
    IndexManager ->> Storage: Load from storage
    Storage ->> IndexManager: Return data
    IndexManager ->> CustomIndex: Create CustomIndex
    CustomIndex ->> IndexManager: Return index object
```

## 4. Component Diagram

```mermaid
graph TD
    subgraph Application
        IndexManager
        CustomIndex
        Storage
    end
    subgraph ExternalSystems
        AzureOpenAI
        DocumentStore
    end
    IndexManager -->|Communicates| CustomIndex
    IndexManager -->|Uses| Storage
    Storage -->|Stores| DocumentStore
    VisionModel -->|API| AzureOpenAI
```

## 5. Deployment Diagram

```mermaid
graph TD
    user[User System]
    app[Python Application]
    storage[Storage Directory]
    azure[Azure OpenAI Service]
    user --> app
    app --> storage
    app --> azure
```

## 6.Data Flow Diagram

```mermaid
graph LR
    Input[User Input] --> Process[IndexManager]
    Process --> StorageSystem[Storage]
    StorageSystem --> Output[Retrieved Data]
```


## 7.Deployment Diagram

```mermaid
graph TD
    User -->|Requests| PythonApp[Python Application]
    PythonApp -->|Reads/Writes| Storage[Local/Cloud Storage]
    PythonApp -->|Interacts| Azure[Azure OpenAI Services]
```


## 8.State Diagram

```mermaid
stateDiagram-v2
    [*] --> Initialized
    Initialized --> LoadingIndexes
    LoadingIndexes --> IndexReady: Success
    LoadingIndexes --> [*]: Failure
    IndexReady --> [*]
```


## 9.Activity Diagram

```mermaid
flowchart TD
    Start --> LoadIndexes
    LoadIndexes --> ProcessData
    ProcessData --> GenerateEmbeddings
    GenerateEmbeddings --> End
```


## 10.Package Diagram

```mermaid
graph TB
    Package[llama-utils]
    Package --> SubPackage1[Indexing]
    Package --> SubPackage3[Storage]
    SubPackage1 --> Module1[index_manager.py]
    SubPackage1 --> Module2[custom_index.py]
    SubPackage3 --> Module5[storage.py]
    SubPackage3 --> Module6[config_loader.py]
```


## 11.Network Diagram

```mermaid
graph TD
    Client --> API[REST API]
    API --> Storage[Data Storage]
    API --> Service[Azure OpenAI]
```
