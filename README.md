# rag_pdf_qa
```mermaid
graph LR
    A[User Uploads PDF(s)] --> B[Streamlit Interface];
    B -- User Enters Question --> C{LangChain Processing};
    C -- Text Splitting --> D[RecursiveCharacterTextSplitter];
    D -- Embeddings --> E[HuggingFace Embeddings <br> (all-MiniLM-L6-v2)];
    E -- Vector Storage --> F[FAISS];
    C -- Retrieve Relevant Chunks --> F;
    F -- Combine Chunks & Question --> G{Prompt Engineering};
    G -- Send to LLM --> H[ChatGroq];
    H -- Generate Answer --> B;
    B -- Display Answer & Chat History --> J[User];
    style C fill:#f9f,stroke:#333,stroke-width:2px
    style G fill:#ccf,stroke:#333,stroke-width:2px

    subgraph Key Technologies
        D
        E
        F
        H
    end

    ```