# Notebook RAG

A Google NotebookLM-style RAG application. Upload a PDF or plain text document, index it into Qdrant, and ask questions that are answered only from retrieved document chunks.

## Features

- Upload a new `.pdf` or `.txt` document from the browser
- Parse document text on the server
- Chunk the document with overlap
- Embed chunks using Gemini `embedding-001`
- Store and search embeddings in Qdrant
- Generate grounded answers using Gemini `gemini-1.5-flash`
- Show retrieved source snippets for every answer

## RAG Pipeline

1. **Ingestion**: The upload API accepts PDF and text files. PDFs are loaded page-by-page with LangChain's `PDFLoader`; text files are read directly.
2. **Chunking**: The app uses `RecursiveCharacterTextSplitter` with:
   - `chunkSize`: `1000`
   - `chunkOverlap`: `200`

   This strategy keeps chunks small enough for focused retrieval while preserving context across chunk boundaries.

3. **Embedding**: Every chunk is embedded with Gemini `embedding-001`.
4. **Vector storage**: Chunks and metadata are stored in a Qdrant collection. Each uploaded document gets its own collection, so questions only retrieve from the current document.
5. **Retrieval**: For each question, the app retrieves the top 5 most similar chunks from Qdrant.
6. **Generation**: The LLM receives only the retrieved chunks as context and is instructed to say it cannot find the answer if the chunks do not support the answer.

## Tech Stack

- Next.js
- React
- LangChain
- Qdrant
- Google Gemini
- TypeScript

## Environment Variables

Create `.env.local` locally or configure these in your deployment:

```bash
GOOGLE_API_KEY=your-gemini-api-key
QDRANT_URL=https://your-qdrant-cluster-url
QDRANT_API_KEY=your-qdrant-api-key
QDRANT_COLLECTION_PREFIX=notebooklm
```

For local Qdrant without an API key:

```bash
docker run -p 6333:6333 -v "$(pwd)/qdrant_storage:/qdrant/storage" qdrant/qdrant
```

Then set:

```bash
QDRANT_URL=http://localhost:6333
```

## Run Locally

```bash
npm install
npm run dev
```

Open `http://localhost:3000`.

## Deployment

Deploy to Vercel and add the environment variables above. Use Qdrant Cloud for the live project link so the deployed app can store and retrieve vectors without local setup.

```bash
npx vercel@latest --prod
```

## Grounding Behavior

The answer route builds a system prompt from retrieved chunks only. The model is explicitly instructed:

- Do not use outside knowledge
- Do not invent facts
- Say it cannot find the answer if the sources do not contain it

The UI also displays source snippets so the answer can be checked against the uploaded document.
