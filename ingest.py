import fitz  # PyMuPDF
from openai import OpenAI
from qdrant_client import QdrantClient
from qdrant_client.models import PointStruct, VectorParams, Distance
import uuid
import os
import dotenv

dotenv.load_dotenv()
openAI = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))
qdrant = QdrantClient(host=os.getenv("QDRANT_HOST"),
                      port=os.getenv("QDRANT_PORT"))


class PDFIngester:

    def __init__(self, pdf_path: str, 
                 collection_name: str):
        self.pdf_path = pdf_path
        self.collection_name = collection_name

    def load_text(self) -> str:
        doc = fitz.open(self.pdf_path)
        text = ""
        for page in doc:
            text += page.get_text()
        return text

    def chunk_text(self, text: str,
                   chunk_size: int = 500) -> list:
        words = text.split()
        chunks = [" ".join(words[i:i + chunk_size]) for i in range(0, len(words), chunk_size)]
        return chunks

    def embed_and_store(self, chunks: list):
        if qdrant.collection_exists(self.collection_name):
            qdrant.delete_collection(self.collection_name)
        qdrant.create_collection(
            collection_name=self.collection_name,
            vectors_config=VectorParams(size=1536, distance=Distance.COSINE)
        )

        points = []
        for i, chunk in enumerate(chunks):
            response = openAI.embeddings.create(
                model="text-embedding-3-small",
                input=chunk
            )
            vector = response.data[0].embedding
            points.append(PointStruct(
                id=str(uuid.uuid4()),
                vector=vector,
                payload={"text": chunk, "chunk_index": i}
            ))

        qdrant.upsert(
            collection_name = self.collection_name,
            points = points
        )
        print(f"Stored {len(points)} chunks in {self.collection_name}")

    def query_collection(self, question: str) -> str:
        search_result = qdrant.query_points(
            collection_name=self.collection_name,
            query=openAI.embeddings.create(
                model="text-embedding-3-small",
                input=question
            ).data[0].embedding,
            limit=3
        ).points
        context = "\n\n".join([r.payload["text"] for r in search_result])
        answer = openAI.chat.completions.create(
            model= "gpt-4o-mini",
            temperature= 0.2,
            messages=[
                {"role": "system", "content": "Answer based on context only."},
                {"role": "user", "content": f"Context:\n{context}\n\nQuestion: {question}"}
        ], stream=False)
        return answer.choices[0].message.content

class QueryEngine:

    def __init__(self, collection_name: str):
        self.collection_name = collection_name
        
    def validate_question(self, question: str) -> bool:
        return len(question.split()) >= 3
        
    def build_prompt(self, context: str,
                     question: str) -> dict:
        return {"system": "Answer based on context only.",
                "user": f"Context:\n{context}\n\nQuestion: {question}"}
    
    def format_answer(self, raw_answer: str) -> str:
        answer = raw_answer.strip()
        return answer



if __name__ == "__main__":

    ingester = PDFIngester(r"C:\Users\DELL\Downloads\Shaheer_Salal_CV.pdf", "My_CV")
    
    text = ingester.load_text()
    print(f"Extracted {len(text)} characters")

    chunks = ingester.chunk_text(text)
    print(f"Created {len(chunks)} chunks")
    
    ingester.embed_and_store(chunks)
    
    collection_info = qdrant.get_collection("My_CV")
    print(f"Points in Qdrant: {collection_info.points_count}")

    answer = ingester.query_collection("What is Shaheer's experience?")
    print(answer)

    question = QueryEngine('My_CV')
    print(question.validate_question("What   i s Shaheer's     experience?"))
    print(question.build_prompt('what context should i give it', "What is Shaheer's experience?"))
    print(question.format_answer("  This is a test answer.  "))