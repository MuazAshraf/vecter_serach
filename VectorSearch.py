import pinecone
from sentence_transformers import SentenceTransformer
import openai

# Initialize Pinecone
pinecone.init(api_key='Your Pinecone Key',
              environment='YourEnvironemnt')

# Initialize the embedding model
model = SentenceTransformer(
    'sentence-transformers/distilbert-base-nli-mean-tokens')

# Define department data
departments = ["design", "video_production", "marketing"]

# Generate embeddings for the departments
vectors = model.encode(departments)

# Create a Pinecone index
index_name = "mojosolo"
if index_name in pinecone.list_indexes():
    pinecone.delete_index(name=index_name)

pinecone.create_index(name=index_name, dimension=768, metric='cosine')

# Insert department vectors into the Pinecone index
index = pinecone.Index(index_name)
upsert_response = index.upsert(
    vectors=list(zip(departments, [vector.tolist() for vector in vectors])),
    namespace="example-namespace"
)


from scipy.spatial.distance import cosine

def get_department(message):
    query_vector = model.encode([message])[0]
    min_distance = 1.0
    best_department = None

    for department, vector in zip(departments, vectors):
        distance = cosine(query_vector, vector)
        print(f"DEBUG: Department: {department}, Distance: {distance}")
        if distance < min_distance:
            min_distance = distance
            best_department = department

    if best_department is not None:
        return best_department
    else:
        print("DEBUG: No department found")
        return None





openai.api_key = 'Your Api Key'


def chatbot(message):
    department = get_department(message)
    if department is not None:
        response = openai.Completion.create(
            engine="text-davinci-002",
            prompt=f"[{department}] {message}",
            max_tokens=50,
            n=1,
            stop=None,
            temperature=0.7,
            top_p=0.95,
        )
        return response.choices[0].text.strip()
    else:
        return "Sorry, I couldn't understand your query."

while True:
    user_input = input("You:")
    if user_input.lower() == "exit":
        break
    response = chatbot(user_input)
    print(f"Bot: {response}")


# Query the Pinecone index using an example sentence
query_sentence = "We need a new video advertisement campaign."
query_vector = model.encode([query_sentence])[0]
query_response = index.query(
    namespace="example-namespace",
    top_k=1,
    vector=query_vector.tolist()
)

# Print the query results
print("Query results:")
if query_response.results:
    for result in query_response.results:
        print(f"ID: {result.id}, Distance: {result.distance}")
else:
    print("No results found.")
