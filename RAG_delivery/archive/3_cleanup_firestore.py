from google.cloud import firestore
from google.oauth2 import service_account

# 1. Path to your secret JSON key file
cred_path = ".secrets/aria-uploader-key.json"

# 2. Authenticate using the service account file
credentials = service_account.Credentials.from_service_account_file(cred_path)

# 3. Initialize the Firestore client
db = firestore.Client(credentials=credentials)


def delete_entire_collection(collection_name):
  """Deletes a collection and all its documents recursively."""
  collection_ref = db.collection(collection_name)
  print(f"Starting deletion of collection: {collection_name}")

  # Use the client's recursive_delete method
  # Note: bulk_writer handles the underlying batch operations automatically
  bulk_writer = db.bulk_writer()
  db.recursive_delete(reference=collection_ref, bulk_writer=bulk_writer)

  # Flush to ensure all pending delete operations complete
  bulk_writer.flush()
  print(f"Successfully deleted collection: {collection_name}")


if __name__ == "__main__":
  # Replace 'your_collection_name' with your actual target collection
  #delete_entire_collection("your_collection_name")
  delete_entire_collection("object_embeddings")
  delete_entire_collection("rag_object_occurrences")
