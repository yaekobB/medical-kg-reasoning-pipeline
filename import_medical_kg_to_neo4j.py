# import_medical_kg_to_neo4j.py

from rdflib_neo4j import Neo4jStoreConfig, Neo4jStore, HANDLE_VOCAB_URI_STRATEGY
from rdflib import Graph

# --- 1. Neo4j connection settings ---
AUTH_DATA = {
    "uri": "neo4j://127.0.0.1:7687",  # from Desktop: neo4j://127.0.0.1:7687
    "database": "mkg",                # the DB you’re connected to
    "user": "neo4j",                  # username
    "pwd": "12345678",      # <-- put your real password
}

# --- 2. Path to your medical KG TTL file ---
# This is the file your Python code already uses:
#   data\processed\data_medical_large_inferred.ttl
TTL_PATH = r"data\processed\data_medical_large_inferred.ttl"

# --- 3. Configure the Neo4j RDF store ---
config = Neo4jStoreConfig(
    auth_data=AUTH_DATA,
    handle_vocab_uri_strategy=HANDLE_VOCAB_URI_STRATEGY.IGNORE,  # keep local names
    batching=True,  # better performance for many triples
)

# --- 4. Create the RDF graph backed by Neo4j ---
graph_store = Graph(store=Neo4jStore(config=config)) # create the graph and connect to Neo4j to store triples

print(f"[IMPORT] Parsing RDF from {TTL_PATH} ...")
graph_store.parse(TTL_PATH, format="ttl")
print("[IMPORT] Done parsing, closing store...")

graph_store.close(commit_pending_transaction=True)
print("[IMPORT] Import finished successfully.")
