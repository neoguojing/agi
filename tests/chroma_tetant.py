import chromadb
from chromadb import DEFAULT_DATABASE
from chromadb import Settings

adminClient = chromadb.AdminClient(Settings(
    chroma_api_impl="chromadb.api.segment.SegmentAPI",
    is_persistent=True,
    persist_directory="test",
))


# For Remote Chroma server:
# 
# adminClient= chromadb.AdminClient(Settings(
#   chroma_api_impl="chromadb.api.fastapi.FastAPI",
#   chroma_server_host="localhost",
#   chroma_server_http_port="8000",
# ))

def get_or_create_tenant_for_user(user_id):
    tenant_id = user_id
    try:
        adminClient.get_tenant(tenant_id)
    except Exception as e:
        adminClient.create_tenant(tenant_id)
        adminClient.create_database(DEFAULT_DATABASE, tenant_id)
    return tenant_id, DEFAULT_DATABASE


user_id = "user1"

tenant, database = get_or_create_tenant_for_user(user_id)
# replace with chromadb.HttpClient for remote Chroma server
client = chromadb.PersistentClient(path="test", tenant=tenant, database=database)
collection = client.get_or_create_collection("user_collection")
collection.add(
    documents=["This is document1", "This is document2"],
    ids=["doc1", "doc2"],
)