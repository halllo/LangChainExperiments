from langchain.schema import BaseRetriever;
from langchain.embeddings.base import Embeddings;
from langchain_community.vectorstores.chroma import Chroma;

class RedundantFilterRetriever(BaseRetriever):
    embeddings: Embeddings;
    chroma: Chroma;
    
    def get_relevant_documents(self, query):
        # calculate embeeddings for the query strint
        emb = self.embeddings.embed_query(query);

        #take embeddings and feed them into that max_marginal_relvance_search_by_vector
        return self.chroma.max_marginal_relevance_search_by_vector(
            embedding=emb,
            lambda_mult=0.8,
        );

    def aget_relevant_documents(self):
        return [];
