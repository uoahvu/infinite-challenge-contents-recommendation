import torch
import torch.nn as nn
from sentence_transformers import SentenceTransformer
import faiss


class ContentEmbedding(nn.Module):
    def __init__(self, hidden_size, num_season):
        super(ContentEmbedding, self).__init__()
        self.hidden_size = hidden_size
        self.num_season = num_season

        self.season_embedding = nn.Embedding(num_season, hidden_size)
        self.description_embedding = SentenceTransformer(
            "sentence-transformers/all-MiniLM-L6-v2"
        )

    def forward(self, x):
        # 시즌
        embedded_season = self.season_embedding(torch.tensor(x.loc[:, "season"]).long())
        # 특집
        special_epicode = nn.functional.one_hot(torch.tensor(x.loc[:, "special"]))
        # 제목텍스트
        embedded_title = torch.tensor(
            self.description_embedding.encode(x.loc[:, "title_"])
        )
        # 설명텍스트
        embedded_description = torch.tensor(
            self.description_embedding.encode(x.loc[:, "description"])
        )
        result = torch.concat(
            [embedded_season, special_epicode, embedded_title, embedded_description],
            dim=-1,
        )
        print(result.shape)
        return result

    def inference(self, x):
        output = self(x)
        index = faiss.IndexFlatL2(output.size(-1))
        index.add(output.detach().numpy())
        return index, output


def candidate_generator(index, query, k):
    D, I = index.search(query.reshape(1, -1), k=k)

    return D, I
