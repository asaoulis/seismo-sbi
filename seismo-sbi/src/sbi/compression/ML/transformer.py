import torch
import math

from torch import nn

class TransformerEncoder(nn.Module):

    def __init__(self, num_encoder_blocks, token_length, key_dim, value_dim, num_heads) -> None:
        super().__init__()

        self.event_token = nn.Parameter(torch.randn((token_length)), requires_grad=True)

        self.encoder_blocks = nn.ModuleList(
                            [TransformerEncoderBlock(token_length, key_dim, value_dim, num_heads) \
                                for _ in range(num_encoder_blocks)
                            ]
                        )

    def forward(self, x):

        x = self.concat_event_token(x)

        for encoder_block in self.encoder_blocks:
            x = encoder_block.forward(x)
        
        event_token_attention = x[:,-1,:]

        return event_token_attention
    
    def concat_event_token(self, x : torch.Tensor):
        batch_size = x.shape[0]
        batch_event_tokens = self.event_token.unsqueeze(0).unsqueeze(0).repeat(batch_size,1,1)
        x = torch.concat([x,batch_event_tokens], dim=1)
        return x

class TransformerEncoderBlock(nn.Module):

    def __init__(self, token_length, key_dim, value_dim, num_heads) -> None:
        super().__init__()

        self.multi_head_attention = MultiHeadAttention(token_length, key_dim, value_dim, num_heads)
        self.add_norm_1 = AddNorm(token_length)

        self.tokenwise_feedforward = TokenwiseFeedForwardBlock(token_length, 128, token_length)
        self.add_norm_2 = AddNorm(token_length)

    def forward(self, x : torch.Tensor):

        batch_size, seq_len, token_length = x.shape
        attention_weights = self.multi_head_attention(x)
        normed_weights = self.add_norm_1(attention_weights, x)

        flattened_tokens = normed_weights.reshape((batch_size * seq_len, token_length))

        tokenwise_FFN = self.tokenwise_feedforward(flattened_tokens).reshape((batch_size, seq_len, token_length))
        normed_encoder_outputs = self.add_norm_2(tokenwise_FFN, normed_weights)

        return normed_encoder_outputs


class MultiHeadAttention(nn.Module):

    def __init__(self, token_dim, key_dim, value_dim, num_heads) -> None:
        super().__init__()

        self.token_dim = token_dim
        self.value_dim = value_dim
        self.num_heads = num_heads

        self.query_mat = nn.Linear(token_dim, key_dim)
        self.key_mat = nn.Linear(token_dim, key_dim)
        self.value_mat = nn.Linear(token_dim, value_dim)

        multihead_projectors= [AttentionProjectors(key_dim, value_dim) for _ in range(num_heads)]
        self.attention_heads = nn.ModuleList([AttentionHead(projectors, key_dim) for projectors in multihead_projectors])

        self.concat_heads_transform = nn.Linear(num_heads*value_dim, token_dim)
    
    def forward(self, x : torch.Tensor):
        
        batch_size, sequence_length, token_length = x.shape
        x = x.reshape((batch_size * sequence_length, token_length))

        q, k, v = self.query_mat(x), self.key_mat(x), self.value_mat(x)

        concat_attention = torch.concat([attention_head.forward(q,k,v, sequence_length) for attention_head in self.attention_heads], dim=-1)
        transformed_attn = self.concat_heads_transform(concat_attention)
        return transformed_attn.reshape((batch_size, sequence_length, self.token_dim))


class AttentionProjectors(nn.Module):

    def __init__(self, dk, dv) -> None:
        super().__init__()

        self.key_matrix = nn.Linear(dk, dk)
        self.query_matrix = nn.Linear(dk, dk)

        self.value_matrix = nn.Linear(dv, dv)

class AttentionHead(nn.Module):

    def __init__(self, attention_projector : AttentionProjectors, key_dim) -> None:
        super().__init__()

        self.sqrt_dk = math.sqrt(key_dim)
        
        self.attention_projector = attention_projector

    def forward(self, q, k, v, seq_len):

        Q = self.attention_projector.query_matrix(q)
        K = self.attention_projector.key_matrix(k)

        V = self.attention_projector.value_matrix(v)

        score = torch.matmul(Q, K.T)/self.sqrt_dk
        batch_size_times_seq_len, dk = score.shape
        reshaped_score = score.reshape(batch_size_times_seq_len// seq_len, seq_len, dk)
        attention_weights = nn.functional.softmax(reshaped_score, dim=-1)
        attention_weights = attention_weights.reshape(batch_size_times_seq_len, dk)

        return torch.matmul(attention_weights, V)


class AddNorm(nn.Module):

    def __init__(self, norm_shape) -> None:
        super().__init__()
        self.layer_norm = nn.LayerNorm(norm_shape)

    def forward(self, x, y):
        return self.layer_norm(x + y)


class TokenwiseFeedForwardBlock(nn.Module): 
    def __init__(self, ffn_num_input, ffn_num_hiddens, ffn_num_outputs):
        super().__init__()
        self.dense1 = nn.Linear(ffn_num_input, ffn_num_hiddens)
        self.relu = nn.ReLU()
        self.dense2 = nn.Linear(ffn_num_hiddens, ffn_num_outputs)

    def forward(self, X):
        return self.dense2(self.relu(self.dense1(X)))