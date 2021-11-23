import torch.nn as nn
import torch

device = 'cuda:0' if torch.cuda.is_available() else 'cpu'

class ConvModelT(nn.Module):
    def __init__(self, d_model, num_heads, num_enc, num_dec, d_ff, dropout_rate, activation, vocab_size, use_common_enc=False):
        super().__init__()
        intermediate1 = 128
        self.use_common_enc = use_common_enc
        if not use_common_enc:
            self.enc_embedder = nn.Embedding(num_embeddings=vocab_size, embedding_dim=d_model)
            self.dec_embedder = nn.Embedding(num_embeddings=vocab_size, embedding_dim=d_model)
        else: self.common_embedder = nn.Embedding(num_embeddings=vocab_size, embedding_dim=d_model)
        self.transformer = nn.Transformer(d_model=d_model, nhead=num_heads, num_encoder_layers=num_enc, 
                                          num_decoder_layers=num_dec, dim_feedforward=d_ff, dropout=dropout_rate, 
                                          activation=activation, batch_first=True)
        self.intermediate1 = nn.Linear(in_features=d_model, out_features=intermediate1)
        self.final = nn.Linear(in_features=intermediate1, out_features=vocab_size)

    def forward(self, batch_inp_sent, batch_inp_dec):
        # src_mask for padding and tgt mask for preventing the decoder from cheating
        src_mask, tgt_mask, tgt_pad_mask = self._create_masks(batch_inp_sent.clone().detach(), batch_inp_dec) 

        if not self.use_common_enc:
            src = self.enc_embedder(batch_inp_sent)
            tgt = self.dec_embedder(batch_inp_dec)
        else:
            src = self.common_embedder(batch_inp_sent)
            tgt = self.common_embedder(batch_inp_dec)

        transformer_op = self.transformer(src=src, tgt=tgt, src_key_padding_mask=src_mask, tgt_mask=tgt_mask, tgt_key_padding_mask=tgt_pad_mask)
        intermediate1 = nn.functional.leaky_relu(self.intermediate1(transformer_op), negative_slope=0.1)
        logits = self.final(intermediate1)

        return logits

    def predict(self, inp_sent, dec_prompt, stop_id, max_len=15):

        inp_sent = torch.tensor([inp_sent], dtype=torch.int32).to(device) # (batch_size, 1)
        dec_prompt = torch.tensor([dec_prompt], dtype=torch.int32).to(device) # (batch_size, 1)

        prev_output = -1
        pred_seq = []
        iter = 0

        while prev_output != stop_id and iter <= max_len:
            logits = self.forward(inp_sent, dec_prompt)
            pred = torch.argmax(logits, dim=-1) # (batch_size, seq_len)
            dec_prompt = torch.cat((dec_prompt, pred[:, -1].view(-1, 1)), dim=-1)
            prev_output = pred[0, -1].item()
            pred_seq.append(prev_output)
            iter += 1

        return pred_seq, None, None

    def _create_masks(self, src, tgt=None):
        src_mask = torch.eq(src.to('cpu'), torch.zeros(src.size(), dtype=torch.float64)).to(device)
        if tgt is not None: 
            tgt_mask = torch.eq(torch.triu(torch.ones(tgt.size(1), tgt.size(1)), diagonal=1), torch.ones(tgt.size(1), tgt.size(1))).to(device)
            tgt_pad_mask = torch.eq(tgt.to('cpu'), torch.zeros(tgt.size(), dtype=torch.float64)).to(device)
            return src_mask, tgt_mask, tgt_pad_mask
        return src_mask