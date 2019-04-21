import torch
import torch.nn as nn
import torch.nn.utils.rnn as rnn_utils
from utils import to_var

class SentenceVAE(nn.Module):

    def __init__(self, vocab_size, embedding_size, rnn_type, hidden_size, word_dropout, embedding_dropout, latent_size,
                sos_idx, eos_idx, pad_idx, unk_idx, max_sequence_length, num_layers=1, bidirectional=False, classify_label=3):

        super().__init__()
        self.tensor = torch.cuda.FloatTensor if torch.cuda.is_available() else torch.Tensor

        self.max_sequence_length = max_sequence_length
        self.sos_idx = sos_idx
        self.eos_idx = eos_idx
        self.pad_idx = pad_idx
        self.unk_idx = unk_idx

        self.latent_size = latent_size

        self.rnn_type = rnn_type
        self.bidirectional = bidirectional
        self.num_layers = num_layers
        self.hidden_size = hidden_size

        self.embedding = nn.Embedding(vocab_size, embedding_size)
        self.word_dropout_rate = word_dropout
        self.embedding_dropout = nn.Dropout(p=embedding_dropout)
        if rnn_type == 'rnn':
            rnn = nn.RNN
        elif rnn_type == 'gru':
            rnn = nn.GRU
        # elif rnn_type == 'lstm':
        #     rnn = nn.LSTM
        else:
            raise ValueError()

        self.encoder_rnn = rnn(embedding_size, hidden_size, num_layers=num_layers, bidirectional=self.bidirectional, batch_first=True)
        self.decoder_rnn = rnn(embedding_size, hidden_size, num_layers=num_layers, bidirectional=self.bidirectional, batch_first=True)

        self.hidden_factor = (2 if bidirectional else 1) * num_layers

        self.hidden2mean = nn.Linear(hidden_size * self.hidden_factor, latent_size)
        self.hidden2logv = nn.Linear(hidden_size * self.hidden_factor, latent_size)
        self.latent2hidden = nn.Linear(latent_size, hidden_size * self.hidden_factor)
        self.outputs2vocab = nn.Linear(hidden_size * (2 if bidirectional else 1), vocab_size)
        self.label_pred = nn.Sequential([nn.Linear(hidden_size * self.hidden_factor, classify_label), nn.LogSoftmax(dim=-1)])

    def setenceEmbed(self, input_sequence):
        return self.embedding(input_sequence)

    def encoder(self, input_embedding, sorted_lengths, batch_size):

        packed_input = rnn_utils.pack_padded_sequence(input_embedding, sorted_lengths.data.tolist(), batch_first=True)

        _, hidden = self.encoder_rnn(packed_input)

        if self.bidirectional or self.num_layers > 1:
            # flatten hidden state
            hidden = hidden.view(batch_size, self.hidden_size * self.hidden_factor)
        else:
            hidden = hidden.squeeze()

        # REPARAMETERIZATION
        mean = self.hidden2mean(hidden)
        logv = self.hidden2logv(hidden)
        std = torch.exp(0.5 * logv)
        return mean, logv, std

    def decoder(self, input_sequence, input_embedding, sorted_lengths, sorted_idx, batch_size, hidden):
        if self.bidirectional or self.num_layers > 1:
            # unflatten hidden state
            hidden = hidden.view(self.hidden_factor, batch_size, self.hidden_size)
        else:
            hidden = hidden.unsqueeze(0)

        # decoder input
        if self.word_dropout_rate > 0:
            # randomly replace decoder input with <unk>
            prob = torch.rand(input_sequence.size())
            if torch.cuda.is_available():
                prob = prob.cuda()
            prob[(input_sequence.data - self.sos_idx) * (input_sequence.data - self.pad_idx) == 0] = 1
            decoder_input_sequence = input_sequence.clone()
            decoder_input_sequence[prob < self.word_dropout_rate] = self.unk_idx
            input_embedding = self.embedding(decoder_input_sequence)

        input_embedding = self.embedding_dropout(input_embedding)
        packed_input = rnn_utils.pack_padded_sequence(input_embedding, sorted_lengths.data.tolist(), batch_first=True)

        # decoder forward pass
        # rnn read one item one time
        outputs, _ = self.decoder_rnn(packed_input, hidden)

        # process outputs
        padded_outputs = rnn_utils.pad_packed_sequence(outputs, batch_first=True)[0]
        padded_outputs = padded_outputs.contiguous()
        _, reversed_idx = torch.sort(sorted_idx)
        padded_outputs = padded_outputs[reversed_idx]
        b, s, _ = padded_outputs.size()

        # project outputs to vocab
        logp = nn.functional.log_softmax(self.outputs2vocab(padded_outputs.view(-1, padded_outputs.size(2))), dim=-1)
        return logp

    def forward(self, input_sequence, length, classify=False):
        batch_size = input_sequence.size(0)
        #input_embedding, sorted_lengths, batch_size
        sorted_lengths, sorted_idx = torch.sort(length, descending=True)
        input_sequence = input_sequence[sorted_idx]
        input_embedding = self.embedding(input_sequence)

        mean, logv, std = self.encoder(input_embedding, sorted_lengths, batch_size)
        #input_sequence, input_embedding, sorted_lengths, sorted_idx, batch_size, z

        z = to_var(torch.randn([batch_size, self.latent_size]))
        z = z * std + mean
        hidden = self.latent2hidden(z)
        logp = self.decoder(input_sequence, input_embedding, sorted_lengths, sorted_idx, batch_size, hidden)

        if classify:
            #mse
            answer_predic = self.label_pred(hidden)
            #inverse encoder
            inv_sequence = torch.argmax(logp, dim=-1)
            inv_input_embedding = self.embedding(inv_sequence)
            inv_mean, inv_logv, inv_std = self.encoder(inv_input_embedding, sorted_lengths, batch_size)
            inv_z = to_var(torch.randn([batch_size, self.latent_size]))
            inv_z = inv_z * inv_std + inv_mean
            #inv
            inv_hidden = self.latent2hidden(inv_z)
            inv_error = torch.sum(answer_predic - self.label_pred(inv_hidden))
            return logp, mean, logv, z, answer_predic, inv_error

        else:
            return logp, mean, logv, z


    def inference(self, n=4, z=None):

        if z is None:
            batch_size = n
            z = to_var(torch.randn([batch_size, self.latent_size]))
        else:
            batch_size = z.size(0)

        hidden = self.latent2hidden(z)

        if self.bidirectional or self.num_layers > 1:
            # unflatten hidden state
            hidden = hidden.view(self.hidden_factor, batch_size, self.hidden_size)

        hidden = hidden.unsqueeze(0)

        # required for dynamic stopping of sentence generation
        sequence_idx = torch.arange(0, batch_size, out=self.tensor()).long() # all idx of batch
        sequence_running = torch.arange(0, batch_size, out=self.tensor()).long() # all idx of batch which are still generating
        sequence_mask = torch.ones(batch_size, out=self.tensor()).byte()

        running_seqs = torch.arange(0, batch_size, out=self.tensor()).long() # idx of still generating sequences with respect to current loop

        generations = self.tensor(batch_size, self.max_sequence_length).fill_(self.pad_idx).long()

        t=0
        while(t<self.max_sequence_length and len(running_seqs)>0):

            if t == 0:
                input_sequence = to_var(torch.Tensor(batch_size).fill_(self.sos_idx).long())

            input_sequence = input_sequence.unsqueeze(1)

            input_embedding = self.embedding(input_sequence)

            output, hidden = self.decoder_rnn(input_embedding, hidden)

            logits = self.outputs2vocab(output)

            input_sequence = self._sample(logits)

            # save next input
            generations = self._save_sample(generations, input_sequence, sequence_running, t)

            # update gloabl running sequence
            sequence_mask[sequence_running] = (input_sequence != self.eos_idx).data
            sequence_running = sequence_idx.masked_select(sequence_mask)

            # update local running sequences
            running_mask = (input_sequence != self.eos_idx).data
            running_seqs = running_seqs.masked_select(running_mask)

            # prune input and hidden state according to local update
            if len(running_seqs) > 0:
                input_sequence = input_sequence[running_seqs]
                hidden = hidden[:, running_seqs]

                running_seqs = torch.arange(0, len(running_seqs), out=self.tensor()).long()

            t += 1

        return generations, z

    def _sample(self, dist, mode='greedy'):

        if mode == 'greedy':
            _, sample = torch.topk(dist, 1, dim=-1)
        sample = sample.squeeze()

        return sample

    def _save_sample(self, save_to, sample, running_seqs, t):
        # select only still running
        running_latest = save_to[running_seqs]
        # update token at position t
        running_latest[:,t] = sample.data
        # save back
        save_to[running_seqs] = running_latest

        return save_to

class Intervention(nn.Module):
    def __init__(self, question_class, inter_class, model:SentenceVAE):
        super(Intervention, self).__init__()
        self.question_class = question_class
        self.inter_class = inter_class
        self.erase_embedding = nn.Embedding(inter_class+1, 1, padding_idx=0)
        self.add_embedding = nn.Embedding(inter_class + 1, 1, padding_idx=0)
        self.model = model
    def reconstruct_mean(self, source_sequence, source_sequence_len, edit_sequence, edit_sequence_len, interven_class):
        batch_size = edit_sequence.shape[0]
        # get encoder std
        sorted_lengths, sorted_idx = torch.sort(source_sequence_len, descendin=True)
        source_sequence = source_sequence[sorted_idx]
        source_sequence_embed = self.model.embedding(source_sequence)
        #input_embedding, sorted_lengths, batch_size
        _, _, std = self.model.encoder(source_sequence_embed, sorted_lengths, batch_size)
        z = to_var(torch.randn([batch_size, self.latent_size]))
        z = z * std + self.add_embedding(interven_class)

        sorted_lengths, sorted_idx = torch.sort(edit_sequence_len, descending=True)
        edit_sequence = edit_sequence[sorted_idx]
        edit_seq_embed = self.model.embedding(edit_sequence)
        hidden = self.model.latent2hidden(z)
        #input_sequence, input_embedding, sorted_lengths, sorted_idx, batch_size, hidden
        logp = self.model.decoder(edit_sequence, edit_seq_embed, sorted_lengths, sorted_idx, batch_size, hidden)
        # only calculate the BCELOSS
        return logp


    def reconstruct_erase_add(self, source_sequence, source_sequence_len, edit_sequence, edit_sequence_len, interven_class):
        batch_size = edit_sequence.shape[0]
        # get encoder std
        sorted_lengths, sorted_idx = torch.sort(source_sequence_len, descendin=True)
        source_sequence = source_sequence[sorted_idx]
        source_sequence_embed = self.model.embedding(source_sequence)
        # input_embedding, sorted_lengths, batch_size
        mean, _, std = self.model.encoder(source_sequence_embed, sorted_lengths, batch_size)
        z = to_var(torch.randn([batch_size, self.latent_size]))
        z = z * std + mean
        z = z * (1 - self.erase_embedding(interven_class)) + self.add_embedding(interven_class)

        sorted_lengths, sorted_idx = torch.sort(edit_sequence_len, descending=True)
        edit_sequence = edit_sequence[sorted_idx]
        edit_seq_embed = self.model.embedding(edit_sequence)
        hidden = self.model.latent2hidden(z)
        # input_sequence, input_embedding, sorted_lengths, sorted_idx, batch_size, hidden
        logp = self.model.decoder(edit_sequence, edit_seq_embed, sorted_lengths, sorted_idx, batch_size, hidden)
        # only calculate the BCELOSS
        return logp, hidden


    def forward(self, source_sequence, source_sequence_len, edit_sequence, edit_sequence_len, interven_class, recons_type="erase"):
        if recons_type == "erase":
            logp, hidden = self.reconstruct_erase_add(source_sequence, source_sequence_len, edit_sequence, edit_sequence_len, interven_class)
        else:
            logp, hidden = self.reconstruct_mean(source_sequence, source_sequence_len, edit_sequence, edit_sequence_len, interven_class)

        question_pred = self.model.label_pred(hidden)

        return logp, question_pred