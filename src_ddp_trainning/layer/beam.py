import torch
from torch.nn.functional import log_softmax
import numpy as np

class Beam:

    def __init__(
        self,
        beam_size=8,
        min_length=0,
        n_top=1,
        ranker=None,
        start_token_id=1,
        end_token_id=2,
    ):
        self.beam_size = beam_size
        self.min_length = min_length
        self.ranker = ranker

        self.end_token_id = end_token_id
        self.top_sentence_ended = False

        self.prev_ks = []
        self.next_ys = [
            torch.LongTensor(beam_size).fill_(start_token_id)
        ]  # remove padding

        self.current_scores = torch.FloatTensor(beam_size).zero_()
        self.all_scores = []

        # Time and k pair for finished.
        self.finished = []
        self.n_top = n_top

        self.ranker = ranker

    def advance(self, next_log_probs):
        # next_probs : beam_size X vocab_size

        vocabulary_size = next_log_probs.size(1)
        # current_beam_size = next_log_probs.size(0)

        current_length = len(self.next_ys)
        if current_length < self.min_length:
            for beam_index in range(len(next_log_probs)):
                next_log_probs[beam_index][self.end_token_id] = -1e10

        if len(self.prev_ks) > 0:
            beam_scores = next_log_probs + self.current_scores.unsqueeze(1).expand_as(
                next_log_probs
            )
            # Don't let EOS have children.
            last_y = self.next_ys[-1]
            for beam_index in range(last_y.size(0)):
                if last_y[beam_index] == self.end_token_id:
                    beam_scores[beam_index] = -1e10  # -1e20 raises error when executing
        else:
            beam_scores = next_log_probs[0]

        flat_beam_scores = beam_scores.view(-1)
        top_scores, top_score_ids = flat_beam_scores.topk(
            k=self.beam_size, dim=0, largest=True, sorted=True
        )

        self.current_scores = top_scores
        self.all_scores.append(self.current_scores)

        prev_k = top_score_ids // vocabulary_size  # (beam_size, )
        next_y = top_score_ids - prev_k * vocabulary_size  # (beam_size, )

        self.prev_ks.append(prev_k)
        self.next_ys.append(next_y)

        for beam_index, last_token_id in enumerate(next_y):

            if last_token_id == self.end_token_id:

                # skip scoring
                self.finished.append(
                    (self.current_scores[beam_index], len(self.next_ys) - 1, beam_index)
                )

        if next_y[0] == self.end_token_id:
            self.top_sentence_ended = True

    def get_current_state(self):
        "Get the outputs for the current timestep."
        return torch.stack(self.next_ys, dim=1)

    def get_current_origin(self):
        "Get the backpointers for the current timestep."
        return self.prev_ks[-1]

    def done(self):
        return self.top_sentence_ended and len(self.finished) >= self.n_top

    def get_hypothesis(self, timestep, k):
        hypothesis = []
        for j in range(len(self.prev_ks[:timestep]) - 1, -1, -1):
            hypothesis.append(self.next_ys[j + 1][k])
            # for RNN, [:, k, :], and for trnasformer, [k, :, :]
            k = self.prev_ks[j][k]

        return hypothesis[::-1]

    def sort_finished(self, minimum=None):
        if minimum is not None:
            i = 0
            # Add from beam until we have minimum outputs.
            while len(self.finished) < minimum:
                # global_scores = self.global_scorer.score(self, self.scores)
                # s = global_scores[i]
                s = self.current_scores[i]
                self.finished.append((s, len(self.next_ys) - 1, i))
                i += 1

        self.finished = sorted(self.finished, key=lambda a: a[0], reverse=True)
        scores = [sc for sc, _, _ in self.finished]
        ks = [(t, k) for _, t, k in self.finished]
        return scores, ks


def beamsearch(
    memory,
    model,
    device,
    beam_size=4,
    candidates=1,
    max_seq_length=128,
    sos_token=1,
    eos_token=2,
):
    # memory: Tx1xE
    model.eval()

    beam = Beam(
        beam_size=beam_size,
        min_length=0,
        n_top=candidates,
        ranker=None,
        start_token_id=sos_token,
        end_token_id=eos_token,
    )

    with torch.no_grad():
        #        memory = memory.repeat(1, beam_size, 1) # TxNxE
        memory = model.transformer.expand_memory(memory, beam_size)

        for _ in range(max_seq_length):

            tgt_inp = beam.get_current_state().transpose(0, 1).to(device)  # TxN
            decoder_outputs, memory = model.transformer.forward_decoder(tgt_inp, memory)

            log_prob = log_softmax(decoder_outputs[:, -1, :].squeeze(0), dim=-1)
            beam.advance(log_prob.cpu())

            if beam.done():
                break

        scores, ks = beam.sort_finished(minimum=1)

        hypothesises = []
        for i, (times, k) in enumerate(ks[:candidates]):
            hypothesis = beam.get_hypothesis(times, k)
            hypothesises.append(hypothesis)

    return [1] + [int(i) for i in hypothesises[0][:-1]]


def batch_translate_beam_search(
    img, model, beam_size=4, candidates=1, max_seq_length=128, sos_token=1, eos_token=2
):
    # img: NxCxHxW
    model.eval()
    device = img.device
    sents = []

    with torch.no_grad():
        src = model.cnn(img)
        print(src.shap)
        memories = model.transformer.forward_encoder(src)
        for i in range(src.size(0)):
            #            memory = memories[:,i,:].repeat(1, beam_size, 1) # TxNxE
            memory = model.transformer.get_memory(memories, i)
            sent = beamsearch(
                memory,
                model,
                device,
                beam_size,
                candidates,
                max_seq_length,
                sos_token,
                eos_token,
            )
            sents.append(sent)

    sents = np.asarray(sents)

    return sents