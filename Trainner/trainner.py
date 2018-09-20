from Predictor.Base import BaseTrainner
import torch as t
from tqdm import tqdm
import numpy as np



class Trainner(BaseTrainner):

    def __init__(self, args, vocab, model, loss_func, score_func, train_loader, dev_loader, use_multi_gpu=True):
        super(Trainner, self).__init__(args, vocab, model, loss_func, score_func, train_loader, dev_loader, use_multi_gpu=use_multi_gpu)

    def train(self):
        for epoch in range(self.args.epochs):
            self.train_epoch()
            self.global_epoch += 1
            self.reserve_topk_model(5)
        if self.summary_writer:
            self.summary_writer.close()
        print(f'Done')

    def train_epoch(self):
        for data in tqdm(self.train_loader, desc='train step'):
            train_loss = self.train_inference(data)
            train_loss.backward()
            if self.use_multi_gpu:
                self.model.module.encoder.embedding.weight.grad.data[0] = 0
            else:
                self.model.encoder.embedding.weight.grad.data[0] = 0

            t.nn.utils.clip_grad_norm_(parameters=self.model.parameters(), max_norm=5.0)
            self.optim.step_and_update_lr()

            if self.summary_writer:
                self.summary_writer.add_scalar('loss/train_loss', train_loss.item(), self.global_step)
                self.summary_writer.add_scalar('lr', self.optim.current_lr, self.global_step)
            self.global_step += 1

            if self.global_step % self.args.eval_every_step == 0:
                eval_score, eval_loss = self.evaluation()
                if self.global_step % self.args.save_every_step == 0:
                    self.save(eval_score, eval_loss)

    def evaluation(self):
        losses = []
        scores = []
        self.model.eval()
        with t.no_grad():
            for data in tqdm(self.dev_loader, desc='eval_step'):
                loss, score, pre, tru = self.eval_inference(data)
                losses.append(loss.item())
                scores.append(score)
        self.write_sample_result_text(pre, tru)
        eval_loss = np.mean(losses)
        eval_score = np.mean(scores)
        if self.summary_writer:
            self.summary_writer.add_scalar('loss/eval_loss', eval_loss, self.global_step)
            self.summary_writer.add_scalar('score/eval_score', eval_score, self.global_step)
            if self.use_multi_gpu:
                for i,v in self.model.module.name_parameters():
                    self.summary_writer.add_histogram(i.replace('.', '/'), v.clone().cpu().data.numpy(), self.global_step)
            else:
                for i,v in self.model.name_parameters():
                    self.summary_writer.add_histogram(i.replace('.', '/'), v.clone().cpu().data.numpy(), self.global_step)
        self.model.train()
        return eval_loss, eval_score

    def train_inference(self, data):
        context, title = [i.cuda() for i in data]
        self.optim.zero_grad()
        token_id, prob_vector = self.model(context, title)
        loss = self.loss_func(prob_vector, title)
        return loss

    def eval_inference(self, data):
        context, title = [i.cuda() for i in data]
        token_id, prob_vector = self.model(context, title)
        loss = self.loss_func(prob_vector, title)
        score = self.score_func(token_id, title)
        return loss, score, token_id, title

    def write_sample_result_text(self, pre, tru):
        token_list = pre.data.tolist()[0]
        title_list = tru.data.tolist()[0]
        word_list = [self.vocab.from_id_token(word) for word in token_list]
        title_list = [self.vocab.from_id_token(word) for word in title_list]
        word_pre = ' '.join(word_list) + '-    -' + ' '.join(title_list)
        self.summary_writer.add_text('pre', word_pre, global_step=self.global_step)

