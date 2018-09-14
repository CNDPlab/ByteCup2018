import torch as t
import time
import os
from tensorboardX import SummaryWriter
import numpy as np
from tqdm import tqdm
import shutil
import ipdb


class ScheduledOptim(object):
    '''A simple wrapper class for learning rate scheduling'''

    def __init__(self, optimizer, d_model, n_warmup_steps, n_current_steps):
        self._optimizer = optimizer
        self.n_warmup_steps = n_warmup_steps
        self.n_current_steps = n_current_steps
        self.init_lr = np.power(d_model, -0.5)
        self.current_lr = 0

    def step_and_update_lr(self):
        "Step with the inner optimizer"
        self._update_learning_rate()
        self._optimizer.step()

    def zero_grad(self):
        "Zero out the gradients by the inner optimizer"
        self._optimizer.zero_grad()

    def _get_lr_scale(self):
        return np.min([
            np.power(self.n_current_steps, -0.5),
            np.power(self.n_warmup_steps, -1.5) * self.n_current_steps])

    def _update_learning_rate(self):
        ''' Learning rate scheduling per step '''

        self.n_current_steps += 1
        lr = self.init_lr * self._get_lr_scale()
        self.current_lr = lr
        for param_group in self._optimizer.param_groups:
            param_group['lr'] = lr


class Trainner(object):
    def __init__(self, args, vocab, model, loss_func, score_func, train_loader, dev_loader, use_multi_gpu=True):
        self.args = args
        self.vocab = vocab
        self.model = model
        self.loss_func = loss_func
        self.score_func = score_func
        self.train_loader = train_loader
        self.dev_loader = dev_loader
        self.paths = {
            'ckpt_root':self.args.ckpt_root,
            'exp_root': None,
            'tensorboard_root': None,
            'saved_model_root': None
        }
        self.summary_writer = None
        self.use_multi_gpu = use_multi_gpu

    def init_trainner(self, resume_from=None, use_summary_writer=False):
        self.global_step = 0
        self.global_epoch = 0
        self.init_time = time.strftime('%Y%m%d_%H%M%S')
        if resume_from is None:
            optimizer = t.optim.Adam([i for i in self.model.parameters() if i.requires_grad is True])
            self.optim = ScheduledOptim(optimizer, self.args.embedding_dim, self.args.warm_up_step, self.global_step)
        else:
            resume_path = os.path.join(self.paths['ckpt_root'], resume_from)
            self.load(resume_path)

        self.paths['exp_root'] = os.path.join(self.args.ckpt_root, self.init_time)
        self.paths['tensorboard_root'] = os.path.join(self.paths['exp_root'], 'logs')
        self.paths['saved_model_root'] = os.path.join(self.paths['exp_root'], 'saved_models')
        os.mkdir(self.paths['exp_root'])
        os.mkdir(self.paths['tensorboard_root'])
        os.mkdir(self.paths['saved_model_root'])
        if use_summary_writer:
            self.summary_writer = SummaryWriter(self.paths['tensorboard_root'])
            self.write_configs()

        if self.use_multi_gpu:
            self.model = t.nn.DataParallel(self.model).cuda()

    def write_configs(self):
        config_str = ''
        for k, v in self.args.__class__.__dict__.items():
            if not k.startswith('__'):
                config_str += '--'+ f'{k}:{getattr(args, k)}' + '\n'
        self.summary_writer.add_text('configs', config_str, 0)

    def save(self, score, loss):
        current_time = time.strftime('%Y_%m_%d_%H_%M_%S', time.localtime())
        file_name = current_time + 'T' + str(score) + '+' + str(loss)
        path = os.path.join(self.paths['exp_root'], file_name)
        if not os.path.exists(path):
            os.mkdir(path)
        t.save({
            'epoch' : self.global_epoch,
            'step': self.global_step,
            'optim': self.optim,
        }, os.path.join(path, 'trainner_state'))
        if self.use_multi_gpu:
            t.save(self.model.module.state_dict(), os.path.join(path, 'model'))
        else:
            t.save(self.model.state_dict(), os.path.join(path, 'model'))

    def load(self, load_from_exp):
        best_model_path = self.get_best_k_model_path(load_from_exp, k=1)[0]
        trainner_state = t.load(os.path.join(load_from_exp, best_model_path, 'trainner_state'))
        self.model.load_state_dict(t.load(os.path.join(load_from_exp, best_model_path, 'model')))
        self.global_step = trainner_state['epoch']
        self.global_epoch = trainner_state['step']
        self.optim = trainner_state['optim']

    def get_best_k_model_path(self, path, k=1):
        k_best_model_folder = sorted(os.listdir(path), key=lambda x: x.split('T')[1], reverse=True)[:k]
        return k_best_model_folder

    def reserve_topk_model(self, k=5):
        all_saved_models = os.listdir(self.paths['exp_root'])
        k_best_models = self.get_best_k_model_path(self.paths['exp_root'], k)
        drops = [i for i in all_saved_models if i not in k_best_models]
        for i in drops:
            shutil.rmtree(i)

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

            t.nn.utils.clip_grad_norm_(parameters=self.model.parameters, max_norm=5.0)
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
