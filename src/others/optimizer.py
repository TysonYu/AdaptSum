from others.recadam import RecAdam, anneal_function
import torch
import torch.optim as optim

def build_optim(args, model, checkpoint, pretrained_model=None):
    """ Build optimizer """
    if args.recadam:
        print("Using RecAdam")
        no_decay = ["bias", "layer_norm.weight", "layernorm_embedding.weight"]
        optimizer_grouped_parameters = [
            {
                "params": [p for n, p in model.named_parameters() if
                           not any(nd in n for nd in no_decay) and args.model_type in n],
                "weight_decay": args.weight_decay,
                "anneal_w": args.anneal_w,
                "pretrain_params": [p_p for p_n, p_p in pretrained_model.named_parameters() if
                                    not any(nd in p_n for nd in no_decay) and args.model_type in p_n]
            },
            {
                "params": [p for n, p in model.named_parameters() if
                           not any(nd in n for nd in no_decay) and args.model_type not in n],
                "weight_decay": args.weight_decay,
                "anneal_w": 0.0,
                "pretrain_params": [p_p for p_n, p_p in pretrained_model.named_parameters() if
                                    not any(nd in p_n for nd in no_decay) and args.model_type not in p_n]
            },
            {
                "params": [p for n, p in model.named_parameters() if
                           any(nd in n for nd in no_decay) and args.model_type in n],
                "weight_decay": 0.0,
                "anneal_w": args.anneal_w,
                "pretrain_params": [p_p for p_n, p_p in pretrained_model.named_parameters() if
                                    any(nd in p_n for nd in no_decay) and args.model_type in p_n]
            },
            {
                "params": [p for n, p in model.named_parameters() if
                           any(nd in n for nd in no_decay) and args.model_type not in n],
                "weight_decay": 0.0,
                "anneal_w": 0.0,
                "pretrain_params": [p_p for p_n, p_p in pretrained_model.named_parameters() if
                                    any(nd in p_n for nd in no_decay) and args.model_type not in p_n]
            }
        ]
        optim = RecAdam(optimizer_grouped_parameters, lr=args.lr, eps=args.adam_epsilon, anneal_fun=args.anneal_fun, anneal_k=args.anneal_k, anneal_t0=args.anneal_t0, pretrain_cof=args.pretrain_cof)
    else:
        optim = Optimizer(
            args.optim, args.lr, args.max_grad_norm,
            beta1=args.beta1, beta2=args.beta2,
            decay_method=args.decay_method,
            warmup_steps=args.warmup_steps, model_size=args.enc_hidden_size)

        optim.set_parameters(list(model.named_parameters()))

        if args.train_from != '' and 'xsum' not in args.train_from:
            optim.optimizer.load_state_dict(checkpoint['optim'])
            if args.visible_gpu != '-1':
                for state in optim.optimizer.state.values():
                    for k, v in state.items():
                        if torch.is_tensor(v):
                            state[k] = v.cuda()

            if (optim.method == 'adam') and (len(optim.optimizer.state) < 1):
                raise RuntimeError(
                    "Error: loaded Adam optimizer from existing model" +
                    " but optimizer state is empty")


    return optim

class Optimizer(object):
    """
    Controller class for optimization. Mostly a thin
    wrapper for `optim`, but also useful for implementing
    rate scheduling beyond what is currently available.
    Also implements necessary methods for training RNNs such
    as grad manipulations.

    Args:
      method (:obj:`str`): one of [sgd, adagrad, adadelta, adam]
      lr (float): learning rate
      lr_decay (float, optional): learning rate decay multiplier
      start_decay_steps (int, optional): step to start learning rate decay
      beta1, beta2 (float, optional): parameters for adam
      adagrad_accum (float, optional): initialization parameter for adagrad
      decay_method (str, option): custom decay options
      warmup_steps (int, option): parameter for `noam` decay
      model_size (int, option): parameter for `noam` decay

    We use the default parameters for Adam that are suggested by
    the original paper https://arxiv.org/pdf/1412.6980.pdf
    These values are also used by other established implementations,
    e.g. https://www.tensorflow.org/api_docs/python/tf/train/AdamOptimizer
    https://keras.io/optimizers/
    Recently there are slightly different values used in the paper
    "Attention is all you need"
    https://arxiv.org/pdf/1706.03762.pdf, particularly the value beta2=0.98
    was used there however, beta2=0.999 is still arguably the more
    established value, so we use that here as well
    """

    def __init__(self, method, learning_rate, max_grad_norm,
                 lr_decay=1, start_decay_steps=None, decay_steps=None,
                 beta1=0.9, beta2=0.999,
                 adagrad_accum=0.0,
                 decay_method=None,
                 warmup_steps=4000,
                 model_size=None):
        self.last_ppl = None
        self.learning_rate = learning_rate
        self.original_lr = learning_rate
        self.max_grad_norm = max_grad_norm
        self.method = method
        self.lr_decay = lr_decay
        self.start_decay_steps = start_decay_steps
        self.decay_steps = decay_steps
        self.start_decay = False
        self._step = 0
        self.betas = [beta1, beta2]
        self.adagrad_accum = adagrad_accum
        self.decay_method = decay_method
        self.warmup_steps = warmup_steps
        self.model_size = model_size

    def set_parameters(self, params):
        """ ? """
        self.params = []
        self.sparse_params = []
        for k, p in params:
            if p.requires_grad:
                if self.method != 'sparseadam' or "embed" not in k:
                    self.params.append(p)
                else:
                    self.sparse_params.append(p)
        if self.method == 'sgd':
            self.optimizer = optim.SGD(self.params, lr=self.learning_rate)
        elif self.method == 'adagrad':
            self.optimizer = optim.Adagrad(self.params, lr=self.learning_rate)
            for group in self.optimizer.param_groups:
                for p in group['params']:
                    self.optimizer.state[p]['sum'] = self.optimizer\
                        .state[p]['sum'].fill_(self.adagrad_accum)
        elif self.method == 'adadelta':
            self.optimizer = optim.Adadelta(self.params, lr=self.learning_rate)
        elif self.method == 'adam':
            self.optimizer = optim.Adam(self.params, lr=self.learning_rate,
                                        betas=self.betas, eps=1e-9)
        elif self.method == 'sparseadam':
            self.optimizer = MultipleOptimizer(
                [optim.Adam(self.params, lr=self.learning_rate,
                            betas=self.betas, eps=1e-8),
                 optim.SparseAdam(self.sparse_params, lr=self.learning_rate,
                                  betas=self.betas, eps=1e-8)])
        else:
            raise RuntimeError("Invalid optim method: " + self.method)

    def _set_rate(self, learning_rate):
        self.learning_rate = learning_rate
        if self.method != 'sparseadam':
            self.optimizer.param_groups[0]['lr'] = self.learning_rate
        else:
            for op in self.optimizer.optimizers:
                op.param_groups[0]['lr'] = self.learning_rate

    def step(self):
        """Update the model parameters based on current gradients.

        Optionally, will employ gradient modification or update learning
        rate.
        """
        self._step += 1

        # Decay method used in tensor2tensor.
        if self.decay_method == "noam":
            self._set_rate(
                self.original_lr *
                ( self.model_size ** -0.5*min(self._step ** (-0.5),
                     self._step * self.warmup_steps**(-1.5))))
        else:
            if ((self.start_decay_steps is not None) and (
                     self._step >= self.start_decay_steps)):
                self.start_decay = True
            if self.start_decay:
                if ((self._step - self.start_decay_steps)
                   % self.decay_steps == 0):
                    self.learning_rate = self.learning_rate * self.lr_decay

        if self.method != 'sparseadam':
            self.optimizer.param_groups[0]['lr'] = self.learning_rate

        if self.max_grad_norm:
            clip_grad_norm_(self.params, self.max_grad_norm)
        self.optimizer.step()