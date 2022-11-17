import torch
import torch.nn.functional as F

from train.CL.gameplay import gameplay_fwpass

from math import exp, log

def _human_game_fwpass(model, sample):
    history, history_len = sample['history'], sample['history_len']
    visual_features = F.dropout(sample['visual_features'], p=0.5)
    encoder_hidden = model.encoder(history=history,  
        history_len=history_len, visual_features=visual_features)
    return encoder_hidden

def aux_loss(model, hum_sample, beta, adversarial=False, **kwargs):
    spatials = hum_sample['spatials']
    objects = hum_sample['objects']
    hum_hidden = _human_game_fwpass(model, hum_sample)
    logits = model.guesser(encoder_hidden=hum_hidden, 
        spatials=spatials, objects=objects, regress=False)
    loss = F.cross_entropy(logits, hum_sample['target_obj'])
    if adversarial:
        loss += error_gap_loss(model, hum_sample, beta, **kwargs)
    return loss

class GradReverse(torch.autograd.Function):

    @staticmethod
    def forward(ctx, x, beta, reverse=True):
        ctx.beta = beta
        ctx.reverse = reverse
        return x.view_as(x)

    @staticmethod
    def backward(ctx, grad_output):
        if ctx.reverse:
            # print('adv example reversed')
            return (grad_output * -ctx.beta), None, None
        else:
            # print('adv example not reversed')
            return (grad_output * ctx.beta), None, None

def grad_reverse(x, beta=1.0, reverse=True):
    return GradReverse.apply(x, beta, reverse)

def _reverse_and_guess(guesser, encoder_hidden, sample, beta=1.0):
    spatials = sample['spatials']
    objects = sample['objects']
    # reverse grads
    encoder_hidden = grad_reverse(encoder_hidden, beta=beta)
    # and guess
    logits = guesser(encoder_hidden=encoder_hidden, 
        spatials=spatials, objects=objects, regress=False)
    return logits

def _adversarial_fwpass(guesser, model, oracle, gen_sample, hum_sample,
    exp_config, word2i, beta=1.0):
    # compute logits on generated game using model.encoder and aux guesser
    gen_hidden = gameplay_fwpass(q_model=model, o_model=oracle, 
        inputs=gen_sample, exp_config=exp_config, word2i=word2i, 
        hidden_only=True)
    gen_logits = _reverse_and_guess(guesser, gen_hidden, gen_sample, beta=beta)
    # compute logits on human game using model.encoder and aux guesser
    hum_hidden = _human_game_fwpass(model, hum_sample)
    hum_logits = _reverse_and_guess(guesser, hum_hidden, hum_sample, beta=beta)
    return (gen_logits, gen_hidden), (hum_logits, hum_hidden)

def classic_adv(model, hum_sample, beta, aux_guesser, oracle, gen_sample, exp_config, word2i):
    (gen_logits, _), (hum_logits, _) = _adversarial_fwpass(aux_guesser, 
        model, oracle, gen_sample, hum_sample, exp_config, word2i, beta=beta)

def error_gap_loss(model, hum_sample, beta, progress, aux_guesser, oracle, gen_sample, exp_config, word2i):
    adv_suppression = (2.0 / (1. + exp(-10 * progress)) - 1)
    beta = beta * adv_suppression
    (gen_logits, _), (hum_logits, _) = _adversarial_fwpass(aux_guesser, 
        model, oracle, gen_sample, hum_sample, exp_config, word2i, beta=beta)
    gen_loss = F.cross_entropy(gen_logits, gen_sample['wrong_obj'])
    hum_loss = F.cross_entropy(hum_logits, hum_sample['target_obj'])
    # hum_loss = cust_cross_entropy(hum_logits, hum_sample['target_obj'])
    # gen_loss = stable_log1msoftmax(gen_logits, gen_sample['target_obj'])
    # print(gen_loss)
    return hum_loss + gen_loss

def cust_cross_entropy(logits, target):
    # 20 is hard coded!! max_num_objects in config
    ohot = F.one_hot(target, num_classes=20)
    x = F.log_softmax(logits, dim=-1) 
    return (x * ohot).sum(dim=-1).mean()

def stable_log1msoftmax(logits, target):
    # log(1 - softmax(x)), log prob wrong label
    # courtesy of https://stats.stackexchange.com/questions/469706/log1-softmaxx
    xtilde = torch.logsumexp(logits, dim=-1)
    xmax = torch.max(logits, dim=-1).values
    # 20 is hard coded!! max_num_objects in config
    ohot = F.one_hot(target, num_classes=20)
    x = (logits * ohot).sum(dim=-1)
    vk = (torch.log_softmax(logits, dim=-1) * ohot).sum(dim=-1)
    return (log1mexp(xtilde - x) + (xtilde - xmax) + vk).mean()

def log1mexp(a):
    # log(1 - exp(-a))
    # https://cran.r-project.org/web/packages/Rmpfr/vignettes/log1mexp-note.pdf
    sidx = (a > 0) & (a < 0.693)
    a[sidx] = torch.log(-torch.expm1(-a[sidx]))
    bidx = a > 0.693
    a[bidx] = torch.log1p(-torch.exp(-a[bidx]))
    zidx = (~sidx) & (~bidx) # a <= 0, log undefined
    a[zidx] = log(1e-8) # constant won't effect gradients
    return a


# def _only_guess(guesser, encoder_hidden, sample):
#     spatials = sample['spatials']
#     objects = sample['objects']
#     # only guess
#     logits = guesser(encoder_hidden=encoder_hidden, 
#         spatials=spatials, objects=objects, regress=False)
#     return logits

# def _simple_hidden_fwpass(model, oracle, gen_sample, hum_sample,
#     exp_config, word2i, guesser=None):
#     gen_hidden = gameplay_fwpass(q_model=model, o_model=oracle, 
#         inputs=gen_sample, exp_config=exp_config, word2i=word2i, 
#         hidden_only=True)
#     hum_hidden = _human_game_fwpass(model, hum_sample)
#     if guesser is None:
#         return gen_hidden, hum_hidden
#     else:
#         gen_logits = _only_guess(guesser, gen_hidden, gen_sample)
#         hum_logits = _only_guess(guesser, hum_hidden, hum_sample)
#         return gen_logits, hum_logits

# def hidden_energy_loss(guesser, model, oracle, gen_sample, hum_sample,
#     exp_config, word2i, progress):
#     """
#     Compute an energy statistic... similiar to MMD,
#     but does not require specifying a kernel and selecting 
#     hyperparameters. Does better than MMD in many practical
#     settings Atwell et al. (2022) <link>.

#     Accepts guesser for compatability.
#     """
#     beta = get_beta(progress)
#     gen_hidden, hum_hidden = _simple_hidden_fwpass(model, oracle,
#         gen_sample, hum_sample, exp_config, word2i)
#     n1 = gen_hidden.size(0)
#     n2 = hum_hidden.size(0)
#     estat = EnergyStatistic(n1, n2)
#     return beta * estat(gen_hidden.reshape(n1, -1), hum_hidden.reshape(n2, -1), 
#         ret_matrix=False)

# def logits_energy_loss(guesser, model, oracle, gen_sample, hum_sample,
#     exp_config, word2i, progress):
#     """
#     Energy using the logits as feature representations. Besides
#     the use of energy (in place of MMD), this is similiar to 
#     Rabanser et al. (2018) <link>.
#     """
#     beta = get_beta(progress)
#     gen_logits, hum_logits = _simple_hidden_fwpass(model, oracle,
#         gen_sample, hum_sample, exp_config, word2i, guesser=guesser)
#     n1 = gen_logits.size(0)
#     n2 = hum_logits.size(0)
#     estat = EnergyStatistic(n1, n2)
#     return beta * estat(gen_logits.reshape(n1, -1), hum_logits.reshape(n2, -1),
#         ret_matrix=False)

