"""Observe categorical learning without changing actions, losses or gradients."""
import torch

def categorical_metrics(distribution):
    probs=torch.stack([d.probs for d in distribution.distribution],dim=1)
    ent=torch.stack([d.entropy() for d in distribution.distribution],dim=1)
    top=probs[:,0].topk(2,dim=1).values
    modes=distribution.mode()
    result={'global_action_entropy':float(ent[:,0].mean()),
            'spatial_action_entropy':float(ent[:,1:].mean()),
            'global_action_max_probability':float(top[:,0].mean()),
            'global_action_probability_margin':float((top[:,0]-top[:,1]).mean())}
    for i in range(5):
        result['global_action_probability_'+str(i)]=float(probs[:,0,i].mean())
        result['global_action_mode_fraction_'+str(i)]=float((modes[:,0]==i).float().mean())
    return result

def gradient_metrics(policy_loss,auxiliary_loss,policy):
    """An audit at one minibatch; autograd.grad does not populate .grad."""
    features=list(policy.pi_features_extractor.parameters())
    head=list(policy.action_net.parameters())
    pi=torch.autograd.grad(policy_loss,features+head,retain_graph=True,allow_unused=True)
    aux=torch.autograd.grad(auxiliary_loss,features,retain_graph=True,allow_unused=True)
    def vector(gs,ps):
        return torch.cat([(torch.zeros_like(p) if g is None else g.detach()).flatten() for g,p in zip(gs,ps)])
    a=vector(pi[:len(features)],features);b=vector(aux,features);h=vector(pi[len(features):],head)
    an,bn=a.norm(),b.norm()
    return {'audit_policy_feature_gradient_norm':float(an),
            'audit_auxiliary_feature_gradient_norm':float(bn),
            'audit_policy_head_gradient_norm':float(h.norm()),
            'audit_policy_auxiliary_gradient_cosine':float(torch.dot(a,b)/(an*bn).clamp_min(1e-20))}

def post_update_metrics(policy,buffer,batch_size):
    """Full fresh rollout KL after the final optimizer step, no RNG draws."""
    total=0;kl=0.;abs_change=0.;maximum=0.
    count=buffer.buffer_size*buffer.n_envs
    with torch.no_grad():
        for start in range(0,count,batch_size):
            # PPO has already prepared the SB3 2.7 buffer with get().
            import numpy as np
            data=buffer._get_samples(np.arange(start,min(start+batch_size,count)))
            _,logp,_=policy.evaluate_actions(data.observations,data.actions)
            ratio=logp-data.old_log_prob;n=len(ratio)
            kl+=float((torch.expm1(ratio)-ratio).sum());abs_change+=float(ratio.abs().sum())
            maximum=max(maximum,float(ratio.abs().max()));total+=n
    return {'post_update_joint_kl':kl/total,'post_update_abs_logprob_change':abs_change/total,
            'post_update_max_abs_logprob_change':maximum}
