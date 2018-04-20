import edward as ed
import tensorflow as tf
import six

from edward.inferences.klpq import KLpq
from edward.models import RandomVariable
from edward.util import copy, get_descendants

class steinKLpq(KLpq):
  def __init__(self, kern,dkern=None,latent_vars=None, data=None,autoscale=False,n_sample=10):
      super(steinKLpq, self).__init__(latent_vars, data)
      self.kern=kern
      if(dkern==None):
          self.dkern,_ = tf.gradients(kern, tf.all_variable())
      else:
          self.dkern=dkern

      self.n_sample=n_sample      

  def _gram(self,kern,input):
    i=tf.stack([input]*self.n_sample)   
    j=tf.concat([input]*self.n_sample)   
    ii=tf.stack(i,j)
    ii=tf.reshape(ii,[n_samples,n_samples,2])
    return tf.map_fn(lambda x: kern(x[0],x[1]),ii)

  def build_loss_and_gradients(self, var_list):
    p_log_prob = [0.0] * self.n_samples
    q_log_prob = [0.0] * self.n_samples
    base_scope = tf.get_default_graph().unique_name("inference") + '/'
    for s in range(self.n_samples):
      # Form dictionary in order to replace conditioning on prior or
      # observed variable with conditioning on a specific value.
      scope = base_scope + tf.get_default_graph().unique_name("sample")
      dict_swap = {}
      for x, qx in six.iteritems(self.data):
        if isinstance(x, RandomVariable):
          if isinstance(qx, RandomVariable):
            qx_copy = copy(qx, scope=scope)
            dict_swap[x] = qx_copy.value()
          else:
            dict_swap[x] = qx

      for z, qz in six.iteritems(self.latent_vars):
        # Copy q(z) to obtain new set of posterior samples.
        qz_copy = copy(qz, scope=scope)
        dict_swap[z] = qz_copy.value()
        q_log_prob[s] += tf.reduce_sum(
            qz_copy.log_prob(tf.stop_gradient(dict_swap[z])))

      for z in six.iterkeys(self.latent_vars):
        z_copy = copy(z, dict_swap, scope=scope)
        p_log_prob[s] += tf.reduce_sum(z_copy.log_prob(dict_swap[z]))

      for x in six.iterkeys(self.data):
        if isinstance(x, RandomVariable):
          x_copy = copy(x, dict_swap, scope=scope)
          p_log_prob[s] += tf.reduce_sum(x_copy.log_prob(dict_swap[x]))

    p_log_prob = tf.stack(p_log_prob)
    q_log_prob = tf.stack(q_log_prob)
    reg_penalty = tf.reduce_sum(tf.losses.get_regularization_losses())

    if self.logging:
      tf.summary.scalar("loss/p_log_prob", tf.reduce_mean(p_log_prob),
                        collections=[self._summary_key])
      tf.summary.scalar("loss/q_log_prob", tf.reduce_mean(q_log_prob),
                        collections=[self._summary_key])
      tf.summary.scalar("loss/reg_penalty", reg_penalty,
                        collections=[self._summary_key])


    log_w = p_log_prob - q_log_prob
    log_w_norm = log_w - tf.reduce_logsumexp(log_w)
    w_norm = tf.exp(log_w_norm)
    loss = tf.reduce_sum(w_norm * log_w) - reg_penalty

    q_rvs = list(six.itervalues(self.latent_vars))
    q_vars = [v for v in var_list
              if len(get_descendants(tf.convert_to_tensor(v), q_rvs)) != 0]
    q_grads = tf.gradients(
        -(tf.reduce_sum(q_log_prob * tf.stop_gradient(w_norm)) - reg_penalty),
        q_vars)
    p_vars = [v for v in var_list if v not in q_vars]

    p_log_prob_grads = tf.gradients(p_log_prob, p_vars)
    dx=tf.reduce_sum( tf.matmul(self._gram(self.kern,p_log_prob),p_log_prob_grads) + self._gram(self.dkern,p_log_prob))
    p_grads = tf.gradients(-loss,p_vars)*dx

    grads_and_vars = list(zip(q_grads, q_vars)) + list(zip(p_grads, p_vars))
    return loss, grads_and_vars



