import numpy as np
import torch

class RBM(torch.nn.Module):
  def __init__(self, n_vis, n_hid, k = 5, lr = 1e-5):
    self.k = k
    self.lr = lr
    self.n_v = n_vis
    self.n_h = n_hid
    self.w = torch.tensor(np.random.normal(0.04, 0.01, (self.n_v, self.n_h)), dtype = torch.float32)
    self.a = torch.tensor(np.zeros(self.n_v), dtype = torch.float32)
    self.b = torch.tensor(np.zeros(self.n_h), dtype = torch.float32)
    self.h = torch.tensor(np.zeros(self.n_h), dtype = torch.float32)

  def p_h_given_v(self,b, v, w):
    return torch.sigmoid(self.b.T + v@self.w)

  def p_v_given_h(self,a, h, w):
    prob = torch.sigmoid(self.a.T + h @ self.w.T)
    return prob

  def positive_phase(self, b, w, v):
    batch_size = v.shape[0]
    vh_expecation_data = v.T @ self.p_h_given_v(b,v,w) / batch_size
    v_expectation_data = torch.mean(v, dim = 0)
    h_expectation_data = torch.mean(self.p_h_given_v(b,v,w), dim = 0)
    return vh_expecation_data, v_expectation_data, h_expectation_data

  def negative_phase(self, a, b, w, v, k):
    '''
    a - visible_bias = (V, )
    b - hidden_bias = (H, )
    v - visible units = (B, V)
    h - hidden units = (B, H)
    w - weights = (V, H)
    k - k_steps
    '''
    sampled_h = torch.bernoulli(self.p_h_given_v(b,v,w))
    for _ in range(self.k):
      sampled_v = torch.bernoulli(self.p_v_given_h(a,sampled_h,w))
      sampled_h = torch.bernoulli(self.p_h_given_v(b,sampled_v,w))
    batch_size = sampled_v.shape[0]
    vh_expectation_model = sampled_v.T @ sampled_h / batch_size
    v_expectation_model = sampled_v.mean(dim = 0)
    h_expectation_model = sampled_h.mean(dim = 0)
    return vh_expectation_model, v_expectation_model, h_expectation_model

  def forward(self, v):
    vh_data, v_data, h_data =  self.positive_phase(self.b,self.w,v)
    vh_model, v_model, h_model = self.negative_phase(self.a,self.b,self.w,v, self.k)
    return vh_data, vh_model, v_data, v_model, h_data, h_model

  def update(self, vh_data, vh_model, v_data, v_model, h_data, h_model):
    self.w+= self.lr * (vh_data - vh_model)
    self.a+= self.lr * (v_data - v_model)
    self.b+= self.lr * (h_data - h_model)

  def predict(self, v):
    h_prob = torch.sigmoid(self.b + v @ self.w)
    v_prob = torch.sigmoid(self.a + h_prob @ self.w.T)
    return v_prob

  def reconstruction_error(self, v_data, v_model):
    return torch.mean((v_data - v_model)**2)

  def err(self):
    return self.recon_err






