'''
collection of chemisorption models.

newns_anderson:
'''

import torch
import numpy as np


class Chemisorption:

    def __init__(self, model_name, **kwargs):
        # Initialize the class
        if model_name == 'gcnn':
            self.model_num_input = 1
        if model_name == 'gcnn_multitask':
            self.model_num_input = 2
    
    def gcnn_multitask(self, namodel_in, **kwargs):
        
        d_cen_gcnn = namodel_in[:,0]
        full_width_gcnn = torch.nn.functional.softplus(namodel_in[:,1])
        
        idx = kwargs['batch_cif_ids']
        
        d_cen = self.d_cen[idx]
        full_width = self.full_width[idx]
        
        ans = torch.cat(((d_cen-d_cen_gcnn).view(-1, 1),
                         (full_width-full_width_gcnn).view(-1, 1)),1)
        
        ans = ans.view(len(ans),1,-1)
        
        return ans, namodel_in
    
    def gcnn(self, gcnnmodel_in, **kwargs):
        # Do nothing
        return gcnnmodel_in, gcnnmodel_in
