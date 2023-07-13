'''
tight-binding theory:
'''

import torch
import numpy as np


class Tight_binding:

    def __init__(self, model_name, **kwargs):
        # Initialize the class
        if model_name == 'gcnn':
            self.model_num_input = 1
        if model_name == 'moment':
            self.model_num_input = 1
    
    def moment(self, bond_fea, crys_fea, **kwargs):
        
        idx = kwargs['batch_cif_ids']
        crystal_atom_idx = kwargs['crystal_atom_idx']
        
        d_cen = self.d_cen[idx]
        full_width = self.full_width[idx]

        full_width = self.full_width[idx]
        tabulated_d_cen_inf = self.tabulated_d_cen_inf[idx]
        tabulated_full_width_inf = self.tabulated_full_width_inf[idx]
        tabulated_mulliken = self.tabulated_mulliken[idx]
        tabulated_site_index = self.tabulated_site_index[idx]
        tabulated_v2dd = self.tabulated_v2dd[idx]
        tabulated_v2ds = self.tabulated_v2ds[idx]
        
        site_atom_idx = [crystal_atom_idx[i][int(tabulated_site_index[i])] for i in range(len(crystal_atom_idx))]
        
        #order = torch.argsort(torch.stack([nbr_fea_idx[i, :] for i in site_atom_idx]))
        
        zeta = torch.stack([bond_fea[i, :, :] for i in site_atom_idx])[:,:,:]
        #zeta = torch.take_along_dim(zeta, order, 1)
        #zeta = torch.stack([zeta[i][order[i]] for i in range(len(order))])
        
        crys_fea = torch.nn.functional.softplus(crys_fea[:,0,:])
        zeta = torch.nn.functional.softplus(zeta)
        
        m2 = torch.sum(tabulated_v2ds / zeta[:,:,0]
                     + tabulated_v2dd / zeta[:,:,0]**(10.0/7.0), dim=1)
        
        full_width_tinnet = (12*m2)**0.5
        d_cen_tinnet = crys_fea[:,1] * m2**0.5 * (tabulated_d_cen_inf / tabulated_full_width_inf
                                                  - crys_fea[:,0] * tabulated_mulliken)
        
        ans = torch.cat(((d_cen-d_cen_tinnet).view(-1, 1),
                         (full_width-full_width_tinnet).view(-1, 1)),1)
        
        return ans, bond_fea
    
    def gcnn(self, gcnnmodel_in, **kwargs):
        # Do nothing
        return gcnnmodel_in, gcnnmodel_in
