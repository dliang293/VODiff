import math
import torch
from ldm.models.diffusion.gaussian_smoothing import GaussianSmoothing
from torch.nn import functional as F
from torchvision.utils import save_image
import numpy as np
import matplotlib.pyplot as plt



def get_all_attention(attn_maps_mid, attn_maps_up , attn_maps_down, res):

    result  = []
    
    for attn_map_integrated in attn_maps_up:
        if attn_map_integrated == []: continue
        attn_map = attn_map_integrated[0][0]
        b, i, j = attn_map.shape
        H = W = int(math.sqrt(i))
        # print(H)
        if H == res:
            result.append(attn_map.reshape(-1, res, res,attn_map.shape[-1] ))
    for attn_map_integrated in attn_maps_mid:

    # for attn_map_integrated in attn_maps_mid:
        attn_map = attn_map_integrated[0]
        b, i, j = attn_map.shape
        H = W = int(math.sqrt(i))
        # print(H)
        if (H==res):
            result.append(attn_map.reshape(-1, res, res,attn_map.shape[-1] ))
    # import pdb; pdb.set_trace()
    for attn_map_integrated in attn_maps_down:
        if attn_map_integrated == []: continue
        attn_map = attn_map_integrated[0][0]
        if attn_map == []: continue
        b, i, j = attn_map.shape
        H = W = int(math.sqrt(i))
        # print(H)
        if (H==res):
            result.append(attn_map.reshape(-1, res, res,attn_map.shape[-1] ))
    
    result = torch.cat(result, dim=0)
    result = result.sum(0) / result.shape[0]
    return result


def caculate_loss_att_fixed_cnt(attn_maps_mid, attn_maps_up, attn_maps_down, bboxes, object_positions, t, res=16, smooth_att = True,sigma=0.5,kernel_size=3):
    

    attn = get_all_attention(attn_maps_mid, attn_maps_up, attn_maps_down, res)
    
    obj_number = len(bboxes)
    total_loss = 0
   
    attn_text = attn[:, :, 1:-1]
    attn_text *= 100
    attn_text = torch.nn.functional.softmax(attn_text, dim=-1)
    current_res =  attn.shape[0]
    H = W = current_res
    min_all_inside = 1000
    max_outside = 0
    top_num = obj_number - 1
    
    
    for obj_idx in range(obj_number):

        loss_inter = 0.
        loss_inside = 0.
        loss_outside = 0.


        not_top = (obj_idx!=top_num)  
        
        for obj_position in object_positions[obj_idx]:  
            

            true_obj_position = obj_position - 1
            att_map_obj = attn_text[:,:, true_obj_position]
            if smooth_att:
                smoothing = GaussianSmoothing(channels=1, kernel_size=kernel_size, sigma=sigma, dim=2).cuda()
                input = F.pad(att_map_obj.unsqueeze(0).unsqueeze(0), (1, 1, 1, 1), mode='reflect')
                att_map_obj = smoothing(input).squeeze(0).squeeze(0)


            for obj_box in bboxes[obj_idx]:


                x_min, y_min, x_max, y_max = int(obj_box[0] * W), \
                    int(obj_box[1] * H), int(obj_box[2] * W), int(obj_box[3] * H)


                mask_bbox = torch.zeros(size=(H, W)).cuda() if torch.cuda.is_available() else torch.zeros(size=(H, W))  
                x_min, y_min, x_max, y_max = int(obj_box[0] * W), \
                    int(obj_box[1] * H), int(obj_box[2] * W), int(obj_box[3] * H)
                mask_bbox[y_min: y_max, x_min: x_max] = 1.  


                mask_seen = mask_bbox.clone()
                if (not_top):
                    for i in range(obj_idx,top_num):
                        next_layer = i + 1
                        next_box = bboxes[next_layer][0]
                        x_min_next, y_min_next, x_max_next, y_max_next = int(next_box[0] * W), \
                            int(next_box[1] * H), int(next_box[2] * W), int(next_box[3] * H)
                        mask_seen[y_min_next: y_max_next, x_min_next: x_max_next] = 0.

                mask_intersection_list = []
                for i in range (obj_idx,0,-1): 

                    front_box = bboxes[i-1][0]  
                    x_min_front, y_min_front, x_max_front, y_max_front = int(front_box[0] * W), \
                            int(front_box[1] * H), int(front_box[2] * W), int(front_box[3] * H)
                    
                    mask_front = torch.zeros(size=(H, W)).cuda() if torch.cuda.is_available() else torch.zeros(size=(H, W))
                    mask_front[y_min_front: y_max_front, x_min_front: x_max_front] = 1.

                    mask_intersection = torch.mul(mask_seen, mask_front)   
                    mask_intersection_list.append(mask_intersection)
                    
                if len(mask_intersection_list)>1:   
                    mask_intersection = torch.logical_or(mask_intersection_list[0],mask_intersection_list[1])

                    if mask_intersection.max()>0.01: 
                        min_inter = torch.mul(att_map_obj,mask_intersection).min()  
                        if min_inter < 0.1:
                            loss_inter = 4*(1. - min_inter) 
                            total_loss +=  loss_inter
                        elif min_inter < 0.2:
                            loss_inter = 2*(1. - min_inter) 
                            total_loss += loss_inter
                        elif t < 35:
                            loss_inter = (1. - min_inter) 
                            total_loss += loss_inter

                elif len(mask_intersection_list)==1:  
                    

                    mask_intersection = mask_intersection_list[0]

                    if mask_intersection.max()>0.01:  
                        min_inter = torch.mul(att_map_obj,mask_intersection).min()  
                        if min_inter < 0.1:
                            loss_inter = 4*(1. - min_inter)
                            total_loss +=  loss_inter
                        elif min_inter < 0.2:
                            loss_inter = 2*(1. - min_inter) 
                            total_loss += loss_inter
                        elif t < 35:
                            loss_inter = (1. - min_inter) 
                            total_loss += loss_inter

                else:
                    mask_intersection = torch.zeros(size=(H, W)).cuda() if torch.cuda.is_available() else torch.zeros(size=(H, W))
                    loss_inter = 0.

                max_inside = torch.mul(att_map_obj, mask_seen).max()

                if max_inside < 0.1:
                    loss_inside = 6*(1. - max_inside)
                    total_loss += loss_inside
                elif max_inside < 0.2:
                    loss_inside = 2*(1. - max_inside)
                    total_loss += loss_inside
                elif t < 35:
                    loss_inside = 1. - max_inside
                    total_loss += loss_inside


                mask_unseen = 1. - mask_seen 
                loss_outside = torch.mul(att_map_obj, mask_unseen).max()   

                total_loss += loss_outside
   
    return total_loss/obj_number, min_all_inside, max_outside