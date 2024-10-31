import numpy as np
# import Contact_Map
from FittedCoefficient import *
import pandas as pd


# contact_map = Contact_Map.ContactMap()
# linkleg = contact_map.leg.linkleg
# r = linkleg.r
# outer_radius = linkleg.R + r
# radius = linkleg.R

# def get_foothold(theta, beta, contact_rim=0):
#     contact_map.mapping(theta, beta)
#     contact_rim = contact_map.rim if contact_map.rim in [2, 3, 4] else 2
    
#     if contact_rim in [1, 5]:    # upper rim
#         if contact_rim == 1:    # left
#             center_beta0 = U_l_poly[0](theta) +1j*U_l_poly[1](theta)    
#         else:  # right
#             center_beta0 = U_r_poly[0](theta) +1j*U_r_poly[1](theta)   
#         center_exp = center_beta0 * np.exp( 1j*beta )
#         return np.array([center_exp.real, center_exp.imag - outer_radius])
#     elif contact_rim in [2, 4]: # lower rim
#         if contact_rim == 2:    # left
#             center_beta0 = L_l_poly[0](theta) +1j*L_l_poly[1](theta)        
#         else:  # right
#             center_beta0 = L_r_poly[0](theta) +1j*L_r_poly[1](theta)     
#         center_exp = center_beta0 * np.exp( 1j*beta )
#         return np.array([center_exp.real, center_exp.imag - outer_radius])   
#     elif contact_rim == 3: # G
#         center_beta0 = 1j*G_poly[1](theta) # G
#         center_exp = center_beta0 * np.exp( 1j*beta )
#         return np.array([center_exp.real, center_exp.imag - r]) 
    
#     print("ERROR IN get_foothold")
#     return np.array([0, 0])   


def create_command_csv(theta_command, beta_command, file_name, transform=True): # 4*n, 4*n
    # Tramsform beta, theta to right, left motor angles
    theta_0 = np.array([-17, 17])*np.pi/180
    theta_beta = np.array([theta_command, beta_command]).reshape(2, -1)   # 2*(4*n)
    phi_r, phi_l = np.array([[1, 1], [-1, 1]]) @ theta_beta + theta_0.reshape(2, 1)

    phi_r = phi_r.reshape(4, -1)    # 4*n
    phi_l = phi_l.reshape(4, -1)    # 4*n

    #### Tramsform (motor angle from 0 to initial pose) ####
    if transform:
        tramsform_r = []
        tramsform_l = []
        for i in range(4):
            tramsform_r.append( np.hstack((np.linspace(0, phi_r[i, 0], 2000), phi_r[i, 0]*np.ones(500))) )  # finally 4*m
            tramsform_l.append( np.hstack((np.linspace(0, phi_l[i, 0], 2000), phi_l[i, 0]*np.ones(500))) )
        phi_r = np.hstack((tramsform_r, phi_r))
        phi_l = np.hstack((tramsform_l, phi_l))

    # put into the format of motor command
    motor_command = np.empty((phi_r.shape[1], 2*phi_r.shape[0]))
    for i in range(4):
        motor_command[:, 2*i] = phi_r[i, :]
        motor_command[:, 2*i+1] = phi_l[i, :]

    # write motor commands into csv file #
    motor_command = np.hstack(( motor_command, -1*np.ones((motor_command.shape[0], 4)) ))    # add four column of -1 
    df = pd.DataFrame(motor_command)

    # 將 DataFrame 寫入 Excel 檔案
    df.to_csv(file_name + '.csv', index=False, header=False)