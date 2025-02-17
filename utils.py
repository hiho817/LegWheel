import numpy as np
import pandas as pd

def create_command_csv_phi(theta_command, beta_command, file_name, transform=True): # 4*n, 4*n
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
            tramsform_r.append( np.hstack((np.linspace(0, phi_r[i, 0], 5000), phi_r[i, 0]*np.ones(2000))) )  # finally 4*m
            tramsform_l.append( np.hstack((np.linspace(0, phi_l[i, 0], 5000), phi_l[i, 0]*np.ones(2000))) )
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
    
    
def create_command_csv(theta_command, beta_command, file_name, transform=True): # 4*n, 4*n
    #### Tramsform (motor angle from 0 to initial pose) ####
    if transform:
        tramsform_theta = []
        tramsform_beta = []
        for i in range(4):
            tramsform_theta.append( np.hstack((np.linspace(np.deg2rad(17), theta_command[i, 0], 5000), theta_command[i, 0]*np.ones(2000))) )  # finally 4*m
            tramsform_beta.append( np.hstack((np.linspace(0, beta_command[i, 0], 5000), beta_command[i, 0]*np.ones(2000))) )
        theta_command = np.hstack((tramsform_theta, theta_command))
        beta_command = np.hstack((tramsform_beta, beta_command))

    # put into the format of motor command
    motor_command = np.empty((theta_command.shape[1], 2*theta_command.shape[0]))
    for i in range(4):
        motor_command[:, 2*i] = theta_command[i, :]
        if i in [1, 2]:
            motor_command[:, 2*i+1] = beta_command[i, :]
        else:
            motor_command[:, 2*i+1] = -beta_command[i, :]
            
    # transfer motor command to be continuous, i.e. [pi-d, -pi+d] -> [pi-d, pi+d]
    # threshold = np.pi/2
    # last = motor_command[0,:]
    # for angle in motor_command[1:]:
    #     for i in range(8):
    #         while np.abs(angle[i]-last[i]) > threshold: 
    #             angle[i] -= np.pi*np.sign(angle[i]-last[i]) 
    #     last = angle        

    # write motor commands into csv file #
    motor_command = np.hstack(( motor_command, -1*np.ones((motor_command.shape[0], 4)) ))    # add four column of -1 
    df = pd.DataFrame(motor_command)

    # 將 DataFrame 寫入 Excel 檔案
    df.to_csv(file_name + '.csv', index=False, header=False)